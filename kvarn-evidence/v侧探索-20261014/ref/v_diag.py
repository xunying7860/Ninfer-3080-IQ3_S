#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：真实 V 上 2-bit 的损失到底是"天生量化不动"还是"我实现的方向选错了"。

真实数据画像（compare_real.py 打出来的）：层63 V 逐 token 范数 mean=81.0 max=164.7 (max/mean=2.03)
⇒ 逐 token 的尺度差异只有 2 倍量级，那 2-bit 差成这样就更可能是**分组方向**或我的 sinkhorn 复原环节的问题。
KVarN 的 V 路径是 [group=token, D] 上做 Sinkhorn 再按行 RTN；这里把几种口径放在一起量：

  per-token     : 每行=一个 token，对 256 个通道做 RTN（我/上游的 V 方向）
  per-channel   : 每行=一个通道，对 tokens 做 RTN（很多 KV 量化方案走的方向）
  *_clip        : 上加下用分位数取边界（= 上游 KVARN_RTN_QUANTILE 的作用）
  sinkhorn(kvarn): kv_codecs 里的 kvarn V 路径（含 16 轮 log 域 Sinkhorn + scale 吸收）
  fp16 floor    : 只做 fp16 舍入（这一档能到的上限，给上面几个数一个参照）

只看 V 的逐向量 cosine，不掺注意力（那是下一步的事）。
"""
import os
import sys
import glob
import torch

sys.path.insert(0, "/data/kvarn")
from kv_codecs import _rtn, _sinkhorn, _fp16, hadamard_256  # noqa: E402

KV_DIR = "/data/kvarn/kv"
D = 256


def cos(a, b):
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    return ((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)).mean().item()


def rtn_quant(x, bits, quantile=None, axis_rows="token"):
    """按行 RTN，行可以是 token（每行 256 通道）或 channel（每行 tokens）。"""
    t = x if axis_rows == "token" else x.transpose(-1, -2).contiguous()
    q, scale, zp = _rtn(t, bits, quantile)
    deq = scale.to(torch.float32) * q + zp.to(torch.float32)
    return deq if axis_rows == "token" else deq.transpose(-1, -2)


def sinkhorn_v(x, bits=2, iters=16, quantile=None, rotate=True):
    """kv_codecs 里 kvarn 的 V 路径单独跑一遍（tile=64 token 一组）。"""
    T = x.shape[1]
    ntile, G = T // 64, 64
    lead = list(x.shape[:-2])
    N = (1 if not lead else int(torch.tensor(lead).prod().item())) * ntile
    xr = hadamard_256(x).reshape(N, G, D) if rotate else x.reshape(N, G, D)
    balanced, s_col, s_row = _sinkhorn(xr, iters)
    q, scale, zp = _rtn(balanced, bits, quantile)
    deq = (scale.to(torch.float32) * q + zp.to(torch.float32)) * s_col * s_row
    deq = deq.reshape(*lead, T, D) if lead else deq.reshape(x.shape)
    return hadamard_256(deq) if rotate else deq


def sinkhorn_v_channel(x, bits=2, iters=16, quantile=None, rotate=True):
    """把 Sinkhorn 的"行"换成通道：tile 内 [64 token, 256 通道] 转成 [256 通道, 64 token] 再走
    同一套 log 域 Sinkhorn + 按行 RTN。动机是刚才量的 per-channel(0.9397) 明显好过
    per-token(0.8253)，而上游 store 代码里 V 的方向是有开关的，值得一并量掉。"""
    T = x.shape[1]
    ntile, G = T // 64, 64
    lead = list(x.shape[:-2])
    N = (1 if not lead else int(torch.tensor(lead).prod().item())) * ntile
    xr = hadamard_256(x).reshape(N, G, D) if rotate else x.reshape(N, G, D)
    tt = xr.transpose(-1, -2).contiguous()                 # [N, D, G] 行=通道
    balanced, s_col, s_row = _sinkhorn(tt, iters)
    q, scale, zp = _rtn(balanced, bits, quantile)
    deq = (scale.to(torch.float32) * q + zp.to(torch.float32)) * s_col * s_row
    deq = deq.transpose(-1, -2).reshape(N, G, D)
    deq = deq.reshape(x.shape)
    return hadamard_256(deq) if rotate else deq


def main():
    layers = sorted({int(os.path.basename(f)[5:7]) for f in glob.glob(KV_DIR + "/layer*_v_proj.pt")})
    print("层:", layers, flush=True)
    agg = {}
    for idx in layers:
        v = torch.load(os.path.join(KV_DIR, "layer%02d_v_proj.pt" % idx), map_location="cpu").float()
        v = v.reshape(1, v.shape[1], -1, D).permute(0, 2, 1, 3).contiguous().squeeze(0)  # [heads, T, D]
        print("\n层 %d: V shape=%s" % (idx, tuple(v.shape)), flush=True)
        # 通道方向的离群程度（给 per-channel 方案一个预期）
        per_ch = (v.abs().amax(dim=1) / (v.abs().mean(dim=1) + 1e-9))
        per_tok = (v.abs().amax(dim=-1) / (v.abs().mean(dim=-1) + 1e-9))
        print("  通道方向 max/mean: mean=%.2f max=%.2f | token 方向 max/mean: mean=%.2f max=%.2f"
              % (per_ch.mean().item(), per_ch.max().item(), per_tok.mean().item(), per_tok.max().item()), flush=True)
        cases = [
            ("fp16 舍入(上限参照)", _fp16(v).float()),
            ("2bit per-token", rtn_quant(v, 2)),
            ("2bit per-token+clip0.5%", rtn_quant(v, 2, 0.005)),
            ("2bit per-channel", rtn_quant(v, 2, axis_rows="channel")),
            ("2bit per-channel+clip0.5%", rtn_quant(v, 2, 0.005, "channel")),
            ("4bit per-token", rtn_quant(v, 4)),
            ("4bit per-channel", rtn_quant(v, 4, axis_rows="channel")),
            ("kvarn V 路径(sinkhorn 2bit, 带旋转)", sinkhorn_v(v, 2)),
            ("kvarn V 路径(2bit, 不带旋转)", sinkhorn_v(v, 2, rotate=False)),
            ("sinkhorn 4bit 参照", sinkhorn_v(v, 4)),
            ("kvarn V 通道为行 2bit", sinkhorn_v_channel(v, 2)),
            ("kvarn V 通道为行 2bit+clip0.5%", sinkhorn_v_channel(v, 2, quantile=0.005)),
            ("kvarn V 通道为行 2bit+clip1%", sinkhorn_v_channel(v, 2, quantile=0.01)),
            ("kvarn V 通道为行 4bit", sinkhorn_v_channel(v, 4)),
        ]
        for name, deq in cases:
            c = cos(v, deq)
            agg.setdefault(name, []).append(c)
            print("  %-34s V cos = %.6f" % (name, c), flush=True)
    print("\n=== 跨层平均 ===")
    for name, vals in agg.items():
        print("%-34s %.6f" % (name, sum(vals) / len(vals)))


if __name__ == "__main__":
    main()
