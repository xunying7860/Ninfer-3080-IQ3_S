#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 0 · 误差归因实验（doc 07 §7.1）：固定一侧 fp16、只压另一侧，判定 K 侧与 V 侧谁更敏感。

设计（诊断实验，不是档位）：
  baseline_fp16    K/V 都 fp16                     —— 上限参照
  rk4v4_post       K/V 都现生产 rk4v4              —— 判据基准（语料 A 应为 7.9159 / B 8.4558）
  k4v2_g64_OLD     kv_codecs G64 + iters16 的上游口径 —— **口径标定臂**（A 应 ≈+2.553%、B ≈+2.693%）
  K4only_g128      K=kvarn 4bit(G128)，V=fp16     —— K 侧真实代价
  V2only_g128      V=kvarn 2bit(G128)，K=fp16     —— V 侧真实代价
  K2only_g128      K=kvarn 2bit(G128)，V=fp16     —— 位宽对照（把 K 压到与 V 同样的 2bit）
  V4only_g128      V=kvarn 4bit(G128)，K=fp16     —— 位宽对照
  V2conly_g128     V 通道为行 2bit，K=fp16        —— V 分组方向对照（§17/§18 的争议点）
  K4V2_g128        K4 + V2（目标档位本身）
  K4V2c_g128       K4 + V 通道为行 2bit

口径：真实权重 Qwen3.8-27B-3Bit-GSQ，全部 16 个 self_attn 层的 k_proj/v_proj 输出被 fake-quant 替换；
      K 为 **post-RoPE**（先加 RoPE→量化→反旋转，引擎能做的就是这件事）；V 不含 RoPE；
      同语料同位置同种子，PPL = next-token NLL 的 exp（lm_head 一次性反量化到 GPU）；
      同一轮里顺带量：K/V 逐向量 cosine（post-RoPE 真实张量）+ 末层 attention 输出 cosine。

判读（doc 07）：只压 V 的代价 > 只压 K 的代价 ⇒ 余弦论据成立、K4V2 判死；反之 K4V2 的 +2.5% 不是位宽的锅。
"""
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/kvarn")
sys.path.insert(0, "/data/kvarn/g128")

from ppl_run import load_model, lm_head_weight, build_inputs_embeds, D, N_KV  # noqa: E402
from kv_codecs import kvarn_fakequant, rk4v4_fakequant  # noqa: E402
import g128_codecs as G  # noqa: E402

# 臂 -> dict(k=, v=, post=)
ARMS = {
    "baseline_fp16":  dict(k=None, v=None, post=False),
    "rk4v4_post":     dict(k="rk", v="rk", post=True),
    "k4v2_g64_OLD":   dict(k="old", v="old", post=True),
    "K4only_g128":    dict(k="g128k4", v=None, post=True),
    "V2only_g128":    dict(k=None, v="g128v2", post=True),
    "K2only_g128":    dict(k="g128k2", v=None, post=True),
    "V4only_g128":    dict(k=None, v="g128v4", post=True),
    "V2conly_g128":   dict(k=None, v="g128v2c", post=True),
    "K4V2_g128":      dict(k="g128k4", v="g128v2", post=True),
    "K4V2c_g128":     dict(k="g128k4", v="g128v2c", post=True),
    # 同字节（210）与次选（242）的对照：归因出 V 是短板后，把位子从 K 挪给 V 到底换不换得回质量
    "K3only_g128":    dict(k="g128k3", v=None, post=True),
    "V3only_g128":    dict(k=None, v="g128v3", post=True),
    "K3V3_g128":      dict(k="g128k3", v="g128v3", post=True),      # 210 B，同字节不极端
    "K2V4_g128":      dict(k="g128k2", v="g128v4", post=True),      # 210 B，另一侧极端
    "K3V4_g128":      dict(k="g128k3", v="g128v4", post=True),      # 242 B，doc 07 次选
    "K4V3_g128":      dict(k="g128k4", v="g128v3", post=True),      # 242 B，doc 07 次选
    "K4V4_g128":      dict(k="g128k4", v="g128v4", post=True),      # 274 B，doc 07 §7.3 零代价点
}
CAP_LAYERS = []          # 运行期填：末个 self_attn 层的下标（attention cosine 用）
ARM = "baseline_fp16"
RS = {"cos": None, "sin": None}
COS = {}                 # arm -> {"k": [...], "v": [...]} 逐层逐向量 cosine
CAP = {}                 # arm -> {layer: {"k","v","q"}}
BPT = {}
_DBG = set()


def _rh(t):
    h = t.shape[-1] // 2
    return torch.cat((-t[..., h:], t[..., :h]), dim=-1)


def rope_on(x, cos, sin):
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    return torch.cat([xr * cos + _rh(xr) * sin, xp], dim=-1)


def rope_off(x, cos, sin):
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    return torch.cat([xr * cos - _rh(xr) * sin, xp], dim=-1)


def vec_cos(a, b):
    """逐 (head, token) 256 维向量的 cosine，取平均。"""
    a2 = a.reshape(-1, D)
    b2 = b.reshape(-1, D)
    return ((a2 * b2).sum(-1) / (a2.norm(dim=-1) * b2.norm(dim=-1) + 1e-12)).mean().item()


def quantize(part, x):
    """x: [4, T, 256] float32(cpu)。按当前臂把该侧换成 fake-quant 后的版本。"""
    spec = ARMS[ARM]["k" if part == "k" else "v"]
    if spec is None:
        return x
    if spec == "rk":
        kh, vh, bpe = rk4v4_fakequant(x, x)
        BPT[ARM] = bpe
        return kh if part == "k" else vh
    if spec == "old":                       # 口径标定臂：完全照旧档案的 G64 + iters16
        kh, vh, bpe = kvarn_fakequant(x, x, tile_tokens=64, bits_k=4, bits_v=2, iters=16)
        BPT[ARM] = bpe
        return kh if part == "k" else vh
    if spec.startswith("g128k"):
        bits = int(spec[5:])
        return G.g128_quant_k(x, bits, iters=8)
    if spec.startswith("g128v"):
        body = spec[5:]
        mode = "channel" if body.endswith("c") else "token"
        bits = int(body[0])
        return G.g128_quant_v(x, bits, iters=8, mode=mode)
    if spec == "fp16":
        return x
    raise SystemExit("未知 spec " + spec)


def arm_bytes(name):
    """诊断口径的字节/token·KVhead：fp16 侧 = 512，量化侧 = 256*bits/8 + 18。"""
    a = ARMS[name]
    tot = 0.0
    for side in ("k", "v"):
        s = a[side]
        if s is None:
            tot += 512.0
        elif s == "rk":
            tot += 136.0 if side == "k" else 144.0
        elif s == "old":
            tot += 146.0 if side == "k" else 82.0      # 旧档案 §30 的 K4V2 拆分
        elif s.startswith("g128k") or s.startswith("g128v"):
            bits = int("".join(ch for i, ch in enumerate(s) if i >= 5 and ch.isdigit()) or "0")
            tot += 256.0 * bits / 8.0 + G.SIDE_ONE_G128
    return tot


def make_hook(part, layer_idx):
    def f(_m, _i, out):
        T = out.shape[1]
        x = out.reshape(1, T, N_KV, D).permute(0, 2, 1, 3).contiguous().squeeze(0).float().cpu()  # [4,T,256]
        post = ARMS[ARM]["post"]
        c = s = None
        if post and part == "k" and RS["cos"] is not None:
            c = RS["cos"][:T].unsqueeze(0)
            s = RS["sin"][:T].unsqueeze(0)
            x = rope_on(x, c, s)
        if ARM not in _DBG:
            _DBG.add(ARM)
            print("   [dbg] 臂=%s part=%s post=%s T=%d | 首调用张量 通道方向 max/mean=%.2f | token 方向 max/mean=%.2f"
                  % (ARM, part, post, T,
                     (x.abs().amax(dim=1) / (x.abs().mean(dim=1) + 1e-9)).mean().item(),
                     (x.abs().amax(dim=-1) / (x.abs().mean(dim=-1) + 1e-9)).mean().item()), flush=True)
        q = quantize(part, x)
        if part == "k":
            COS.setdefault(ARM, {}).setdefault("k", []).append(vec_cos(x, q))
        else:
            COS.setdefault(ARM, {}).setdefault("v", []).append(vec_cos(x, q))
        if layer_idx in CAP_LAYERS:
            rec = CAP.setdefault(ARM, {}).setdefault(layer_idx, {"k": None, "v": None, "q": None,
                                                                "khat": None, "vhat": None})
            rec[part] = x.clone()
            rec[part + "hat"] = q.clone()
        if post and part == "k" and c is not None:
            q = rope_off(q, c, s)
        return q.to(device=out.device, dtype=out.dtype).permute(1, 0, 2).reshape(1, T, N_KV * D)
    return f


def attn_cos(ref, cur, n_q=512, chunk=128):
    """末层 attention 输出 cosine（baseline 为参照）：最后 n_q 个 query 位置，带完整因果历史。"""
    q0, k0, v0 = ref["q"], ref["khat"], ref["vhat"]
    q1, k1, v1 = cur["q"], cur["khat"], cur["vhat"]
    T = q0.shape[1]
    st = max(0, T - n_q)
    rep = q0.shape[0] // k0.shape[0]
    k0 = k0.repeat_interleave(rep, 0)
    v0 = v0.repeat_interleave(rep, 0)
    k1 = k1.repeat_interleave(rep, 0)
    v1 = v1.repeat_interleave(rep, 0)
    sc = 1.0 / math.sqrt(D)
    keys = torch.arange(T)
    sims = []
    for s in range(st, T, chunk):
        e = min(s + chunk, T)
        mask = torch.arange(s, e)[:, None] >= keys[None, :]
        a = ((q0[:, s:e] @ k0.transpose(-1, -2)) * sc).masked_fill(~mask, float("-inf")).softmax(-1) @ v0
        b = ((q1[:, s:e] @ k1.transpose(-1, -2)) * sc).masked_fill(~mask, float("-inf")).softmax(-1) @ v1
        sims.append(F.cosine_similarity(a.reshape(-1, D), b.reshape(-1, D), dim=-1))
    return torch.cat(sims).mean().item()


def run_arm(m, layers, ids, W):
    global ARM
    handles = []
    for idx, L in enumerate(layers):
        at = getattr(L, "self_attn", None)
        if at is None:
            continue
        for part, nm in (("k", "k_proj"), ("v", "v_proj")):
            handles.append(getattr(at, nm).register_forward_hook(make_hook(part, idx)))
        if idx in CAP_LAYERS:
            handles.append(at.q_proj.register_forward_hook(make_qhook(idx)))
    emb = build_inputs_embeds(ids)
    pos = torch.arange(emb.shape[1], device=emb.device)
    pos3 = pos.view(1, 1, -1).expand(3, 1, -1).contiguous()
    t0 = time.time()
    with torch.no_grad():
        hs = m.model.language_model(inputs_embeds=emb, position_ids=pos3).last_hidden_state[0]
        nll, n = 0.0, 0
        for st in range(0, hs.shape[0] - 1, 256):
            e = min(st + 256, hs.shape[0] - 1)
            lg = hs[st:e].to(torch.float16) @ W.t()
            tgt = torch.tensor(ids[st + 1:e + 1], device=lg.device)
            nll += (torch.logsumexp(lg.float(), -1) - lg.gather(1, tgt[:, None]).squeeze(1).float()).sum().item()
            n += tgt.numel()
    for h in handles:
        h.remove()
    return nll / n, math.exp(nll / n), time.time() - t0


def make_qhook(idx):
    def f(_m, _i, out):
        T = out.shape[1]
        x = out.reshape(1, T, -1, D).permute(0, 2, 1, 3).contiguous().squeeze(0).float().cpu()
        CAP.setdefault(ARM, {}).setdefault(idx, {"k": None, "v": None, "khat": None, "vhat": None})
        CAP[ARM][idx]["q"] = x
        return out
    return f


def main():
    global ARM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/data/models/Qwen3.8-27B-3Bit-GSQ", trust_remote_code=True)
    DOC = os.environ.get("PPL_DOC", "/data/kvarn/ppl_doc.txt")
    MAX_TOK = int(os.environ.get("PPL_TOK", "2944"))
    CH = int(os.environ.get("PPL_CHARS", "20000"))
    parts = [open(p.strip(), encoding="utf-8", errors="ignore").read() for p in DOC.split(",") if p.strip()]
    text = re.sub(r"\s+", " ", "\n".join(parts))[:CH]
    ids = tok(text, add_special_tokens=False)["input_ids"][:MAX_TOK]
    ids = ids[:(len(ids) // 128) * 128]
    assert len(ids) % 128 == 0
    print("语料 %s → %d token（%d 字符）" % (DOC, len(ids), len(text)), flush=True)

    m, layers = load_model()
    W = lm_head_weight(m)
    self_idx = [i for i, L in enumerate(layers) if getattr(L, "self_attn", None) is not None]
    CAP_LAYERS.append(self_idx[-1])
    print("self_attn 层数=%d，attention cosine 取末层 idx=%d" % (len(self_idx), self_idx[-1]), flush=True)

    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding as _RE
        tc2 = getattr(m.config, "text_config", m.config)
        rot = _RE(tc2).to(device="cuda:0", dtype=torch.float32)
        p = torch.arange(len(ids), device="cuda:0").view(1, 1, -1).expand(3, 1, -1).contiguous()
        with torch.no_grad():
            o = rot(torch.zeros(1, len(ids), 1, device="cuda:0", dtype=torch.float16), p)
        RS["cos"], RS["sin"] = o[0][0].float().cpu(), o[1][0].float().cpu()
        print("离线 RoPE: cos %s 均值 %.4f 标准差 %.4f" % (tuple(RS["cos"].shape), RS["cos"].mean().item(),
                                                           RS["cos"].std().item()), flush=True)
    except Exception as e:
        raise SystemExit("RoPE 没算成，post 口径做不了: %s %s" % (type(e).__name__, str(e)[:200]))

    only = os.environ.get("S0_ARMS")
    arms = only.split(",") if only else list(ARMS)
    out = {}
    ref = None
    acos = {}
    for arm in arms:
        ARM = arm
        t0 = time.time()
        try:
            nll, ppl, ft = run_arm(m, layers, ids, W)
        except Exception as e:
            print("%-14s 前向失败: %s %s" % (arm, type(e).__name__, str(e)[:200]), flush=True)
            continue
        out[arm] = (nll, ppl)
        if arm == "baseline_fp16":
            ref = CAP.get(arm)
        try:
            line = ("%-14s mean NLL=%.6f PPL=%.4f 字节≈%.0f 用时%.0fs"
                    % (arm, nll, ppl, arm_bytes(arm), time.time() - t0))
            if arm in COS:
                ck = sum(COS[arm]["k"]) / len(COS[arm]["k"]) if COS[arm].get("k") else float("nan")
                cv = sum(COS[arm]["v"]) / len(COS[arm]["v"]) if COS[arm].get("v") else float("nan")
                line += " | K cos=%.6f V cos=%.6f" % (ck, cv)
            if arm != "baseline_fp16" and ref and CAP.get(arm) and CAP_LAYERS[0] in ref:
                L = CAP_LAYERS[0]
                acos[arm] = attn_cos(ref[L], CAP[arm][L])
                line += " | attn cos=%.6f" % acos[arm]
            print(line, flush=True)
        except Exception as e:
            print("%-14s 报表失败(不影响上面数字): %s %s" % (arm, type(e).__name__, str(e)[:160]), flush=True)
        if ARM != "baseline_fp16":
            CAP.pop(arm, None)

    base = out["rk4v4_post"][1] if "rk4v4_post" in out else None
    print("\n=== 逐层 K/V 逐向量 cosine（真实张量，post-RoPE 口径，16 层平均）===", flush=True)
    for a in arms:
        if a in COS:
            ck = COS[a].get("k") or [float("nan")]
            cv = COS[a].get("v") or [float("nan")]
            print("  %-14s K cos=%.6f (n=%d) V cos=%.6f (n=%d)"
                  % (a, sum(ck) / len(ck), len(ck), sum(cv) / len(cv), len(cv)), flush=True)

    print("\n=== 归因表（判据：doc 07 §7.1）===", flush=True)
    if base:
        for a in arms:
            if a in out:
                print("  %-14s PPL=%.4f vs rk4v4 %+.3f%%" % (a, out[a][1], 100.0 * (out[a][1] / base - 1.0)), flush=True)
    if "K4only_g128" in out and "V2only_g128" in out and base:
        dk = 100.0 * (out["K4only_g128"][1] / base - 1.0)
        dv = 100.0 * (out["V2only_g128"][1] / base - 1.0)
        print("\n  只压 K(4bit) 代价 = %+.3f%% | 只压 V(2bit) 代价 = %+.3f%% ⇒ %s"
              % (dk, dv, "V 更敏感（余弦论据成立）" if dv > dk else "K 更敏感（论文论据成立）"), flush=True)
    if "K2only_g128" in out and "V2only_g128" in out and base:
        dk2 = 100.0 * (out["K2only_g128"][1] / base - 1.0)
        dv2 = 100.0 * (out["V2only_g128"][1] / base - 1.0)
        print("  同 2bit 对比：只压 K(2bit) = %+.3f%% vs 只压 V(2bit) = %+.3f%% ⇒ %s"
              % (dk2, dv2, "V 更敏感" if dv2 > dk2 else "K 更敏感"), flush=True)
    print("S0_DONE", flush=True)


if __name__ == "__main__":
    main()
