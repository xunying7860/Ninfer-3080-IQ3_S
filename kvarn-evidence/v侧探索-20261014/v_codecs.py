#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V 侧探索编解码器（纯 PyTorch CPU）。

基线（与 /data/kvarn/g128/g128_codecs.py 的 g128_quant_v mode='token' 逐行等价，未改上游数学）：
  旋转 H256 → 切 128×128 slice（行=token，列=该 slice 的 128 维）→ Sinkhorn(8) → 逐行非对称 RTN
  → (scale*q + zp) * s_col * s_row → 反旋转。

本文件在**基线之上**加 V 侧变体（只改"尺度/因子的求解"，不改位宽；除 res/mix 外字节数不变）：

  rtn      基线：scale = fp16((hi-lo)/qmax)，zp = fp16(lo)，q = round((b-lo)/scale)  —— 复现用
  ls       量化后按重建误差重解 (scale, zp)：加权最小二乘（权重 = sinkhorn 因子的平方）
  ls1/ls2  ls 之后再迭代 1/2 轮"重新取整 + 重解 LS"（交替最小化，仍是均匀网格）
  lsq      ls + 分位数裁剪（quantile 取 lo/hi；上游 KVARN_RTN_QUANTILE 的免费旋钮）
  lsr      ls 的"去权重"版本（忽略 sinkhorn 因子，只对 b 做最小二乘）
  sinkmse  Sinkhorn 的迭代选择改用"重建 MSE 最小"而不是 imbalance 最小（免费旋钮）
  res      3bit 主 + 1bit 残差（4bit 总量，字节=4bit 档）
  chan     通道为行（stage0 的 V2conly/K4V2c 对照方向）
"""
from __future__ import annotations

import math
import torch

SLICE = 128
SIDE_ONE_G128 = 9.0     # 单侧 side info（与 g128_codecs 一致）


def hadamard_256(x: torch.Tensor) -> torch.Tensor:
    v = x.float().clone()
    n = 256
    assert v.shape[-1] == n
    h = 1
    while h < n:
        v2 = v.reshape(*v.shape[:-1], n // (2 * h), 2, h)
        a = v2[..., 0, :].clone()
        b = v2[..., 1, :].clone()
        v2[..., 0, :] = a + b
        v2[..., 1, :] = a - b
        v = v2.reshape(*v.shape[:-1], n)
        h *= 2
    return v / math.sqrt(n)


def _fp16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(torch.float32)


def _imb(t):
    sc = t.std(dim=-2)
    sr = t.std(dim=-1)
    return sc.amax(-1) / sc.amin(-1).clamp_min(1e-8) + sr.amax(-1) / sr.amin(-1).clamp_min(1e-8)


def _sinkhorn(tile: torch.Tensor, iters: int = 8, pick: str = "imb"):
    """逐行照 kv_codecs._sinkhorn（best-so-far 选版）。
    pick='imb' = 上游口径（imbalance 最小）；pick='mse' = 重建 MSE 最小（本探索新增，免费旋钮）。"""
    m = tile.float()
    N, R, C = m.shape
    log_s_col = torch.zeros(N, 1, C, dtype=torch.float32)
    log_s_row = torch.zeros(N, R, 1, dtype=torch.float32)

    def score(t, sc, sr):
        if pick == "mse":
            return ((t * sc * sr - m) ** 2).mean(dim=(-1, -2))
        return _imb(t)

    cur = m
    best = score(cur, torch.ones(N, 1, C), torch.ones(N, R, 1))
    sc_best = torch.ones(N, 1, C)
    sr_best = torch.ones(N, R, 1)
    for _ in range(iters):
        col_std = cur.std(dim=1, keepdim=True).clamp(1e-3, 1e3)
        log_s_col = (log_s_col + col_std.log()).clamp(-0.3, 10.0)
        cur = m / log_s_col.exp() / log_s_row.exp()
        row_std = cur.std(dim=2, keepdim=True).clamp(1e-3, 1e3)
        log_s_row = (log_s_row + row_std.log()).clamp(-0.3, 10.0)
        cur = m / log_s_col.exp() / log_s_row.exp()
        now = score(cur, log_s_col.exp(), log_s_row.exp())
        better = now <= best
        mask = better.view(N, 1, 1).to(log_s_col.dtype)
        sc_best = mask * log_s_col.exp() + (1 - mask) * sc_best
        sr_best = mask * log_s_row.exp() + (1 - mask) * sr_best
        best = torch.where(better, now, best)
    return m / sc_best / sr_best, sc_best, sr_best


def _rtn(tile, bits, quantile=None):
    qmax = (1 << bits) - 1
    if quantile is not None and quantile > 0:
        lo = torch.quantile(tile, quantile, dim=-1, keepdim=True)
        hi = torch.quantile(tile, 1.0 - quantile, dim=-1, keepdim=True)
    else:
        lo = tile.amin(dim=-1, keepdim=True)
        hi = tile.amax(dim=-1, keepdim=True)
    scale = ((hi - lo) / qmax).clamp_min(1e-10)
    q = torch.clamp(torch.round((tile - lo) / scale), 0, qmax)
    return q, _fp16(scale), _fp16(lo)


# ─────────────────── 逐行求解：给定整数码，重解 (scale, zp) ───────────────────
def _ls_affine(b, q, w, qmax):
    """在给定整数码 q 下，解 min_{s,zp} Σ w_i (s·q_i + zp − b_i)²（加权最小二乘，闭式）。
    b: 平衡后目标 [.., C]；q: 整数码；w: 权重（= sinkhorn 因子的平方，可全 1）。"""
    qf = q.to(torch.float32)
    bf = b.to(torch.float32)
    W = w
    Sw = W.sum(-1, keepdim=True).clamp_min(1e-12)
    Sq = (W * qf).sum(-1, keepdim=True)
    Sqq = (W * qf * qf).sum(-1, keepdim=True)
    Sb = (W * bf).sum(-1, keepdim=True)
    Sqb = (W * qf * bf).sum(-1, keepdim=True)
    det = (Sqq * Sw - Sq * Sq)
    ok = det.abs() > 1e-12
    s = torch.where(ok, (Sqb * Sw - Sq * Sb) / det.clamp_min(1e-30), torch.zeros_like(det))
    zp = torch.where(ok, (Sb - s * Sq) / Sw, Sb / Sw)
    s = s.clamp_min(1e-10)
    qmin, qmaxi = 0, qmax
    s = torch.where(ok & (qmaxi > qmin), s, torch.full_like(s, 1e-10))
    return s, zp


def _requant(b, s, zp, qmax):
    return torch.clamp(torch.round((b - zp) / s.clamp_min(1e-12)), 0, qmax)


def _rows_quant(bal, s_col, s_row, bits, variant="rtn", quantile=None, refine=0, wmode="f2"):
    """对平衡张量 bal [N, R, C] 逐行量化；返回重建（已乘回因子）。"""
    qmax = (1 << bits) - 1
    f = (s_col * s_row)                      # [N, R, C] 反量化时要乘回的因子
    q, scale, zp = _rtn(bal, bits, quantile)
    if variant.startswith("ls"):
        w = (f * f) if wmode == "f2" else torch.ones_like(f)
        for it in range(refine + 1):
            scale, zp = _ls_affine(bal, q, w, qmax)
            scale = _fp16(scale)
            zp = _fp16(zp)
            if it < refine:
                q = _requant(bal, scale, zp, qmax)
        # 裁剪档：LS 之后再夹一次码（分位数裁剪在 _rtn 里已生效）
    return (scale * q + zp) * f, q, scale, zp


def _blocks(x):
    H, T, D = x.shape
    assert D == 256 and T % SLICE == 0
    nt = T // SLICE
    b = x.reshape(H, nt, SLICE, 2, SLICE).permute(0, 1, 3, 2, 4).reshape(H * nt * 2, SLICE, SLICE)
    return b, (H, nt)


def _unblocks(b, H, nt):
    return b.reshape(H, nt, 2, SLICE, SLICE).permute(0, 1, 3, 2, 4).reshape(H, nt * SLICE, 256)


def _v_to_channel_rows(vr, W=1):
    """通道为行：M = [N, 256(dim), 128(token)]（与 g128_codecs mode='channel' 逐位等价）。
    W>1 时 RTN 的行改成"W 个相邻通道 × 128 token"的二维子块（行数 256/W，side info 更省）。"""
    H, T, _ = vr.shape
    nt = T // SLICE
    M = vr.reshape(H, nt, SLICE, 2, SLICE).permute(0, 1, 3, 4, 2).reshape(H * nt, 2 * SLICE, SLICE)
    if W > 1:
        M = M.reshape(H * nt, (2 * SLICE) // W, W * SLICE)
    return M, (H, nt, W)


def _v_from_channel_rows(M, H, nt, W=1):
    if W > 1:
        M = M.reshape(H * nt, 2 * SLICE, SLICE)
    return M.reshape(H, nt, 2, SLICE, SLICE).permute(0, 1, 4, 2, 3).reshape(H, nt * SLICE, 256)


def v_quant(v, bits=3, iters=8, variant="rtn", quantile=None, refine=0,
            rotate=True, mode="token", sink_pick="imb", wmode="f2", row_dims=1):
    """返回 (v_hat, 字节/ token·KVhead, 附加信息 dict)。"""
    vr = hadamard_256(v) if rotate else v.float().clone()
    H, T, _ = vr.shape
    if mode.startswith("channel"):
        M, (H2, nt, W) = _v_to_channel_rows(vr, row_dims)
    else:
        M, (H2, nt) = _blocks(vr)
        W = 1
    bal, s_col, s_row = _sinkhorn(M, iters, pick=sink_pick)
    rec, q, scale, zp = _rows_quant(bal, s_col, s_row, bits, variant, quantile, refine, wmode)
    out = _v_from_channel_rows(rec, H2, nt, W) if mode.startswith("channel") else _unblocks(rec, H2, nt)
    # side info（每 token·KVhead，fp16）：record 口径 = 逐行 scale+zp(4B) + 逐列 other(2B)
    if mode.startswith("channel"):
        n_rows = 2 * SLICE // row_dims
        n_cols = row_dims * SLICE
        side = (4.0 * n_rows + 2.0 * n_cols) / SLICE
    else:
        side = (4.0 * SLICE + 2.0 * 2 * SLICE) / SLICE
    return (hadamard_256(out) if rotate else out), 256.0 * bits / 8.0 + side, dict(q=q, scale=scale, zp=zp)


def v_quant_res(v, bits_main=3, iters=8, rotate=True, quantile=None):
    """3bit 主 + 1bit 残差（只在 V 上）。残差每行一个 ±δ（δ = mean|e|，fp16 存 2 B/行）。"""
    vr = hadamard_256(v) if rotate else v.float().clone()
    H, T, _ = vr.shape
    M, (H2, nt) = _blocks(vr)
    bal, s_col, s_row = _sinkhorn(M, iters)
    f = s_col * s_row
    q, scale, zp = _rtn(bal, bits_main, quantile)
    scale, zp = _ls_affine(bal, q, f * f, (1 << bits_main) - 1)
    scale = _fp16(scale)
    zp = _fp16(zp)
    r0 = scale * q + zp
    e = bal - r0
    delta = _fp16(e.abs().mean(-1, keepdim=True).clamp_min(1e-10))
    e_hat = torch.sign(e) * delta
    rec = (r0 + e_hat) * f
    out = _unblocks(rec, H2, nt)
    # 字节：3bit 码 + 1bit 码 + 每行 delta(fp16) 摊到 128 列 + side
    bpe = 256.0 * 4 / 8.0 + SIDE_ONE_G128 + 2.0 * 2 / 128.0
    return (hadamard_256(out) if rotate else out), bpe, dict()


def v_quant_mixed(v, bits_per_head, iters=8, variant="rtn", quantile=None, refine=0,
                  rotate=True, mode="channel", sink_pick="imb"):
    """按 head 给不同位宽（bits_per_head: 长度=KV head 数的列表）。
    返回 (v_hat, 平均位宽, 平均字节, 附加)。"""
    H = v.shape[0]
    assert len(bits_per_head) == H
    outs = []
    for h in range(H):
        xh, _, _ = v_quant(v[h:h + 1], bits=bits_per_head[h], iters=iters, variant=variant,
                           quantile=quantile, refine=refine, rotate=rotate, mode=mode,
                           sink_pick=sink_pick)
        outs.append(xh)
    vh = torch.cat(outs, 0)
    avg_bits = sum(bits_per_head) / H
    return vh, avg_bits, 256.0 * avg_bits / 8.0 + SIDE_ONE_G128, dict()


def k_quant(k, bits=4, iters=8, variant="rtn", quantile=None, refine=0, wmode="f2", rotate=True):
    """K 侧（= g128_codecs.g128_quant_k 的结构：每个 128 维 slice 一张 [dim, tok] 表）。
    variant/quantile/refine 是本探索加的免费旋钮（不改字节）。"""
    kr = hadamard_256(k) if rotate else k.float().clone()
    H, T, _ = kr.shape
    b, (H2, nt) = _blocks(kr)
    kb = b.transpose(-1, -2).contiguous()                 # [N, 128(dim), 128(tok)]
    bal, s_col, s_row = _sinkhorn(kb, iters)
    rec, q, scale, zp = _rows_quant(bal, s_col, s_row, bits, variant, quantile, refine, wmode)
    out = _unblocks(rec.transpose(-1, -2).contiguous(), H2, nt)
    return (hadamard_256(out) if rotate else out), 256.0 * bits / 8.0 + 10.0


def v_quant_promote(v, bits_lo=3, bits_hi=4, k_promote=32, iters=8, variant="ls",
                    quantile=None, refine=1, rotate=True, mask_bytes_per_tile=32.0):
    """通道为行 + 逐 tile 挑能量最高的 k_promote 个通道给高位宽（其余低位宽）。
    掩码按每 tile 32 B（256 bit）计费。返回 (v_hat, 平均位宽, bpe, 附加)。"""
    vr = hadamard_256(v) if rotate else v.float().clone()
    H, T, _ = vr.shape
    M, (H2, nt, W) = _v_to_channel_rows(vr, 1)          # [H*nt, 256(dim), 128(tok)]
    bal, s_col, s_row = _sinkhorn(M, iters)
    f = s_col * s_row
    energy = (bal.float() ** 2).mean(dim=-1)            # [N, 256] 每通道能量（平衡后）
    idx = torch.argsort(energy, dim=-1, descending=True)
    hi = torch.zeros_like(energy, dtype=torch.bool)
    hi.scatter_(1, idx[:, :k_promote], True)
    qmax_lo, qmax_hi = (1 << bits_lo) - 1, (1 << bits_hi) - 1
    rec = torch.zeros_like(bal)
    qs = torch.zeros_like(bal)
    for sel, bits in ((~hi, bits_lo), (hi, bits_hi)):
        qmax = (1 << bits) - 1
        m3 = sel.unsqueeze(-1)
        b_sel = torch.where(m3, bal, torch.zeros_like(bal))
        f_sel = torch.where(m3, f, torch.zeros_like(f))
        q, s, z = _rtn(b_sel, bits, quantile)
        if variant.startswith("ls"):
            w = f_sel * f_sel
            for it in range(refine + 1):
                s, z = _ls_affine(b_sel, q, w, qmax)
                s = _fp16(s); z = _fp16(z)
                if it < refine:
                    q = _requant(b_sel, s, z, qmax)
        r = (s * q + z) * f_sel
        rec = torch.where(m3, r, rec)
        qs = torch.where(m3, q.to(torch.float32), qs)
    out = _v_from_channel_rows(rec, H2, nt, 1)
    avg_bits = (bits_lo * (256 - k_promote) + bits_hi * k_promote) / 256.0
    bpe = 256.0 * avg_bits / 8.0 + 10.0 + mask_bytes_per_tile / SLICE
    return (hadamard_256(out) if rotate else out), avg_bits, bpe, dict()


def metrics(x, xh, D=256):
    """逐 (head, token) 256 维向量的 cos 与 relL2。"""
    a = x.reshape(-1, D).float()
    b = xh.reshape(-1, D).float()
    num = (a * b).sum(-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-12
    cos = num / den
    rel = (a - b).norm(dim=-1) / (a.norm(dim=-1) + 1e-12)
    return dict(cos_mean=cos.mean().item(), cos_min=cos.min().item(),
                relL2_mean=rel.mean().item(), relL2_max=rel.max().item(),
                relL2_glob=((a - b).norm() / a.norm()).item())


def record_bytes(bits_k, bits_v):
    return 32.0 * (bits_k + bits_v) + 24.0
