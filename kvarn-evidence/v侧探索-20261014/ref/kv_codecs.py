#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 0 的三条 fake-quant 编解码器（纯 PyTorch，CPU 可跑）。

对撞对象（全部按 2026-10-12 从两边源码里抠出来的事实实现，出处写在每个函数上）：
  1) kvarn_g64   —— 上游 huawei-csl/KVarN 的 dense 路径，tile=64（对齐 ninfer 的 64-token 页）
                    来源：ref/sinkhorn.py（16 轮 log 域 Sinkhorn，best-so-far）
                          ref/kvarn_store.py（每行非对称 RTN，RTN 的 scale/zp 吸进 sinkhorn scale）
  2) rk4v4       —— ninfer 现生产档：K = 旋转 + G64 Lloyd-Max 4-bit + 最小二乘 FP16 scale
                                          V = G32 absmax/7 的 signed int4
                    来源：ref-ninfer/.../src/ops/kv_cache/int8_g64_codec.cuh（149-330 行）
                          .../hadamard_d256.cuh（固定归一化 Sylvester，2^-4，无随机符号）
  3) rk8v4       —— 质量上界参照：K = 旋转 + G8(int8) G64 absmax/127，V = int8 G64
                    来源：同一文件 49-71 行

诚实声明（这些是"参考"，不是"逐字节等价的移植"）：
  * 旋转用的是标准 Sylvester/Hadamard（与 ninfer 同族、同归一化、自逆）；ninfer kernel 里的 lane 映射
    与 butterfly 顺序只在"逐字节对拍"时才需要一致（阶段 2 的 Gate），对质量研究不影响。
  * kvarn 的 2-bit V 走上游 batched 变体 `_pack_lowbit(bits=2)`（4 值一字节），
    K 侧仍是 4-bit（上游 `kvarn_store_tile_k` 只验收了 4-bit，2-bit V 正是 k4v2 预设）。
  * Lloyd 判决阈值取重建点的中点（Lloyd-Max 的定义），与 kernel 里那套 packed-nibble 比较器
    在阈值上等价，这点到阶段 2 再用真实张量逐字节对拍。
"""
from __future__ import annotations
import math
import torch

# ───────────────────────── 旋转（两边都用，且必须自逆） ─────────────────────────
def hadamard_256(x: torch.Tensor) -> torch.Tensor:
    """归一化 Sylvester H256：x @ H / 16，H 自逆。等价 ninfer 的 normalized_hadamard_d256_inplace
    （hadamard_d256.cuh 里明确写 0x1p-4 = 1/16，且没有随机符号翻转）。"""
    v = x.float().clone()
    n = 256
    assert v.shape[-1] == n, "head_dim 必须是 256"
    # 标准 FWHT（Sylvester 序）
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
    """走一趟 fp16：kernel 存的 scale 都是 __float2half_rn，回读的是被表示过的值。"""
    return x.to(torch.float16).to(torch.float32)


# ───────────────────────── kvarn（上游 dense 路径） ─────────────────────────
def _sinkhorn(tile: torch.Tensor, iters: int = 16):
    """log 域交替列/行标准差归一化，取 imbalance 最低的那一版（best-so-far）。
    逐行照 ref/sinkhorn.py 的 variance_normalize_batched 搬过来。tile: [N, R, C]"""
    m = tile.float()
    N, R, C = m.shape
    log_s_col = torch.zeros(N, 1, C, dtype=torch.float32)
    log_s_row = torch.zeros(N, R, 1, dtype=torch.float32)

    def imb(t):
        sc = t.std(dim=-2)
        sr = t.std(dim=-1)
        return sc.amax(-1) / sc.amin(-1).clamp_min(1e-8) + sr.amax(-1) / sr.amin(-1).clamp_min(1e-8)

    cur = m
    imb_best = imb(cur)
    sc_best = torch.ones(N, 1, C)
    sr_best = torch.ones(N, R, 1)
    for _ in range(iters):
        col_std = cur.std(dim=1, keepdim=True).clamp(1e-3, 1e3)
        log_s_col = (log_s_col + col_std.log()).clamp(-0.3, 10.0)
        cur = m / log_s_col.exp() / log_s_row.exp()
        row_std = cur.std(dim=2, keepdim=True).clamp(1e-3, 1e3)
        log_s_row = (log_s_row + row_std.log()).clamp(-0.3, 10.0)
        cur = m / log_s_col.exp() / log_s_row.exp()
        imb_now = imb(cur)
        better = imb_now <= imb_best
        mask = better.view(N, 1, 1).to(log_s_col.dtype)
        sc_best = mask * log_s_col.exp() + (1 - mask) * sc_best
        sr_best = mask * log_s_row.exp() + (1 - mask) * sr_best
        imb_best = torch.where(better, imb_now, imb_best)
    return m / sc_best / sr_best, sc_best, sr_best


def _rtn(tile: torch.Tensor, bits: int, quantile: float | None = None):
    """每行（last dim）非对称 RTN：q = round((x-zp)/scale) clamp[0,2^bits-1]，scale=fp16((hi-lo)/qmax)。

    quantile 不为 None 时按分位数取 lo/hi（离群值被 clamp，换取批量分辨率）——这正是上游
    `kvarn_store._rtn_range` 里 `KVARN_RTN_QUANTILE` 的作用，上游注释点名它是给 2-bit V 与
    K 离群（max/std≈6）准备的。不给 rk4v4 这个待遇是故意的：ninfer 的 V codec 就是 absmax/7，
    没有分位数分支（int8_g64_codec.cuh:101-118）。
    """
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


def _fake_unif(q, scale, zp, bits):
    qmax = (1 << bits) - 1
    return scale.to(torch.float32) * (q.to(torch.float32) * (scale / scale).to(torch.float32) * 0 + q) + zp.to(torch.float32)


def kvarn_fakequant(k: torch.Tensor, v: torch.Tensor, tile_tokens: int = 64,
                    bits_k: int = 4, bits_v: int = 2, iters: int = 16,
                    quantile_k: float | None = None, quantile_v: float | None = None):
    """kvarn k4v2 的 fake-quant：K 走 [D, group] 的 K 路径，V 走 [group, D] 的 V 路径（都是旋转之后）。

    k: [.., tokens, D]   v: [.., tokens, D]   （tokens 必须整除 tile_tokens）
    返回 (k_hat, v_hat, 每 token 字节数)
    """
    *lead, T, D = k.shape
    assert D == 256 and T % tile_tokens == 0
    lead_shape = list(k.shape[:-2])
    ntile = T // tile_tokens
    G = tile_tokens
    Ntot = (math.prod(lead_shape) if lead_shape else 1) * ntile

    def one_path(t_rot_flat: torch.Tensor, orient: str, bits: int, quantile=None) -> torch.Tensor:
        """输入 [Ntot, G, D] 已旋转的 tile；fake-quant 后返回同布局。
        K 路径按上游要 [D, group]（行=通道），V 路径按上游要 [group, D]（行=token）。"""
        tt = t_rot_flat if orient == "v" else t_rot_flat.transpose(-1, -2).contiguous()
        balanced, s_col, s_row = _sinkhorn(tt, iters)
        q, scale, zp = _rtn(balanced, bits, quantile)
        # 反量化：RTN 的 scale/zp 先还原，再乘回 sinkhorn 吸进去的两个 scale
        return (scale.to(torch.float32) * q + zp.to(torch.float32)) * s_col * s_row

    k_rot = hadamard_256(k).reshape(Ntot, G, D)
    v_rot = hadamard_256(v).reshape(Ntot, G, D)
    k_hat = one_path(k_rot, "k", bits_k, quantile_k).transpose(-1, -2).reshape(Ntot, G, D)
    v_hat = one_path(v_rot, "v", bits_v, quantile_v).reshape(Ntot, G, D)
    k_hat = hadamard_256(k_hat.reshape(*lead_shape, T, D))   # 同一变换即反旋转（H 自逆）
    v_hat = hadamard_256(v_hat.reshape(*lead_shape, T, D))

    # 字节账（每 token 每 KV head）：码 + scale/zp 摊到 token
    b_code_k = D * bits_k / 8.0
    b_scale_k = (D * 2 + D * 2) / G                        # s_col_K[D] + zp_K[D]，fp16，摊到 G 个 token
    b_row_k = G * 2 / G                                   # s_row_K[G] fp16
    b_code_v = D * bits_v / 8.0
    b_scale_v = (D * 2) / G                                # s_col_V[D] fp16
    b_row_v = (G * 2 * 2) / G                              # s_row_V[G] + zp_V[G] fp16
    bpe = (b_code_k + b_scale_k + b_row_k) + (b_code_v + b_scale_v + b_row_v)
    return k_hat, v_hat, bpe


# ───────────────────────── ninfer 现生产：rk4v4 / rk8v4 ─────────────────────────
_LLOYD_POINTS = torch.tensor([0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326])
_LLOYD_CODES = torch.tensor([6, 18, 31, 44, 58, 75, 96, 127], dtype=torch.float32)


def _lloyd_mag_index(z_abs: torch.Tensor) -> torch.Tensor:
    """|z|（以 group RMS 为单位）→ 0..7 的幅度索引；判决阈值 = 相邻重建点中点（Lloyd-Max 定义）。"""
    mid = (_LLOYD_POINTS[1:] + _LLOYD_POINTS[:-1]) / 2.0   # 7 个阈值
    return (z_abs.unsqueeze(-1) >= mid).sum(-1)


def _group_rms(x: torch.Tensor, group: int):
    return (x.pow(2).mean(dim=-1, keepdim=True)).sqrt()


def rk4v4_fakequant(k: torch.Tensor, v: torch.Tensor, iters_note: str = ""):
    """rk4v4 = RotatedLloyd4KeyInt4Value（d256_profile.h:215-129 与 int8_g64_codec.cuh:149-330）
      K：旋转 → G64 求 RMS → 16 码（符号 | 幅度索引）→ scale = FP16(Σx·c / Σc²)（最小二乘）
      V：不旋转？—— 源码里 V 走 int4 的 G32 absmax/7；这里与 kernel 一致：**V 也过同一个固定旋转**
          （若阶段 2 对拍发现 V 不旋转，改这一行即可，已在这里标注）
    """
    *lead, T, D = k.shape
    assert D == 256
    k_hat = hadamard_256(hadamard_256(k))  # 占位，下面真正做量化
    # --- K：Lloyd4 ---
    kr = hadamard_256(k).reshape(*lead, T, D // 64, 64)
    rms = _group_rms(kr, 64)
    z = torch.where(rms > 0, kr / rms.clamp_min(1e-12), torch.zeros_like(kr))
    mag = _lloyd_mag_index(z.abs())                              # [.., T, 4]
    codes = _LLOYD_CODES[mag.long()]                             # 幅度码
    codes = torch.where(z < 0, -codes, codes)                    # 符号位（kernel 用 index 的 bit3）
    xc = (kr * codes).sum(-1, keepdim=True)
    cc = codes.pow(2).sum(-1, keepdim=True)
    scale = _fp16(torch.where(rms > 0, xc / cc.clamp_min(1e-12), torch.zeros_like(xc)))
    kq = scale * codes
    k_hat = hadamard_256(kq.reshape(*lead, T, D))
    # --- V：int4 G32 ---
    vr = hadamard_256(v).reshape(*lead, T, D // 32, 32)
    amax = vr.abs().amax(-1, keepdim=True)
    vscale = _fp16(torch.where(amax > 0, amax / 7.0, torch.zeros_like(amax)))
    inv = torch.where(vscale > 0, 1.0 / vscale.clamp_min(1e-12), torch.zeros_like(vscale))
    vq = torch.clamp(torch.round(vr * inv), -7, 7)
    v_hat = hadamard_256((vscale * vq).reshape(*lead, T, D))
    bpe = (256 * 4 / 8 + 4 * 2) + (256 * 4 / 8 + 8 * 2)          # K 136 + V 144 = 280 B/token
    return k_hat, v_hat, bpe


def rk8v4_fakequant(k: torch.Tensor, v: torch.Tensor):
    """rk8v4 = RotatedInt8KeyInt4ValueGroup64（int8_g64_codec.cuh:49-118）：
      K：旋转 + int8（G64，scale=FP16(absmax/127)）；V：signed int4（G32，absmax/7）"""
    *lead, T, D = k.shape
    kr = hadamard_256(k).reshape(*lead, T, D // 64, 64)
    amax = kr.abs().amax(-1, keepdim=True)
    kscale = _fp16(torch.where(amax > 0, amax / 127.0, torch.zeros_like(amax)))
    kinv = torch.where(kscale > 0, 1.0 / kscale.clamp_min(1e-12), torch.zeros_like(kscale))
    kq = torch.clamp(torch.round(kr * kinv), -127, 127)
    k_hat = hadamard_256((kscale * kq).reshape(*lead, T, D))
    vr = hadamard_256(v).reshape(*lead, T, D // 32, 32)
    vam = vr.abs().amax(-1, keepdim=True)
    vscale = _fp16(torch.where(vam > 0, vam / 7.0, torch.zeros_like(vam)))
    vinv = torch.where(vscale > 0, 1.0 / vscale.clamp_min(1e-12), torch.zeros_like(vscale))
    vq = torch.clamp(torch.round(vr * vinv), -7, 7)
    v_hat = hadamard_256((vscale * vq).reshape(*lead, T, D))
    bpe = (256 + 4 * 2) + (256 * 4 / 8 + 8 * 2)                  # K 264 + V 144 = 408 B/token
    return k_hat, v_hat, bpe


# ───────────────────────── 自一致性测试 ─────────────────────────
def attn_cosine(q, k, v, k_hat, v_hat):
    """量化前后 attention 输出的 cosine（阶段 3 的 Gate 就是这个判据 ≥0.999）。"""
    def run(K, V):
        s = (q @ K.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        return torch.softmax(s, dim=-1) @ V
    a = run(k, v)
    b = run(k_hat, v_hat)
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1])
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


def make_tensors(T=2048, heads=4, outlier=True, seed=0):
    """合成但带真实特征的 K/V：每通道不同尺度 + 少数大离群值（K 离群是 KVarN 论文与
    kvarn_store 里 KVARN_RTN_QUANTILE 存在的原因）。"""
    g = torch.Generator().manual_seed(seed)
    chan = torch.randn(1, 1, 1, 256, generator=g).abs() + 0.4          # 通道尺度差异
    k = torch.randn(1, heads, T, 256, generator=g) * chan
    v = torch.randn(1, heads, T, 256, generator=g) * chan
    if outlier:
        idx = torch.randint(0, 256, (1, heads, T, 6), generator=g)
        for b in range(heads):
            for t in range(T):
                k[0, b, t, idx[0, b, t, :]] *= 6.0                      # K 离群 ~ max/std≈6
    return k, v


def main():
    torch.manual_seed(0)
    T, heads = 512, 4
    q = hadamard_256(torch.randn(1, heads, T, 256)) * 0.5
    print("方案             每token字节   K误差cosine   V误差cosine   attention cosine   Sinkhorn 后 imbalance")
    rows = []
    for name, fn, kwargs in [("kvarn_g64", kvarn_fakequant, dict(tile_tokens=64, bits_k=4, bits_v=2)),
                             ("kvarn_g64+clip", kvarn_fakequant, dict(tile_tokens=64, bits_k=4, bits_v=2, quantile_v=0.005)),
                             ("kvarn_g64+klclip", kvarn_fakequant, dict(tile_tokens=64, bits_k=4, bits_v=2, quantile_k=0.005, quantile_v=0.005)),
                             ("kvarn_g128", kvarn_fakequant, dict(tile_tokens=128, bits_k=4, bits_v=2)),
                             ("kvarn_g128+clip", kvarn_fakequant, dict(tile_tokens=128, bits_k=4, bits_v=2, quantile_v=0.005)),
                             ("rk4v4(生产)", rk4v4_fakequant, {}),
                             ("rk8v4(上界)", rk8v4_fakequant, {})]:
        for outlier in (True, False):
            k, v = make_tensors(T=T, heads=heads, outlier=outlier)
            out = fn(k, v, **kwargs) if kwargs else fn(k, v)
            k_hat, v_hat, bpe = out
            ck = torch.nn.functional.cosine_similarity(k.reshape(-1, 256), k_hat.reshape(-1, 256), dim=-1).mean().item()
            cv = torch.nn.functional.cosine_similarity(v.reshape(-1, 256), v_hat.reshape(-1, 256), dim=-1).mean().item()
            ac = attn_cosine(q, k, v, k_hat, v_hat)
            print("%-14s %9.1f      %.6f      %.6f      %.6f          %s" %
                  (name, bpe, ck, cv, ac, "离群" if outlier else "干净"))
            rows.append((name, outlier, bpe, ck, cv, ac))
    print("\n注：这是合成张量上的方法学校准（真实分布要等阶段 0 的正式跑），")
    print("    真实权重的 K/V 一到就换真张量，判据沿用 attn cosine ≥ 0.999 与 PPL ≤ +0.1%。")


if __name__ == "__main__":
    main()
