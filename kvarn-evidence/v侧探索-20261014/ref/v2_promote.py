# v2_promote.py —— V 保持 2 bit，但把能量最高的 frac 比例通道提到 4 bit（离群通道混合精度）
# 阶段 0 三轮筛选的结论：2-bit V 只有这条路真的有效（两篇真实 K/V 上一致），
# Lloyd 码本（全局或每瓦片）都输给 uniform —— MSE 最优不等于余弦最优，点积被离群值主导。
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_codecs import hadamard_256, _sinkhorn  # noqa: E402

D, G = 256, 64


def _u(bal, levels):
    """uniform 非对称：scale/zp 由每行 min/max 定（kernel 里两趟归约即可）。"""
    lo = bal.amin(dim=-1, keepdim=True)
    hi = bal.amax(dim=-1, keepdim=True)
    sc = ((hi - lo) / levels).clamp_min(1e-6)
    q = ((bal - lo) / sc).round().clamp(0, levels)
    return sc.float() * q + lo.float()


def sinkhorn_v_channel_promote(x, frac=0.125, iters=16, bits=2, rotate=True, mode="chan"):
    """x: [.., T, 256] float32(cpu)。返回 fake-quant 后的同形状张量。

    frac = 提到 4 bit 的通道比例（按瓦片内每通道 std 排序取前 frac）。
    字节账：V 码 64 B + side 18 B + 256*frac*2/8 B。
    """
    T = x.shape[-2]
    lead = list(x.shape[:-2])
    ntile = T // G
    N = (1 if not lead else int(torch.tensor(lead).prod().item())) * ntile
    xr = hadamard_256(x).reshape(N, G, D) if rotate else x.reshape(N, G, D)
    tt = xr.transpose(-1, -2).contiguous()                      # [N, D, G] 行=通道
    balanced, s_col, s_row = _sinkhorn(tt, iters)
    rowstd = balanced.std(dim=-1, unbiased=False)
    k = max(1, int(round(balanced.shape[-2] * frac)))
    order = rowstd.argsort(dim=-1, descending=True)
    keep = torch.zeros_like(balanced, dtype=torch.bool)
    ar = torch.arange(balanced.shape[0])[:, None] if balanced.dim() == 3 else None
    if balanced.dim() == 3:
        keep[ar, order[:, :k]] = True
    else:
        keep.scatter_(-1, order[..., :k], True)
    # mode: chan = 只提离群通道；tok = 只提离群 token；both = 两边都提
    if mode in ("tok", "both"):
        colsd = balanced.std(dim=-2, unbiased=False)                     # [N, G] 每 token 的能量
        kc = max(1, int(round(balanced.shape[-1] * frac)))
        corder = colsd.argsort(dim=-1, descending=True)
        ktok = torch.zeros_like(balanced, dtype=torch.bool)
        ktok.scatter_(-1, corder[..., None, :kc].expand(-1, balanced.shape[-2], kc).transpose(-1, -2).contiguous()
                     if False else corder[..., None, :kc].expand(-1, balanced.shape[-2], kc).transpose(-1, -2), True)
        keep = (keep | ktok) if mode == "both" else ktok
    deq = torch.where(keep, _u(balanced, 15), _u(balanced, (1 << bits) - 1))
    # 必须像 v_diag 那样先乘列/行因子（此时布局是 [N,D,G]，因子形状只对得上这个布局），
    # 再转置回 [N,G,D]；顺序写反会报 "size of tensor a (256) must match ... (64)"
    deq = (deq * s_col * s_row).transpose(-1, -2).reshape(N, G, D)
    deq = deq.reshape(x.shape)
    return hadamard_256(deq) if rotate else deq
