#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新路线（beellama KVarN，G128）的 KV fake-quant，供阶段 0 误差归因使用。

与 /data/kvarn/kv_codecs.py 的 kvarn_fakequant（G64/G128）同一条算法，差别列清（不改上游数学）：

1) tile 语义按 doc 03 §3.7/§3.8 的 record 定义切成 128×128 的 slice：
     K 的 row = dim（一个 slice 的 128 个通道）、col = token
     V 的 row = token、col = dim（一个 slice 的 128 个通道）
   旧实现把整个头当成 [256, G] / [G, 256] 一块。对 K 的逐行 RTN 数值上等价（行独立、行内只看 token），
   对 V 不等价：旧实现"每 token 一行覆盖 256 维"，新语义"每 token 两行、每行 128 维" ⇒ 这里按新语义。
2) Sinkhorn 迭代 8（beellama KVAR_N_OP_PARAM_ITERS=8），旧实现 16。
3) 旋转沿用 ninfer 生产同族：归一化 Sylvester H256（自逆、无随机符号）。
   beellama 是 WHT128 + cross-slice 混合，两者是否等价是 doc 03 §3.9 明列的阶段 2 Gate 项
   （"不要默认等价"）。本阶段要的是 K/V 相对敏感度排序，不依赖这个绝对值。
4) RTN 非对称（scale+zp），与上游 kvarn_store 的每行 RTN 一致（不是 ninfer V 那种 absmax/7 对称）。

字节账：按 doc 00 的表反解，G128 的 side info = 18 B/token（4+2=210、4+4=274、8+8=530 ⇒
256*(bk+bv)/8 + S = 总值 ⇒ S=18）；G64 同法 = 36 B/token（4+2=228、4+4=292）。
"""
from __future__ import annotations

import torch

from kv_codecs import hadamard_256, _fp16, _rtn, _sinkhorn  # noqa: F401

SLICE = 128           # tile 边长：128 个 token × 一个 slice 的 128 维
SIDE_G128 = 18.0      # 每 KV head·token 的 side info 合计（K+V 两侧一起），反解自 doc 00 字节表：
                      #   4+2=210、4+4=274、8+8=530 ⇒ 256*(bk+bv)/8 + 18 = 总值
SIDE_ONE_G128 = 9.0   # 单侧摊到的 side info（K4=137、V2=73、合计 210）
SIDE_G64 = 36.0       # 同法反解 G64 的旧档：4+2=228、4+4=292（旧档案 §30）


def _blocks(x: torch.Tensor):
    """[H, T, 256] -> [H*nt*2, 128(tok), 128(dim)]，附 (H, nt) 以便逆变换。"""
    H, T, Dd = x.shape
    assert Dd == 256 and T % SLICE == 0, (T, Dd)
    nt = T // SLICE
    b = x.reshape(H, nt, SLICE, 2, SLICE).permute(0, 1, 3, 2, 4).reshape(H * nt * 2, SLICE, SLICE)
    return b, (H, nt)


def _unblocks(b: torch.Tensor, H: int, nt: int) -> torch.Tensor:
    return b.reshape(H, nt, 2, SLICE, SLICE).permute(0, 1, 3, 2, 4).reshape(H, nt * SLICE, 256)


def _rows_rtn(M: torch.Tensor, bits: int, iters: int = 8, quantile=None) -> torch.Tensor:
    """每行（最后一个维度）独立非对称 RTN，Sinkhorn 的 scale 吸进反量化。M: [N, R, C]"""
    balanced, s_col, s_row = _sinkhorn(M, iters)
    q, scale, zp = _rtn(balanced, bits, quantile)
    return (scale.to(torch.float32) * q + zp.to(torch.float32)) * s_col * s_row


def g128_quant_k(k: torch.Tensor, bits: int, iters: int = 8, quantile=None,
                 rotate: bool = True) -> torch.Tensor:
    """K 侧：tile 内 128 个 token 一行（行 = 一个通道）。k: [H, T, 256] float32。"""
    kr = hadamard_256(k) if rotate else k.float().clone()
    H, T, _ = kr.shape
    b, (H, nt) = _blocks(kr)                    # [N, tok, dim]
    kb = b.transpose(-1, -2).contiguous()       # [N, dim, tok]  row=dim
    kb = _rows_rtn(kb, bits, iters, quantile)
    out = _unblocks(kb.transpose(-1, -2).contiguous(), H, nt)
    return hadamard_256(out) if rotate else out


def _v_to_channel_rows(vr: torch.Tensor):
    """把 V 的两片拼成 256 通道为行、128 token 为列（§17 的对照方向）。"""
    H, T, _ = vr.shape
    nt = T // SLICE
    M = vr.reshape(H, nt, SLICE, 2, SLICE).permute(0, 1, 3, 4, 2).reshape(H * nt, 2 * SLICE, SLICE)
    return M, (H, nt)


def _v_from_channel_rows(M: torch.Tensor, H: int, nt: int) -> torch.Tensor:
    return M.reshape(H, nt, 2, SLICE, SLICE).permute(0, 1, 4, 2, 3).reshape(H, nt * SLICE, 256)


def g128_quant_v(v: torch.Tensor, bits: int, iters: int = 8, quantile=None,
                 rotate: bool = True, mode: str = "token") -> torch.Tensor:
    """V 侧。mode='token' = beellama 语义（每 token 一行、行内 128 维）；
    mode='channel' = 对照方向（每通道一行、行内 128 token）。"""
    vr = hadamard_256(v) if rotate else v.float().clone()
    H, T, _ = vr.shape
    if mode == "channel":
        M, (H, nt) = _v_to_channel_rows(vr)
        M = _rows_rtn(M, bits, iters, quantile)
        out = _v_from_channel_rows(M, H, nt)
    else:
        b, (H, nt) = _blocks(vr)                # [N, tok, dim]  row=token
        b = _rows_rtn(b, bits, iters, quantile)
        out = _unblocks(b, H, nt)
    return hadamard_256(out) if rotate else out


def bpt(bits_k, bits_v, side: float = SIDE_G128) -> float:
    """每 KV head·token 的字节数（doc 00 口径：256*bits/8 + side）。"""
    return 256.0 * bits_k / 8.0 + 256.0 * bits_v / 8.0 + side
