#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验：v_codecs 的 channel 模式 == g128_codecs 的 K 路径（同一结构），用于 K 侧免费旋钮。"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_codecs as V  # noqa: E402

TILES = r"C:\Users\xunying\AppData\Local\hermes\cache\scratch\kvarn-tiles"


def g128_quant_k_ref(k, bits, iters=8, quantile=None):
    """照抄 ref/g128_codecs.py 的 K 路径（去掉 import 依赖）。"""
    def _rows_rtn(M, bits, iters=8, quantile=None):
        bal, sc, sr = V._sinkhorn(M, iters)
        q, scale, zp = V._rtn(bal, bits, quantile)
        return (scale.to(torch.float32) * q + zp.to(torch.float32)) * sc * sr
    kr = V.hadamard_256(k)
    H, T, _ = kr.shape
    b, (H2, nt) = V._blocks(kr)
    kb = b.transpose(-1, -2).contiguous()
    kb = _rows_rtn(kb, bits, iters, quantile)
    out = V._unblocks(kb.transpose(-1, -2).contiguous(), H2, nt)
    return V.hadamard_256(out)


for lay in (35, 63):
    k = torch.load(os.path.join(TILES, "layer%d_k_proj.pt" % lay), map_location="cpu").float()[0]
    k = k.reshape(256, 4, 256).permute(1, 0, 2)
    a = g128_quant_k_ref(k, 4, iters=8)
    b = V.v_quant(k, bits=4, mode="channel", variant="rtn")[0]
    d = (a - b).abs().max().item()
    rel = (a - b).norm().item() / a.norm().item()
    print("layer%d: max|Δ|=%.3e  relL2=%.3e  ⇒ %s" % (lay, d, rel, "等价" if d < 1e-5 else "不等价"))
