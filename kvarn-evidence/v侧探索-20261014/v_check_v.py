#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验：v_codecs 的 channel/token 模式与 ref/g128_codecs.py 逐位等价（同一 codec）。"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_codecs as V  # noqa: E402

TILES = r"C:\Users\xunying\AppData\Local\hermes\cache\scratch\kvarn-tiles"
SLICE = 128


def g128_quant_v_ref(v, bits, iters=8, mode="token"):
    """照抄 ref/g128_codecs.py 的 g128_quant_v（去掉 import 依赖）。"""
    def _rows_rtn(M, bits, iters=8, quantile=None):
        bal, sc, sr = V._sinkhorn(M, iters)
        q, scale, zp = V._rtn(bal, bits, quantile)
        return (scale.to(torch.float32) * q + zp.to(torch.float32)) * sc * sr

    def _blocks(x):
        H, T, Dd = x.shape
        nt = T // SLICE
        b = x.reshape(H, nt, SLICE, 2, SLICE).permute(0, 1, 3, 2, 4).reshape(H * nt * 2, SLICE, SLICE)
        return b, (H, nt)

    def _unblocks(b, H, nt):
        return b.reshape(H, nt, 2, SLICE, SLICE).permute(0, 1, 3, 2, 4).reshape(H, nt * SLICE, 256)

    def _v_to_channel_rows(vr):
        H, T, _ = vr.shape
        nt = T // SLICE
        M = vr.reshape(H, nt, SLICE, 2, SLICE).permute(0, 1, 3, 4, 2).reshape(H * nt, 2 * SLICE, SLICE)
        return M, (H, nt)

    def _v_from_channel_rows(M, H, nt):
        return M.reshape(H, nt, 2, SLICE, SLICE).permute(0, 1, 4, 2, 3).reshape(H, nt * SLICE, 256)

    vr = V.hadamard_256(v)
    H, T, _ = vr.shape
    if mode == "channel":
        M, (H2, nt) = _v_to_channel_rows(vr)
        M = _rows_rtn(M, bits, iters, None)
        out = _v_from_channel_rows(M, H2, nt)
    else:
        b, (H2, nt) = _blocks(vr)
        b = _rows_rtn(b, bits, iters, None)
        out = _unblocks(b, H2, nt)
    return V.hadamard_256(out)


def cos(a, b):
    a = a.reshape(-1, 256); b = b.reshape(-1, 256)
    return ((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)).mean().item()


for lay in (35, 63):
    v = torch.load(os.path.join(TILES, "layer%d_v_proj.pt" % lay), map_location="cpu").float()[0]
    v = v.reshape(256, 4, 256).permute(1, 0, 2)
    for mode in ("token", "channel"):
        for bits in (2, 3, 4):
            ref = g128_quant_v_ref(v, bits, 8, mode)
            mine = V.v_quant(v, bits=bits, mode=mode, variant="rtn")[0]
            d = (ref - mine).abs().max().item()
            rel = (ref - mine).norm().item() / ref.norm().item()
            print("layer%d %-8s b%d: max|Δ|=%.2e rel=%.2e  Vcos(ref)=%.6f  ⇒ %s"
                  % (lay, mode, bits, d, rel, cos(v, ref), "等价" if rel < 1e-5 else "不等价"))
