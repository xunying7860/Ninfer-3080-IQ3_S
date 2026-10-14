#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V 侧重建探针 · 第四轮：通道为行 + 逐 tile 通道提位（混合位宽掩码）。"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_codecs as V  # noqa: E402

TILES = r"C:\Users\xunying\AppData\Local\hermes\cache\scratch\kvarn-tiles"


def load_v():
    outs = []
    for lay in (35, 63):
        t = torch.load(os.path.join(TILES, "layer%d_v_proj.pt" % lay), map_location="cpu").float()
        if t.dim() == 3 and t.shape[0] == 1:
            t = t[0]
        outs.append(t.reshape(256, 4, 256).permute(1, 0, 2))
    return outs


vs = load_v()
print("%-26s %7s %8s %11s %11s %11s %11s" % ("臂", "bits", "bpe", "Vcos_mean", "Vcos_min", "relL2_mean", "relL2_max"))
for k in (8, 16, 32, 64, 128):
    for q in (None, 0.01):
        cs, cm, rl, rx = [], [], [], []
        for x in vs:
            xh, ab, bpe, _ = V.v_quant_promote(x, 3, 4, k, variant="ls", quantile=q, refine=1)
            m = V.metrics(x, xh)
            cs.append(m["cos_mean"]); cm.append(m["cos_min"]); rl.append(m["relL2_mean"]); rx.append(m["relL2_max"])
        print("%-26s %7.4f %8.2f %11.6f %11.6f %11.6f %11.6f" % (
            "promote_k%d_q%s" % (k, q), ab, bpe, sum(cs) / len(cs), min(cm), sum(rl) / len(rl), max(rx)))
# 参照
for lab, fn, bits in [("chan3_lsq01_r1", lambda x: V.v_quant(x, 3, mode="channel", variant="ls", quantile=0.01, refine=1)[0], 3),
                      ("v4tok_rtn", lambda x: V.v_quant(x, 4, mode="token", variant="rtn")[0], 4),
                      ("chan4_ls_r1", lambda x: V.v_quant(x, 4, mode="channel", variant="ls", refine=1)[0], 4)]:
    cs, rl = [], []
    for x in vs:
        m = V.metrics(x, fn(x))
        cs.append(m["cos_mean"]); rl.append(m["relL2_mean"])
    print("%-26s %7d %8s %11.6f %11s %11.6f" % (lab, bits, "-", sum(cs) / len(cs), "-", sum(rl) / len(rl)))
