#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V 侧重建探针 · 第二轮：通道为行（chan）家族 + 尺度求解 + per-head 混合位宽。"""
import argparse
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
        t = t.reshape(256, 4, 256).permute(1, 0, 2)
        assert t.shape == (4, 256, 256), t.shape
        outs.append(t)
    return outs


ARMS = [
    ("V3_chan",          3, dict(mode="channel")),
    ("V3_chan_ls",       3, dict(mode="channel", variant="ls")),
    ("V3_chan_ls1",      3, dict(mode="channel", variant="ls", refine=1)),
    ("V3_chan_ls2",      3, dict(mode="channel", variant="ls", refine=2)),
    ("V3_chan_lsq005",   3, dict(mode="channel", variant="ls", quantile=0.005)),
    ("V3_chan_lsq01",    3, dict(mode="channel", variant="ls", quantile=0.01)),
    ("V3_chan_lsq02",    3, dict(mode="channel", variant="ls", quantile=0.02)),
    ("V3_chan_lsq05",    3, dict(mode="channel", variant="ls", quantile=0.05)),
    ("V3_chan_sinkmse",  3, dict(mode="channel", sink_pick="mse")),
    ("V3_chan_ls_smse",  3, dict(mode="channel", variant="ls", sink_pick="mse")),
    ("V3_chan_it16",     3, dict(mode="channel", iters=16)),
    ("V3_chan_it4",      3, dict(mode="channel", iters=4)),
    ("V3_chan_ls_it16",  3, dict(mode="channel", variant="ls", iters=16)),
    ("V4_chan",          4, dict(mode="channel")),
    ("V4_chan_lsq01",    4, dict(mode="channel", variant="ls", quantile=0.01)),
    ("V2_chan_ls",       2, dict(mode="channel", variant="ls")),
    ("V2_chan_ls1",      2, dict(mode="channel", variant="ls", refine=1)),
    ("V2_chan_lsq01",    2, dict(mode="channel", variant="ls", quantile=0.01)),
    ("V2_chan_lsq02",    2, dict(mode="channel", variant="ls", quantile=0.02)),
    ("V2_chan_lsq05",    2, dict(mode="channel", variant="ls", quantile=0.05)),
]

MIXED = [
    ("M4333_chan_ls",  [4, 3, 3, 3], dict(mode="channel", variant="ls")),
    ("M4433_chan_ls",  [4, 4, 3, 3], dict(mode="channel", variant="ls")),
    ("M4443_chan_ls",  [4, 4, 4, 3], dict(mode="channel", variant="ls")),
    ("M4333tok_lsq",   [4, 3, 3, 3], dict(mode="token", variant="ls", quantile=0.01)),
    ("M4433tok_lsq",   [4, 4, 3, 3], dict(mode="token", variant="ls", quantile=0.01)),
    ("M4443tok_lsq",   [4, 4, 4, 3], dict(mode="token", variant="ls", quantile=0.01)),
    ("M5333_chan_ls",  [5, 3, 3, 3], dict(mode="channel", variant="ls")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-mixed", action="store_true")
    a = ap.parse_args()
    vs = load_v()
    lines = []
    hdr = "%-16s %5s %8s %11s %11s %11s %11s" % ("臂", "bits", "recB", "Vcos_mean", "Vcos_min", "relL2_mean", "relL2_max")
    print(hdr, flush=True); lines.append(hdr)
    for lab, bits, kw in ARMS:
        cs, cm, rl, rx = [], [], [], []
        for x in vs:
            xh, bpe, _ = V.v_quant(x, bits=bits, **kw)
            m = V.metrics(x, xh)
            cs.append(m["cos_mean"]); cm.append(m["cos_min"]); rl.append(m["relL2_mean"]); rx.append(m["relL2_max"])
        line = "%-16s %5d %8.0f %11.6f %11.6f %11.6f %11.6f" % (
            lab, bits, V.record_bytes(4, bits), sum(cs) / len(cs), min(cm), sum(rl) / len(rl), max(rx))
        print(line, flush=True); lines.append(line)
    if not a.skip_mixed:
        for lab, bph, kw in MIXED:
            cs, cm, rl, rx = [], [], [], []
            for x in vs:
                xh, ab, bpe, _ = V.v_quant_mixed(x, bph, **kw)
                m = V.metrics(x, xh)
                cs.append(m["cos_mean"]); cm.append(m["cos_min"]); rl.append(m["relL2_mean"]); rx.append(m["relL2_max"])
            line = "%-16s %5.2f %8.0f %11.6f %11.6f %11.6f %11.6f" % (
                lab, sum(bph) / 4.0, V.record_bytes(4, sum(bph) / 4.0), sum(cs) / len(cs), min(cm),
                sum(rl) / len(rl), max(rx))
            print(line, flush=True); lines.append(line)

    # per-head 明细（token 3bit / chan 3bit / 4bit 参照）
    print("\n=== per-head V 明细（cos_mean / relL2_mean，层35/63 平均）===", flush=True)
    lines.append("")
    lines.append("=== per-head 明细 ===")
    for lab, bits, kw in [("V3_rtn", 3, dict(variant="rtn")), ("V4_rtn", 4, dict(variant="rtn")),
                          ("V3_chan", 3, dict(mode="channel")), ("V3_chan_ls", 3, dict(mode="channel", variant="ls")),
                          ("V4_chan", 4, dict(mode="channel"))]:
        per = []
        for h in range(4):
            cs, rl = [], []
            for x in vs:
                xh, _, _ = V.v_quant(x[h:h + 1], bits=bits, **kw)
                m = V.metrics(x[h:h + 1], xh)
                cs.append(m["cos_mean"]); rl.append(m["relL2_mean"])
            per.append("h%d %.5f/%.4f" % (h, sum(cs) / len(cs), sum(rl) / len(rl)))
        line = "  %-12s %s" % (lab, " | ".join(per))
        print(line, flush=True); lines.append(line)

    if a.out:
        open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
