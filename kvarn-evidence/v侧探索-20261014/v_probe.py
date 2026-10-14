#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V 侧重建探针：真实 pre-RoPE V（layer35/63 v_proj.pt）上跑 v_codecs 的全部变体，出表。"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_codecs as V  # noqa: E402

TILES = r"C:\Users\xunying\AppData\Local\hermes\cache\scratch\kvarn-tiles"


def load_v():
    """返回 [2层, 4head, 256tok, 256dim] 的 V（pre-RoPE）。"""
    outs = []
    for lay in (35, 63):
        p = os.path.join(TILES, "layer%d_v_proj.pt" % lay)
        t = torch.load(p, map_location="cpu")
        t = t.float()
        if t.dim() == 3 and t.shape[0] == 1:
            t = t[0]
        if t.shape == (256, 1024):
            t = t.reshape(256, 4, 256).permute(1, 0, 2)
        elif t.shape == (1024, 256):
            t = t.permute(1, 0).reshape(256, 4, 256).permute(1, 0, 2)
        assert t.shape == (4, 256, 256), t.shape
        outs.append(t)
    return outs


ARMS = [
    # (标签, bits, kwargs)
    ("V3_rtn",        3, dict(variant="rtn")),
    ("V4_rtn",        4, dict(variant="rtn")),
    ("V2_rtn",        2, dict(variant="rtn")),
    ("V3_ls",         3, dict(variant="ls")),
    ("V3_ls1",        3, dict(variant="ls", refine=1)),
    ("V3_ls2",        3, dict(variant="ls", refine=2)),
    ("V3_lsr",        3, dict(variant="ls", wmode="one")),
    ("V3_lsq005",     3, dict(variant="ls", quantile=0.005)),
    ("V3_lsq01",      3, dict(variant="ls", quantile=0.01)),
    ("V3_lsq02",      3, dict(variant="ls", quantile=0.02)),
    ("V3_sinkmse",    3, dict(variant="rtn", sink_pick="mse")),
    ("V3_ls_sinkmse", 3, dict(variant="ls", sink_pick="mse")),
    ("V3_ls_it16",    3, dict(variant="ls", iters=16)),
    ("V4_ls",         4, dict(variant="rtn")),
    ("V2_ls",         2, dict(variant="ls")),
    ("V2_lsq01",      2, dict(variant="ls", quantile=0.01)),
    ("V2_chan",       2, dict(variant="rtn", mode="channel")),
    ("V3_chan",       3, dict(variant="rtn", mode="channel")),
    ("V3_noRot",      3, dict(variant="rtn", rotate=False)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="35,63")
    ap.add_argument("--arms", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    vs = load_v()
    lines = []
    hdr = ("%-14s %4s %8s %11s %11s %11s %11s %11s" %
           ("臂", "bits", "recB", "Vcos_mean", "Vcos_min", "relL2_mean", "relL2_max", "relL2_glob"))
    lines.append(hdr)
    print(hdr, flush=True)
    want = a.arms.split(",") if a.arms else None
    for lab, bits, kw in ARMS:
        if want and lab not in want:
            continue
        t0 = time.time()
        cs, cm, rl, rx, rg = [], [], [], [], []
        for lay, x in zip((35, 63), vs):
            xh, bpe, _ = V.v_quant(x, bits=bits, **kw)
            m = V.metrics(x, xh)
            cs.append(m["cos_mean"]); cm.append(m["cos_min"])
            rl.append(m["relL2_mean"]); rx.append(m["relL2_max"]); rg.append(m["relL2_glob"])
        rec = V.record_bytes(4, bits)
        line = ("%-14s %4d %8.0f %11.6f %11.6f %11.6f %11.6f %11.6f" %
                (lab, bits, rec, sum(cs) / len(cs), min(cm), sum(rl) / len(rl), max(rx), sum(rg) / len(rg)))
        lines.append(line)
        print(line + "   (%.1fs)" % (time.time() - t0), flush=True)

    # 残差二段（4bit 总量）
    t0 = time.time()
    cs, cm, rl = [], [], []
    for x in vs:
        xh, bpe, _ = V.v_quant_res(x, bits_main=3)
        m = V.metrics(x, xh)
        cs.append(m["cos_mean"]); cm.append(m["cos_min"]); rl.append(m["relL2_mean"])
    line = ("%-14s %4s %8.0f %11.6f %11.6f %11.6f %11s %11s" %
            ("V3+1res", "3+1", V.record_bytes(4, 4), sum(cs) / len(cs), min(cm),
             sum(rl) / len(rl), "-", "-"))
    lines.append(line)
    print(line + "   (%.1fs)" % (time.time() - t0), flush=True)

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
