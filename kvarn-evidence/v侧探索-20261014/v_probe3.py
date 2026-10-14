#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V 侧重建探针 · 第三轮：尺度求解细化（refine×clip）+ 行宽 + per-head 混合位宽。"""
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
        outs.append(t.reshape(256, 4, 256).permute(1, 0, 2))
    return outs


def run(vs, fn, **kw):
    cs, cm, rl, rx, rg = [], [], [], [], []
    for x in vs:
        xh, bpe, _ = fn(x, **kw)
        m = V.metrics(x, xh)
        cs.append(m["cos_mean"]); cm.append(m["cos_min"]); rl.append(m["relL2_mean"])
        rx.append(m["relL2_max"]); rg.append(m["relL2_glob"])
    return (sum(cs) / len(cs), min(cm), sum(rl) / len(rl), max(rx), sum(rg) / len(rg), bpe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    vs = load_v()
    lines = []

    def emit(tag, bits, kw, fn=None):
        fn = fn or V.v_quant
        c, cmin, r, rmax, rg, bpe = run(vs, fn, bits=bits, **kw)
        rec = V.record_bytes(4, bits)
        line = "%-22s %6s %8.0f %10.1f %11.6f %11.6f %11.6f %10.5f" % (
            tag, bits, rec, bpe, c, cmin, r, rmax)
        print(line, flush=True)
        lines.append(line)

    hdr = "%-22s %6s %8s %10s %11s %11s %11s %10s" % ("臂", "bits", "recB", "bpe_honest", "Vcos_mean", "Vcos_min", "relL2_mean", "relL2_max")
    print(hdr, flush=True); lines.append(hdr)

    print("--- 通道为行(行宽1) · 3bit · 尺度求解 ---", flush=True)
    for ref in (0, 1, 2, 3):
        emit("chan3_ls_r%d" % ref, 3, dict(mode="channel", variant="ls", refine=ref))
    for q in (0.005, 0.01, 0.02):
        for ref in (0, 1):
            emit("chan3_lsq%g_r%d" % (q, ref), 3, dict(mode="channel", variant="ls", quantile=q, refine=ref))
    print("--- 通道为行(行宽1) · 2bit ---", flush=True)
    for ref in (0, 1, 2, 3, 4):
        emit("chan2_ls_r%d" % ref, 2, dict(mode="channel", variant="ls", refine=ref))
    for q in (0.01, 0.02, 0.05):
        emit("chan2_lsq%g_r1" % q, 2, dict(mode="channel", variant="ls", quantile=q, refine=1))
    print("--- 通道为行(行宽1) · 4bit ---", flush=True)
    for ref in (0, 1):
        emit("chan4_ls_r%d" % ref, 4, dict(mode="channel", variant="ls", refine=ref))
    print("--- 行宽扫描 · 3bit ls1 ---", flush=True)
    for W in (1, 2, 4, 8, 16, 32):
        emit("chan3_W%d_ls1" % W, 3, dict(mode="channel", row_dims=W, variant="ls", refine=1))
    print("--- token 侧免费改进（对照）---", flush=True)
    emit("tok3_rtn", 3, dict(variant="rtn"))
    emit("tok3_ls1", 3, dict(variant="ls", refine=1))
    emit("tok3_lsq01_r1", 3, dict(variant="ls", quantile=0.01, refine=1))
    print("--- per-head 混合位宽（通道为行 + ls1）---", flush=True)
    for bph in ([3, 3, 3, 3], [4, 3, 3, 3], [4, 4, 3, 3], [4, 4, 4, 3], [5, 3, 3, 3], [4, 3, 3, 2]):
        cs, cm, rl, rx = [], [], [], []
        for x in vs:
            xh, ab, bpe, _ = V.v_quant_mixed(x, bph, mode="channel", variant="ls", refine=1)
            m = V.metrics(x, xh)
            cs.append(m["cos_mean"]); cm.append(m["cos_min"]); rl.append(m["relL2_mean"]); rx.append(m["relL2_max"])
        ab = sum(bph) / 4.0
        line = "%-22s %6.2f %8.0f %10.0f %11.6f %11.6f %11.6f %10.5f" % (
            "mix%s_chan_ls1" % "".join(str(b) for b in bph), ab, V.record_bytes(4, ab), 256 * ab / 8,
            sum(cs) / len(cs), min(cm), sum(rl) / len(rl), max(rx))
        print(line, flush=True); lines.append(line)

    if a.out:
        open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
