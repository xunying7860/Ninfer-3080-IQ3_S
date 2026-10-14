#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 V 侧探索的全部读数：PPL（语料 A/B，Δ vs rk4v4）+ 字节账 + 重建指标。"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# 字节账：record 口径 = 32*(Kb+Vb)+24（父任务判据）；诚实 side info 口径另列
BYTES = {
    "rk4v4_post":        (280.0, "rk4v4 生产（K4 旋转 Lloyd + V4 G32）"),
    "K4V4_g128":         (280.0, "K4 + V4 token（零代价点）"),
    "K4V3_g128":         (248.0, "K4 + V3 token（上游次小档）"),
    "V2only_g128":       (None, "只压 V2（诊断臂）"),
    "K4V3chan_ls1":      (248.0, "V 通道为行 3bit + 加权LS(refine1)"),
    "K4V3chan_lsq01r1":  (248.0, "V 通道为行 3bit + LS + 1% 分位裁剪"),
    "K4V3chan_ls2":      (248.0, "V 通道为行 3bit + LS(refine2)"),
    "K4V3chan_m4333":    (256.0, "V per-head [4,3,3,3] 通道为行 + LS1"),
    "K4V3chan_m4433":    (264.0, "V per-head [4,4,3,3] 通道为行 + LS1"),
    "K4V3tok_lsq01r1":   (248.0, "V token 3bit + LS + 裁剪（免费旋钮对照）"),
    "K4V2chan_ls4":      (216.0, "V 通道为行 2bit + LS(refine4)"),
    "K4V3chan_lsq005r2": (248.0, "V 通道为行 3bit + LS + 0.5% 裁剪(refine2)"),
    "K4V3chan_lsq01r2":  (248.0, "V 通道为行 3bit + LS + 1% 裁剪(refine2)"),
    "K4Vprom_k16":       (250.0, "V 通道为行 3bit + 16/256 通道提 4bit（掩码计费）"),
    "K4Vprom_k32":       (252.0, "V 通道为行 3bit + 32/256 通道提 4bit"),
    "K4Vprom_k64":       (256.0, "V 通道为行 3bit + 64/256 通道提 4bit"),
    "K4Vprom_k128":      (264.0, "V 通道为行 3bit + 128/256 通道提 4bit"),
    "baseline_fp16":     (None, "K/V 全 fp16 上限参照"),
}
ORDER = ["baseline_fp16", "rk4v4_post", "K4V4_g128", "K4V3_g128", "V2only_g128",
         "K4V3chan_lsq01r1", "K4V3chan_lsq005r2", "K4V3chan_lsq01r2", "K4V3chan_ls1",
         "K4V3chan_ls2", "K4V3chan_m4333", "K4V3chan_m4433", "K4V3tok_lsq01r1",
         "K4V2chan_ls4", "K4Vprom_k16", "K4Vprom_k32", "K4Vprom_k64", "K4Vprom_k128"]
LINE = re.compile(r"^(\S+)\s+mean NLL=([\d.]+) PPL=([\d.]+).*?V cos=([\d.]+)(?:\s*\|\s*attn cos=([\d.]+))?")


def parse_logs():
    """返回 {corpus_tag: {arm: dict(ppl, vcos, acos)}}"""
    res = {}
    for fn in sorted(glob.glob(os.path.join(HERE, "logs", "*.log"))) + \
              sorted(glob.glob(os.path.join(HERE, "logs", "*.tsv"))):
        tag = "ppl_doc.txt" if "A_run1" in fn else os.path.basename(fn)
        for ln in open(fn, encoding="utf-8", errors="ignore"):
            m = LINE.match(ln.strip())
            if m:
                res.setdefault(tag, {})[m.group(1)] = dict(
                    ppl=float(m.group(3)), vcos=float(m.group(4)),
                    acos=float(m.group(5)) if m.group(5) else float("nan"))
    # results.tsv（x99 上的续跑产物，若已拉回）
    for fn in glob.glob(os.path.join(HERE, "logs", "results*.tsv")):
        for ln in open(fn, encoding="utf-8"):
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 7:
                res.setdefault(p[0], {})[p[1]] = dict(ppl=float(p[3]), vcos=float(p[5]), acos=float(p[6]))
    return res


def main():
    res = parse_logs()
    print("读到的语料表：%s" % {k: len(v) for k, v in res.items()})
    for tag, arms in res.items():
        base = arms.get("rk4v4_post", {}).get("ppl")
        print("\n### 语料 %s（基准 rk4v4 = %s）" % (tag, base))
        print("| 臂 | record B | PPL | Δ vs rk4v4 | V cos | attn cos | 说明 |")
        print("|---|---:|---:|---:|---:|---:|---|")
        for a in ORDER:
            if a not in arms:
                continue
            d = arms[a]
            by = BYTES.get(a, (None, ""))[0]
            delta = ("%+.4f%%" % (100.0 * (d["ppl"] / base - 1.0))) if base else "-"
            print("| %s | %s | %.4f | %s | %.6f | %.6f | %s |" % (
                a, ("%g" % by) if by else "-", d["ppl"], delta, d["vcos"], d["acos"], BYTES.get(a, (None, ""))[1]))


if __name__ == "__main__":
    main()
