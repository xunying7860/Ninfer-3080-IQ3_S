#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三语料 Δ 汇总（round2）。数据来源：
 - results.tsv（x99 /data/kvarn/vside-20261014/results.tsv，27 行，全精度）
 - A 侧 chan 两臂只有 logs/ppl_A_run1.log 的 4 位小数（results.tsv 无该行，如实标注）
判据：record < 280 B 且 三语料均值 Δ ≤ +0.10% 且 无单语料 Δ > +0.30%
"""
import statistics as st

# 语料：A=ppl_doc.txt(2944 tok) B=ppl_doc2.txt(6016) C=ppl_doc3.txt(2432)
DOCS = ["A", "B", "C"]
ANCHOR = {"A": 7.915895, "B": 8.455766, "C": 7.482736}   # rk4v4_post

PPL = {
    # 臂:                A                B                C
    "baseline_fp16":  (7.891693,       8.442825,        7.458948),
    "rk4v4_post":     (7.915895,       8.455766,        7.482736),
    "K4V4_g128":      (7.908848,       8.474659,        7.480831),
    "K4V3chan_lsq01r1": (7.9253,       8.482036,        7.488645),   # A 侧 4dp
    "K4V3chan_ls1":   (7.9347,         8.451625,        7.479288),   # A 侧 4dp
    "K4Vprom_k8":     (7.910308,       8.478178,        7.480992),
    "K4Vprom_k16":    (7.908859,       8.466968,        7.491290),
    "K4Vprom_k32":    (7.925583,       8.460535,        7.490566),
    "K4Vprom_k64":    (7.925366,       8.473052,        7.486826),
    "K4Vprom_k128":   (7.927300,       8.444555,        7.487089),
}

# record B/token·head-slice（公式口径 32*(Kb+Vb)+24）
BYTES = {
    "baseline_fp16": None, "rk4v4_post": 280, "K4V4_g128": 280,
    "K4V3chan_lsq01r1": 248, "K4V3chan_ls1": 248,
    "K4Vprom_k8": 249, "K4Vprom_k16": 250, "K4Vprom_k32": 252,
    "K4Vprom_k64": 256, "K4Vprom_k128": 264,
}
ORDER = ["baseline_fp16", "rk4v4_post", "K4V4_g128", "K4V3chan_lsq01r1", "K4V3chan_ls1",
         "K4Vprom_k8", "K4Vprom_k16", "K4Vprom_k32", "K4Vprom_k64", "K4Vprom_k128"]

print("| 臂 | record B | Δ_A | Δ_B | Δ_C | 三语料均值 | 最大摆幅 | 判定 |")
print("|---|---:|---:|---:|---:|---:|---:|---|")
rows = {}
for arm in ORDER:
    d = {}
    for i, doc in enumerate(DOCS):
        d[doc] = 100.0 * (PPL[arm][i] / ANCHOR[doc] - 1.0)
    mean = st.fmean(d.values())
    swing = max(d.values()) - min(d.values())
    worst = max(d.values())
    b = BYTES[arm]
    if b is None:
        verdict = "参照"
    else:
        ok = (b < 280) and (mean <= 0.10) and (worst <= 0.30)
        verdict = "候选" if ok else ("不过" if b < 280 else "字节不省")
    rows[arm] = (d, mean, swing, worst, b, verdict)
    print("| %s | %s | %+.4f%% | %+.4f%% | %+.4f%% | %+.4f%% | %.4f%% | %s |"
          % (arm, b if b else "–", d["A"], d["B"], d["C"], mean, swing, verdict))

print()
print("## 判据三条件逐臂核对（字节<280 且 均值≤+0.10% 且 最差语料≤+0.30%）")
print("| 臂 | 字节<280 | 均值≤+0.10% | 最差语料≤+0.30% | 摆幅>0.10%⇒不可判定 | 判定 |")
print("|---|---|---|---|---|---|")
for arm in ORDER:
    d, mean, swing, worst, b, verdict = rows[arm]
    if b is None:
        print("| %s | – | – | – | – | 参照 |" % arm)
        continue
    print("| %s | %s | %s | %s | %s | %s |"
          % (arm, "是" if b < 280 else "否", "是" if mean <= 0.10 else "否",
             "是" if worst <= 0.30 else "否", "是" if swing > 0.10 else "否", verdict))

print()
print("## 锚点自检（应与阶段 0 / 首轮一字不差）")
for doc, i in zip(DOCS, range(3)):
    print("  %s: baseline %.6f rk4v4 %.6f K4V4 %.6f  (Δ K4V4 = %+.4f%%)"
          % (doc, PPL["baseline_fp16"][i], PPL["rk4v4_post"][i], PPL["K4V4_g128"][i],
             100.0 * (PPL["K4V4_g128"][i] / ANCHOR[doc] - 1.0)))

print()
print("## 摆幅统计（判据 = 0.10%）")
sw = sorted(((rows[a][2], a) for a in ORDER if BYTES[a] is not None), reverse=True)
for s, a in sw:
    print("  %-18s 摆幅 %.4f%%  (判据的 %.1f 倍)" % (a, s, s / 0.10))
