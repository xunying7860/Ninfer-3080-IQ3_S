#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三语料 Δ 汇总（round2 终版，全精度）。
数据来源：x99 /data/kvarn/vside-20261014/results.tsv（29 行，全精度；本目录 results_after_round2.tsv 为副本）。
语料：A=ppl_doc.txt(2944 tok) / B=ppl_doc2.txt(6016 tok) / C=ppl_doc3.txt(2432 tok，本轮到齐锚点)。
判据（父任务）：record < 280 B 且 三语料均值 Δ ≤ +0.10% 且 无单语料 Δ > +0.30%。
附加：摆幅 > 0.10%（=判据本身）⇒ 单语料口径不可判定。
"""
import statistics as st

DOCS = ["A", "B", "C"]
ANCHOR = {"A": 7.915895, "B": 8.455766, "C": 7.482736}   # rk4v4_post（本轮实测）

PPL = {
    "baseline_fp16":    (7.891693, 8.442825, 7.458948),
    "rk4v4_post":       (7.915895, 8.455766, 7.482736),
    "K4V4_g128":        (7.908848, 8.474659, 7.480831),
    "K4V3chan_lsq01r1": (7.925306, 8.482036, 7.488645),
    "K4V3chan_ls1":     (7.934748, 8.451625, 7.479288),
    "K4Vprom_k8":       (7.910308, 8.478178, 7.480992),
    "K4Vprom_k16":      (7.908859, 8.466968, 7.491290),
    "K4Vprom_k32":      (7.925583, 8.460535, 7.490566),
    "K4Vprom_k64":      (7.925366, 8.473052, 7.486826),
    "K4Vprom_k128":     (7.927300, 8.444555, 7.487089),
}
BYTES = {"baseline_fp16": None, "rk4v4_post": 280, "K4V4_g128": 280,
         "K4V3chan_lsq01r1": 248, "K4V3chan_ls1": 248,
         "K4Vprom_k8": 249, "K4Vprom_k16": 250, "K4Vprom_k32": 252,
         "K4Vprom_k64": 256, "K4Vprom_k128": 264}
ORDER = ["baseline_fp16", "rk4v4_post", "K4V4_g128", "K4V3chan_lsq01r1", "K4V3chan_ls1",
         "K4Vprom_k8", "K4Vprom_k16", "K4Vprom_k32", "K4Vprom_k64", "K4Vprom_k128"]

rows = {}
print("| 臂 | record B | Δ_A | Δ_B | Δ_C | 三语料均值 | 最大摆幅 | 判定 |")
print("|---|---:|---:|---:|---:|---:|---:|---|")
for arm in ORDER:
    d = {doc: 100.0 * (PPL[arm][i] / ANCHOR[doc] - 1.0) for i, doc in enumerate(DOCS)}
    mean, swing, worst = st.fmean(d.values()), max(d.values()) - min(d.values()), max(d.values())
    b = BYTES[arm]
    if b is None:
        v = "参照"
    elif b >= 280:
        v = "字节不省"
    elif not (mean <= 0.10 and worst <= 0.30):
        v = "判死"
    elif swing > 0.10:
        v = "候选·不可判定"
    else:
        v = "**真候选（可判定）**"
    rows[arm] = (d, mean, swing, worst, b, v)
    print("| %s | %s | %+.4f%% | %+.4f%% | %+.4f%% | %+.4f%% | %.4f%% | %s |"
          % (arm, b if b else "–", d["A"], d["B"], d["C"], mean, swing, v))

print("\n## 三条件 + 可判定性逐臂核对")
print("| 臂 | 字节<280 | 均值≤+0.10% | 最差语料≤+0.30% | 摆幅≤+0.10%(可判定) | 结论 |")
print("|---|---|---|---|---|---|")
for arm in ORDER:
    d, mean, swing, worst, b, v = rows[arm]
    if b is None:
        print("| %s | – | – | – | – | 参照 |" % arm); continue
    print("| %s | %s | %s | %s | %s | %s |" % (arm, "是" if b < 280 else "否",
          "是" if mean <= 0.10 else "否", "是" if worst <= 0.30 else "否",
          "是" if swing <= 0.10 else "否", v))

print("\n## 锚点自检（三语料）")
for i, doc in enumerate(DOCS):
    print("  %s: baseline %.6f  rk4v4 %.6f  K4V4 %.6f  (Δ K4V4 = %+.4f%%)"
          % (doc, PPL["baseline_fp16"][i], PPL["rk4v4_post"][i], PPL["K4V4_g128"][i],
             100.0 * (PPL["K4V4_g128"][i] / ANCHOR[doc] - 1.0)))

print("\n## k16 关键对照（“250 B 同质量”是否成立）")
for i, doc in enumerate(DOCS):
    k16, k4, rk = PPL["K4Vprom_k16"][i], PPL["K4V4_g128"][i], PPL["rk4v4_post"][i]
    print("  %s: k16=%.6f | vs rk4v4 %+.4f%% | vs K4V4(280B) %+.4f%%"
          % (doc, k16, 100 * (k16 / rk - 1), 100 * (k16 / k4 - 1)))

print("\n## 摆幅排行（判据 0.10%）")
for s, a in sorted(((rows[a][2], a) for a in ORDER if BYTES[a] is not None), reverse=True):
    print("  %-18s %6.4f%%  = 判据的 %.1f 倍" % (a, s, s / 0.10))
