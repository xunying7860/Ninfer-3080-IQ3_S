#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aggregate.py —— 汇总 out/*.txt 里的原始读数，产出报告用的数字（不重算，只解析）。"""
import glob
import os
import re
import statistics

O = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

REAL = r"^(\S+)\s+bits=(\d+) (\d+)x(\d+) tok=(\w+) rec=\s*(\d+) B \| cos_glob=([\d.]+) cos_vec_mean=([\d.]+) cos_vec_min=([\d.]+) relL2=([\d.]+) maxAbs=(\S+)"


def parse(fn):
    rows = []
    for ln in open(os.path.join(O, fn), encoding="utf-8"):
        m = re.match(REAL, ln)
        if m:
            rows.append(dict(name=m.group(1), bits=int(m.group(2)), rows=int(m.group(3)), cols=int(m.group(4)),
                             cos_glob=float(m.group(7)), cos_vec=float(m.group(8)),
                             cos_vec_min=float(m.group(9)), relL2=float(m.group(10))))
    return rows


def summ(rows, label):
    k = [r for r in rows if "_k_" in r["name"]]
    v = [r for r in rows if "_v_" in r["name"]]
    f = lambda x, key: statistics.fmean(a[key] for a in x) if x else float("nan")
    return dict(cfg=label, K_cos=f(k, "cos_glob"), V_cos=f(v, "cos_glob"),
                K_min=min((a["cos_vec_min"] for a in k), default=float("nan")),
                V_min=min((a["cos_vec_min"] for a in v), default=float("nan")),
                K_relL2=f(k, "relL2"), V_relL2=f(v, "relL2"))


def main():
    print("配置                     K cos_glob  V cos_glob   K relL2   V relL2")
    res = {}
    for fn, lab in [("real_g128_k4v2.txt", "K4V2_g128"), ("real_g128_k4v4.txt", "K4V4_g128"),
                    ("real_g64_k4v2.txt", "K4V2_g64"), ("real_g64_k4v4.txt", "K4V4_g64"),
                    ("real_g128_k4v4_cpuorder.txt", "K4V4_g128_cpuorder"),
                    ("real_g128_k4v4_iters8.txt", "K4V4_g128_iters8")]:
        if os.path.exists(os.path.join(O, fn)):
            r = summ(parse(fn), lab); res[lab] = r
            print("%-24s %9.7f  %9.7f  %8.5f  %8.5f" % (lab, r["K_cos"], r["V_cos"], r["K_relL2"], r["V_relL2"]))
    print("\nG128 vs G64（真实数据，同一 token 范围）")
    for k, v in [(4, 2), (4, 4)]:
        a, b = res["K%dV%d_g64" % (k, v)], res["K%dV%d_g128" % (k, v)]
        print("  K%dV%d: K %+.7f  V %+.7f  (K relL2 %.5f->%.5f, V relL2 %.5f->%.5f)" % (
            k, v, b["K_cos"] - a["K_cos"], b["V_cos"] - a["V_cos"],
            a["K_relL2"], b["K_relL2"], a["V_relL2"], b["V_relL2"]))
    print("\n合成数据（>>> 汇总行）")
    for f in sorted(glob.glob(os.path.join(O, "synth_*.txt"))):
        for ln in open(f, encoding="utf-8"):
            m = re.search(r">>> (\S+) 汇总\((\d+) tiles\): K cos_glob=([\d.]+) cos_vec_mean=([\d.]+) \| V cos_glob=([\d.]+) cos_vec_mean=([\d.]+)", ln)
            if m:
                print("  %-26s tiles=%-3s K=%.7f V=%.7f" % (os.path.basename(f)[:-4], m.group(2), float(m.group(3)), float(m.group(5))))
    print("\n关键行（table/packtest/f16check/sinktest）")
    for fn, pat in [("table.txt", "不一致格数"), ("table.txt", "record 口径 ="),
                    ("packtest.txt", "失败项"), ("f16check.txt", "总计不一致"), ("sinktest.txt", "max\\|s_col_gpu")]:
        p = os.path.join(O, fn)
        if os.path.exists(p):
            for ln in open(p, encoding="utf-8"):
                if re.search(pat, ln):
                    print("  [%s] %s" % (fn, ln.strip()))


if __name__ == "__main__":
    main()
