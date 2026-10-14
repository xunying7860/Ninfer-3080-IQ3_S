#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agg_llama.py —— 解析 out/*.txt，产出「等字节对照」与「等位宽对照」两张关键表。

只解析、不重算。字段口径与 bee_oracle 的 aggregate.py 一致（cos_glob/cos_vec_mean/
cos_vec_min/relL2/maxAbs），另加 bpt = 每 token·head-slice(128 dim) 的字节数。
"""
import glob
import os
import re
import sys
from collections import defaultdict

O = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# <data>/<mode>  <name>  bits=N|type=T  RxC tok=row|col rec=N B bpt=X | cos_glob=... ...
RE = re.compile(
    r"^(?P<data>[^/\s]+)/(?P<mode>\S+)\s+(?P<name>\S+)\s+(?P<spec>bits=\d+|type=\S+)\s+"
    r"(?P<rows>\d+)x(?P<cols>\d+)\s+tok=(?P<tok>row|col)\s+rec=\s*(?P<rec>\d+) B\s+bpt=\s*(?P<bpt>[\d.]+)\s*\|"
    r"\s*cos_glob=(?P<cg>[\d.]+)\s+cos_vec_mean=(?P<cvm>[\d.]+)\s+cos_vec_min=(?P<cmin>[\d.]+)\s+"
    r"relL2=(?P<rl2>[\d.]+)\s+maxAbs=(?P<ma>\S+)")


def parse_all():
    rows = []
    for fn in sorted(glob.glob(os.path.join(O, "*.txt"))):
        base = os.path.basename(fn)[:-4]
        if base in ("build", "f16check", "packtest", "table", "sinktest", "selftest", "bytes"):
            continue
        for ln in open(fn, encoding="utf-8", errors="replace"):
            m = RE.match(ln.strip())
            if not m:
                continue
            d = m.groupdict()
            rows.append(dict(
                file=base, data=d["data"], mode=d["mode"], name=d["name"], spec=d["spec"],
                kv="V" if d["tok"] == "row" else "K",
                rec=int(d["rec"]), bpt=float(d["bpt"]),
                cg=float(d["cg"]), cvm=float(d["cvm"]), cmin=float(d["cmin"]),
                rl2=float(d["rl2"]), ma=float(d["ma"])))
    return rows


def agg(rows):
    """(data, mode, spec, kv) -> 汇总"""
    g = defaultdict(list)
    for r in rows:
        g[(r["data"], r["mode"], r["spec"], r["kv"])].append(r)
    out = {}
    for k, v in g.items():
        n = len(v)
        out[k] = dict(
            n=n,
            bpt=sum(x["bpt"] for x in v) / n,
            cg=sum(x["cg"] for x in v) / n,
            cvm=sum(x["cvm"] for x in v) / n,
            cmin=min(x["cmin"] for x in v),
            rl2=sum(x["rl2"] for x in v) / n,
            ma=max(x["ma"] for x in v))
    return out


LABEL = {
    "kvarn": "KVarN", "rot-only": "rot-only", "bal-only": "bal-only", "neither": "neither",
    "llama": "llama", "llama-rot": "llama+rot",
}


def scheme_name(mode, spec):
    lab = LABEL.get(mode, mode)
    if spec.startswith("bits="):
        b = spec.split("=")[1]
        return "%s K%s" % (lab, b)
    return "%s %s" % (lab, spec.split("=")[1])


def main():
    rows = parse_all()
    if not rows:
        print("没有解析到任何行"); return 1
    A = agg(rows)
    print("解析到 %d 条 tile 行，%d 个 (data,mode,spec,kv) 组\n" % (len(rows), len(A)))

    # ---------- 明细 CSV ----------
    csvp = os.path.join(O, "agg_detail.csv")
    with open(csvp, "w", encoding="utf-8") as f:
        f.write("data,mode,spec,kv,scheme,n,bpt,cg,cvm,cmin,rl2,ma\n")
        for (data, mode, spec, kv), v in sorted(A.items()):
            f.write("%s,%s,%s,%s,%s,%d,%.3f,%.7f,%.7f,%.7f,%.5f,%.4g\n" % (
                data, mode, spec, kv, scheme_name(mode, spec), v["n"], v["bpt"],
                v["cg"], v["cvm"], v["cmin"], v["rl2"], v["ma"]))
    print("明细已写 %s\n" % csvp)

    # ---------- 1. 等字节对照 ----------
    for data in ["real_g128", "real_g64", "synth_gauss_g128", "synth_gauss_g64",
                 "synth_outlier_g128", "synth_outlier_g64"]:
        for kv in ["K", "V"]:
            items = [(k, v) for k, v in A.items() if k[0] == data and k[3] == kv]
            if not items:
                continue
            items.sort(key=lambda t: t[1]["bpt"])
            print("=" * 132)
            print("【等字节对照】data=%s  %s 侧（B/token·head-slice=128dim；误差单位=一条 token 的 256-dim head 向量）" % (data, kv))
            print("-" * 132)
            print("%-20s %7s | %10s %10s %10s %9s %9s" %
                  ("方案", "B/tok", "cos_glob", "cos_vec_mu", "cos_vec_min", "relL2", "maxAbs"))
            for (d, mode, spec, k2), v in items:
                print("%-20s %7.2f | %10.7f %10.7f %10.7f %9.5f %9.4g" %
                      (scheme_name(mode, spec), v["bpt"], v["cg"], v["cvm"], v["cmin"], v["rl2"], v["ma"]))
            print()

    # ---------- 2. 等字节「同预算内排序」----------
    print("#" * 132)
    print("【等字节预算内排序】把 bpt 归到最近的预算档，档内按 cos_glob 排序（越靠前越好）")
    print("#" * 132)
    BUDGETS = [38, 54, 56, 64, 70, 72, 80, 86, 88, 96, 102, 134, 136]
    for data in ["real_g128", "real_g64", "synth_gauss_g128", "synth_outlier_g128"]:
        for kv in ["K", "V"]:
            items = [(k, v) for k, v in A.items() if k[0] == data and k[3] == kv]
            if not items:
                continue
            buckets = defaultdict(list)
            for k, v in items:
                b = min(BUDGETS, key=lambda x: abs(x - v["bpt"]))
                if abs(b - v["bpt"]) > 3.5:
                    continue
                buckets[b].append((k, v))
            print("--- %s / %s ---" % (data, kv))
            for b in sorted(buckets):
                lst = sorted(buckets[b], key=lambda t: -t[1]["cg"])
                print("  预算 %3d B : " % b + "  >  ".join(
                    "%s(%.2fB,cos=%.7f,relL2=%.5f)" % (scheme_name(k[1], k[2]), v["bpt"], v["cg"], v["rl2"])
                    for k, v in lst))
            print()

    # ---------- 3. 等位宽对照 ----------
    print("#" * 132)
    print("【等位宽对照】同一 bits/value 下谁更好（KVarN K_nV_n vs llama q4_0/q4_1/q4_K ...）")
    print("#" * 132)
    BITS_GROUP = {
        3: ["q3_0", "q3_1"],
        4: ["q4_0", "iq4_nl", "q4_1", "q4_K"],
        5: ["q5_0", "q5_1"],
        8: ["q8_0"],
    }
    for data in ["real_g128", "real_g64", "synth_gauss_g128", "synth_outlier_g128"]:
        for kv in ["K", "V"]:
            print("--- %s / %s ---" % (data, kv))
            for bits in sorted(BITS_GROUP):
                cand = []
                types = set(BITS_GROUP[bits])
                for (d, mode, spec, k2), v in A.items():
                    if d != data or k2 != kv:
                        continue
                    if mode in ("kvarn", "rot-only") and spec == "bits=%d" % bits:
                        cand.append(((d, mode, spec, k2), v))
                    if mode in ("llama", "llama-rot") and spec.startswith("type=") and spec.split("=")[1] in types:
                        cand.append(((d, mode, spec, k2), v))
                if not cand:
                    continue
                cand.sort(key=lambda t: -t[1]["cg"])
                print("  %d-bit : " % bits + "  >  ".join(
                    "%s(%.2fB,cos=%.7f,relL2=%.5f)" % (scheme_name(k[1], k[2]), v["bpt"], v["cg"], v["rl2"])
                    for k, v in cand))
            print()
    return 0


def bits_to_type(bits):
    return {4: "q4_0", 5: "q5_0", 8: "q8_0"}.get(bits, "")


if __name__ == "__main__":
    sys.exit(main())
