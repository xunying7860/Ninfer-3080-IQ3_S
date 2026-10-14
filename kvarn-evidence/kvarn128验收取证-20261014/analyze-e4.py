#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze-e4.py <KV128.json> <RK4V4.json> <INT8.json>
输出三臂逐 (id,run) 的引擎自报行指标（TTFT / cache% / prefill / decode / mtp / mixed / out），
以及各臂在长答题（Q1/Q2）与全部题上的均值，供「吞吐有差」判定（TTFT 相对差>15% 或 decode 相对差>10%）。"""
import json, sys

arms = {}
for p in sys.argv[1:]:
    d = json.load(open(p, encoding="utf-8"))
    arms[d["arm"]] = d

IDS = ["F1", "F2", "C1", "C2", "Z1", "Z2", "Q1", "Q2"]


def f(x):
    try:
        return float(x)
    except Exception:
        return None


for arm, d in arms.items():
    print("===== %s =====" % arm)
    print("%-4s %-4s %-8s %-7s %-10s %-10s %-8s %-22s %-6s" %
          ("id", "run", "ttft_ms", "cache%", "prefill", "decode", "out", "mtp/mixed", "client"))
    for r in d["results"]:
        e = r["engine"]
        print("%-4s %-4s %-8s %-7s %-10s %-10s %-8s %-22s %-6s" %
              (r["id"], r.get("run"), e.get("ttft_ms"), e.get("cache_pct"),
               e.get("prefill_tps"), e.get("decode_tps"), e.get("output_tokens"),
               ("%s|%s" % (e.get("mtp"), e.get("mixed")))[:22], r.get("client_s")))
    print()

print("===== 汇总（各臂 run1 口径）=====")
print("%-6s %-10s %-10s %-10s %-10s" % ("arm", "ttft_ms", "prefill", "decode", "out_tok"))
for arm, d in arms.items():
    for tag, subset in (("Q1/Q2", ("Q1", "Q2")),):
        rows = [r for r in d["results"] if r["id"] in subset and r.get("run") == 1]
        tt = [f(r["engine"].get("ttft_ms")) for r in rows if f(r["engine"].get("ttft_ms"))]
        pf = [f(r["engine"].get("prefill_tps")) for r in rows if f(r["engine"].get("prefill_tps"))]
        dc = [f(r["engine"].get("decode_tps")) for r in rows if f(r["engine"].get("decode_tps"))]
        ot = [f(r["engine"].get("output_tokens")) for r in rows if f(r["engine"].get("output_tokens"))]
        avg = lambda a: (sum(a) / len(a)) if a else None
        print("%-6s %-10s %-10s %-10s %-10s" % (arm + "/" + tag, avg(tt), avg(pf), avg(dc), avg(ot)))
