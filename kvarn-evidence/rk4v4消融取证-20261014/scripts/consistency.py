#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""consistency.py —— 统一口径逐字一致性核算（combined = reasoning+content，前 200 字逐字）。

用法:
  consistency.py within <json> [<json> ...]      # 同臂内 run-to-run（同 id 各 run 互比）
  consistency.py cross  <A.json> <B.json>        # 两臂对照（按 (id,run) 配对）
口径与判据（与父任务一致）：
  * 逐字一致率 = 前 200 字相同的 (id,run) 对数 / 总对数，阈值 80%。
  * 每个 id 另给 len(combined) 与「首个分歧位置」（同 id 以第 1 次运行为基准）。
"""
import json, sys

N = 200


def load(p):
    return json.load(open(p, encoding="utf-8"))


def firstdiff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else n


def keyof(r):
    return (r["id"], r.get("round"))


def runkey(r):
    return (r["id"], r.get("run"), r.get("round"))


def grouped(d):
    res = {}
    order = []
    for r in d["results"]:
        k = keyof(r)
        if k not in res:
            res[k] = []
            order.append(k)
        res[k].append(r)
    return res, order


def grouped_runs(d):
    res = {}
    order = []
    for r in d["results"]:
        k = runkey(r)
        if k not in res:
            res[k] = r
            order.append(k)
    return res, order


def within(p):
    d = load(p)
    res, order = grouped(d)
    print("== within %s (arm=%s, effort_env=%s) ==" % (p, d.get("arm"), d.get("effort_env")))
    print("%-6s %-5s %-9s %-9s %-11s %-11s" % ("id", "run", "len(cmb)", "prefix200", "firstdiff", "len(reason/content)"))
    n_pairs = n_eq = 0
    for k in order:
        rs = res[k]
        base = rs[0]
        for r in rs:
            fd = firstdiff(base["combined"], r["combined"])
            pe = base["combined"][:N] == r["combined"][:N]
            print("%-6s %-5s %-9d %-9s %-11s %d/%d" %
                  ("%s%s" % (k[0], "" if k[1] is None else "#%s" % k[1]), r["run"], len(r["combined"]),
                   pe, ("same" if fd < 0 else "@%d" % fd), len(r["reasoning"]), len(r["content"])))
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                n_pairs += 1
                if rs[i]["combined"][:N] == rs[j]["combined"][:N]:
                    n_eq += 1
    print("同臂 run-to-run 前200字一致对 = %d/%d = %.1f%%" %
          (n_eq, n_pairs, 100.0 * n_eq / n_pairs if n_pairs else 0.0))
    return res, order


def cross(pa, pb):
    A, B = load(pa), load(pb)
    ares, aorder = grouped_runs(A)
    bres, _ = grouped_runs(B)
    print("== cross A=%s(arm=%s effort=%s)  B=%s(arm=%s effort=%s) ==" %
          (pa, A.get("arm"), A.get("effort_env"), pb, B.get("arm"), B.get("effort_env")))
    print("%-6s %-5s %-4s %-9s %-9s %-9s %-11s %-9s" %
          ("id", "run", "rnd", "lenA", "lenB", "prefix200", "firstdiff", "A/B out"))
    n_tot = n_pre = 0
    per_id = {}
    for k in aorder:
        ra = ares[k]
        rb = bres.get(k)
        if rb is None:
            print("%-6s %-5s %-4s MISSING in B" % (k[0], k[1], k[2])); n_tot += 1; continue
        pe = ra["combined"][:N] == rb["combined"][:N]
        fd = firstdiff(ra["combined"], rb["combined"])
        n_tot += 1; n_pre += pe
        per_id.setdefault(k[0], []).append((k[1], k[2], len(ra["combined"]), len(rb["combined"]), pe, fd))
        print("%-6s %-5s %-4s %-9d %-9d %-9s %-11s %s/%s" %
              (k[0], k[1], k[2], len(ra["combined"]), len(rb["combined"]), pe, ("same" if fd < 0 else "@%d" % fd),
               ra["engine"].get("output_tokens"), rb["engine"].get("output_tokens")))
    print()
    print("(id,run,round) 条目数 = %d" % n_tot)
    print("前200字逐字一致 = %d/%d = %.1f%%  (阈值 80%%)" % (n_pre, n_tot, 100.0 * n_pre / n_tot if n_tot else 0.0))
    print("### 判定：%s" % ("数值一致" if n_tot and n_pre >= 0.8 * n_tot else "有分歧"))
    print()
    print("== 每个 id 的对齐明细（run, round, lenA, lenB, prefix200eq, firstdiff）==")
    for i in per_id:
        print("%-4s %s" % (i, per_id[i]))
    return per_id


def main():
    mode = sys.argv[1]
    if mode == "within":
        for p in sys.argv[2:]:
            within(p); print()
    elif mode == "cross":
        cross(sys.argv[2], sys.argv[3])
    else:
        print(__doc__); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
