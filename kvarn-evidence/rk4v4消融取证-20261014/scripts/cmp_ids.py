#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cmp_ids.py <A.json> <B.json> [id1,id2,...]

指定 id 的逐字对照（口径：combined = reasoning+content，前 200 字逐字 + 全文逐字）。
默认只比 run=1；给 ALL 则比所有 run。输出：每 id 的 lenA/lenB、前200字是否相同、
首个分歧位置（-1=全文相同）、全文是否相同。
"""
import json, sys

N = 200
ALL = "ALL" in sys.argv


def load(p):
    return json.load(open(p, encoding="utf-8"))


def firstdiff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else n


def index(d):
    m = {}
    for r in d["results"]:
        m.setdefault(r["id"], []).append(r)
    return m


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    ids = [x for x in sys.argv[3].split(",")] if len(sys.argv) > 3 and sys.argv[3] not in ("", "ALL") else None
    if ids is None:
        ids = [x for x in sys.argv[3:] if x not in ("ALL",)]
    A, B = index(a), index(b)
    if not ids:
        ids = [k for k in A]
    print("A=%s(arm=%s effort=%s)  B=%s(arm=%s effort=%s)" %
          (sys.argv[1], a.get("arm"), a.get("effort_env"), sys.argv[2], b.get("arm"), b.get("effort_env")))
    print("%-5s %-4s %-8s %-8s %-10s %-10s %-10s" % ("id", "run", "lenA", "lenB", "prefix200", "firstdiff", "full_eq"))
    n_tot = n_pre = n_full = 0
    for i in ids:
        ra_list = A.get(i) or []
        rb_list = B.get(i) or []
        if not ra_list or not rb_list:
            print("%-5s MISSING (A=%d B=%d)" % (i, len(ra_list), len(rb_list))); continue
        pairs = list(zip(ra_list, rb_list)) if ALL else [(ra_list[0], rb_list[0])]
        for ra, rb in pairs:
            pe = ra["combined"][:N] == rb["combined"][:N]
            fd = firstdiff(ra["combined"], rb["combined"])
            fe = ra["combined"] == rb["combined"]
            n_tot += 1; n_pre += pe; n_full += fe
            print("%-5s %-4s %-8d %-8d %-10s %-10s %-10s" %
                  (i, ra.get("run"), len(ra["combined"]), len(rb["combined"]), pe,
                   ("same" if fd < 0 else "@%d" % fd), fe))
    print("前200字逐字一致 = %d/%d = %.1f%%   全文逐字一致 = %d/%d = %.1f%%" %
          (n_pre, n_tot, 100.0 * n_pre / n_tot if n_tot else 0, n_full, n_tot, 100.0 * n_full / n_tot if n_tot else 0))


if __name__ == "__main__":
    main()
