#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_archive.py —— KVarN 路径保真检查：把本工具 `--slice` 打出的 128-dim 片口径数字，
与旧 oracle（`../../oracle/out/real_g128_k4v4.txt`）的 `_s0_` 行逐字段比对。
两者应完全相同（同一份 codec、同一份 stage_f16、同一份参照口径）。

用法：
    ./llama_oracle quant --mode kvarn --tiles data/real_g128.bin --bits-k 4 --bits-v 4 \
        --group 128 --iters 16 --stage-f16 --ref-h16 --slice --tag real_g128 \
        > out/archive_check_slice.txt
    python check_archive.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MINE = os.path.join(HERE, "out", "archive_check_slice.txt")
ARCH = os.path.join(HERE, "..", "..", "oracle", "out", "real_g128_k4v4.txt")


def parse_mine(p):
    out = {}
    for ln in open(p, encoding="utf-8", errors="replace"):
        m = re.match(r"SLICE \S+\s+(\S+)\s+bits=4 (\d+x\d+) tok=(col|row) \| (.*)", ln.strip())
        if m:
            out[m.group(1)] = m.group(4).strip()     # 键 = pair 名（已去掉 _sN）
    return out


def parse_arch(p):
    out = {}
    for ln in open(p, encoding="utf-8", errors="replace"):
        m = re.match(r"(\S+)\s+bits=4 (\d+x\d+) tok=(col|row) rec=\s*\d+ B \| (.*)", ln.strip())
        if m and m.group(1).endswith("_s0_g0"):      # 本工具的 SLICE 行 = 每个 pair 的首片
            out[re.sub(r"_s\d+", "", m.group(1))] = m.group(4).strip()
    return out


def main():
    if not os.path.exists(MINE):
        print("缺 %s（先按 docstring 跑一次 --slice）" % MINE); return 2
    if not os.path.exists(ARCH):
        print("缺归档文件 %s" % ARCH); return 2
    a, b = parse_mine(MINE), parse_arch(ARCH)
    common = sorted(set(a) & set(b))
    diff = [k for k in common if a[k] != b[k]]
    print("我的 SLICE 行 = %d，归档 _s0_ 行 = %d，同名可比 = %d" % (len(a), len(b), len(common)))
    for k in diff:
        print("  差异 %-16s\n    我: %s\n    档: %s" % (k, a[k], b[k]))
    ok = (len(common) == 16 and not diff)
    print("逐字段完全相同 = %s（不同 %d 行）%s" % ("是" if not diff else "否", len(diff), "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
