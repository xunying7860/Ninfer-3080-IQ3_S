#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_oracle.py —— 三方对拍：oracle 的 rotate_d256（= 阶段 1 参照实现，也是 gate2 输入 tile 的旋法）
                             vs  路径 A（kit 里逐字抄的 beellama kvarn_wht_128 + cross_slices）
                             vs  路径 B（kit 里逐字抄的 ninfer normalized_hadamard_d256_inplace）

输入：out/ab_probe.bin（gate.cu 落盘的 8 个向量 ×（输入, A, B））
参照实现**不重抄**：直接 import oracle/make_tiles.py 里的 rotate_d256（只读引用，不写 oracle/）。
运行：python check_oracle.py            （宿主 python + torch；容器里没有 torch）
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(os.path.dirname(HERE), "oracle")
sys.path.insert(0, ORACLE)

import torch                      # noqa: E402
import make_tiles                 # noqa: E402  oracle 的 D256 口径实现（make_tiles.py:53-63）

LIMIT = 1e-6
R = 256


def bits(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def main():
    path = os.path.join(HERE, "out", "ab_probe.bin")
    with open(path, "rb") as fh:
        buf = fh.read()
    n, = struct.unpack_from("<i", buf, 0)
    off = 4
    print("[orch ] torch %s ; 参照实现 = %s（make_tiles.rotate_d256，只读 import）" % (torch.__version__, ORACLE))
    print("[orch ] 探测向量数 = %d，每个 %d 维" % (n, R))
    worstOA = worstOB = 0.0
    bitOA = bitOB = 0
    tot = 0
    for t in range(n):
        inp = struct.unpack_from("<%df" % R, buf, off); off += 4 * R
        A = struct.unpack_from("<%df" % R, buf, off); off += 4 * R
        B = struct.unpack_from("<%df" % R, buf, off); off += 4 * R
        x = torch.tensor(inp, dtype=torch.float32).reshape(1, 1, R)
        o = make_tiles.rotate_d256(x).reshape(-1).tolist()
        mOA = max(abs(o[i] - A[i]) for i in range(R))
        mOB = max(abs(o[i] - B[i]) for i in range(R))
        worstOA = max(worstOA, mOA); worstOB = max(worstOB, mOB)
        bitOA += sum(1 for i in range(R) if bits(o[i]) != bits(A[i]))
        bitOB += sum(1 for i in range(R) if bits(o[i]) != bits(B[i]))
        tot += R
        print("        vec%d: max|oracle-A| = %.6e   max|oracle-B| = %.6e" % (t, mOA, mOB))
    print("[orch ] 汇总（%d 元素）：max|oracle-A| = %.6e ；max|oracle-B| = %.6e" % (tot, worstOA, worstOB))
    print("[orch ] bit 不同的元素数：oracle vs A = %d / %d ；oracle vs B = %d / %d" % (bitOA, tot, bitOB, tot))
    ok = (worstOA <= LIMIT)
    print("[orch ] 判定（≤1e-6）：oracle 与 beellama 逐字路径 A %s ；与 ninfer 逐字路径 B %s"
          % ("PASS" if worstOA <= LIMIT else "FAIL", "PASS" if worstOB <= LIMIT else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
