#!/usr/bin/env python3
# extract_verbatim.py —— gate 1（WHT 等价性）逐字取证脚本
#
# 目的：把两侧被对拍的实现**按行号逐字**抠成独立头文件，杜绝手抄/记忆重写。
# 输入（只读）：
#   A) beellama  : ref-beellama/src/ggml/src/ggml-cuda/kvarn.cu      (commit 58a1629274d8d74ea5a6acf34f30fb5946c2422b)
#   B) ninfer    : kvarn-任务A-20261012/engine/ninfer-all-kvarn/src/ops/kv_cache/hadamard_d256.cuh
# 输出（本目录）：
#   bee_const_verbatim.h          KVAR_N_DIM 等常量（kvarn.cu:12-19）
#   bee_wht128_verbatim.h         kvarn_wht_128（kvarn.cu:456-472）
#   bee_cross_slices_verbatim.h   kvarn_wht_cross_slices（kvarn.cu:510-543）
#   bee_store_call_verbatim.h     调用点（kvarn.cu:1113-1131，头文件形式仅作取证展示）
#   ninfer_hadamard_d256_verbatim.h  hadamard_d256.cuh:1-67 全文
#
# 每个输出文件头部写死 <源文件>:<起-止行>，并打印 sha256 供报告引用。
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # kvarn-全档位移植-20261013

BEE = os.path.join(ROOT, "ref-beellama", "src", "ggml", "src", "ggml-cuda", "kvarn.cu")
NIN = os.path.join(os.path.dirname(ROOT), "kvarn-任务A-20261012", "engine", "ninfer-all-kvarn",
                   "src", "ops", "kv_cache", "hadamard_d256.cuh")

JOBS = [
    ("bee_const_verbatim.h", BEE, 12, 19, "beellama KVarN 常量（KVAR_N_DIM 等）"),
    ("bee_wht128_verbatim.h", BEE, 456, 472, "beellama kvarn_wht_128（128 点 Sylvester FWHT + 1/sqrt(128)）"),
    ("bee_cross_slices_verbatim.h", BEE, 510, 543, "beellama kvarn_wht_cross_slices（跨 slice 蝶形 + 1/sqrt(head_slices)）"),
    ("bee_store_call_verbatim.h", BEE, 1113, 1131, "beellama headwide store 的调用点（旋转 + 跨 slice）"),
    ("ninfer_hadamard_d256_verbatim.h", NIN, 1, 67, "ninfer normalized_hadamard_d256_inplace（整 D256 归一化 FWHT）"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("[sha256 源文件]")
    for p in (BEE, NIN):
        print("  %s  %s" % (sha256(p), p))
    for name, src, lo, hi, note in JOBS:
        with open(src, "r", encoding="utf-8") as f:
            lines = f.readlines()
        body = "".join(lines[lo - 1:hi])
        src_rel = os.path.relpath(src, os.path.dirname(ROOT)).replace("\\", "/")
        header = (
            "// ============================================================================\n"
            "// GATE 1 逐字取证：本文件不是手抄，由 extract_verbatim.py 按行号从原文抠出。\n"
            "// 来源：%s:%d-%d  （%s）\n"
            "// 源文件 sha256：%s\n"
            "// 内容：%s\n"
            "// ============================================================================\n"
            % (src_rel, lo, hi, note, sha256(src), note)
        )
        with open(os.path.join(HERE, name), "w", encoding="utf-8", newline="") as f:
            f.write(header)
            f.write(body)
        print("[out] %-34s <= %s:%d-%d  (%d 行)" % (name, os.path.basename(src), lo, hi, hi - lo + 1))
        # 往返校验：抠出的行必须与源文件区间逐字节相同
        with open(os.path.join(HERE, name), "r", encoding="utf-8", newline="") as f:
            got = f.read()[len(header):]
        assert got == body, "round-trip mismatch for %s" % name
    print("[ok] 全部往返校验通过（抠出行 == 源区间逐字节相同）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
