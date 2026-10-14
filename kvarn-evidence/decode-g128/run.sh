#!/usr/bin/env bash
# decode-g128 一键复现（本机 docker，RTX 3080 / sm_86；不用 kvarn-build/kvarnbuild 容器或卷）
#
#   docker run --rm --gpus all \
#     -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/decode-g128:/w" \
#     -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/oracle:/oracle:ro" \
#     -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/offsets-g128:/offsets:ro" \
#     -w /w nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run.sh"
#
# 可选：加 -v ".../kvarn-任务A-20261012/engine/ninfer-all-kvarn:/eng:ro" 会多跑一步
#       「内嵌逆旋转副本 vs 引擎真头」的逐位对拍（hadamard_crosscheck.cu）。
#
# 约定：/oracle、/offsets、/eng 只读挂载（真实 tile、槽位表 CSV、引擎真头）；本目录只写 out/。
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out

echo "[build] nvcc -O2 -std=c++17 -arch=sm_86 -fmad=false -Xcompiler -ffp-contract=off"
nvcc -O2 -std=c++17 -arch=sm_86 -fmad=false -lineinfo \
     -Xcompiler -ffp-contract=off \
     -o harness decode_harness.cu

ORACLE_BIN="${ORACLE_BIN:-/oracle/data/real_g128.bin}"
SLOT_CSV="${SLOT_CSV:-/offsets/out/p1_slot_map_g128.csv}"

echo "[run] ./harness $ORACLE_BIN $SLOT_CSV out"
./harness "$ORACLE_BIN" "$SLOT_CSV" out 2>&1 | tee out/run.txt

# 同一输入再跑一次，确认结果确定（除 device 名/时间外逐行相同）
./harness "$ORACLE_BIN" "$SLOT_CSV" out > out/run2.txt 2>&1
if diff -q out/run.txt out/run2.txt >/dev/null; then
  echo "[repro] 两次运行输出逐行相同：PASS"
else
  echo "[repro] 两次运行输出有差异（见 out/run.txt vs out/run2.txt）"
fi

# 可选：内嵌逆旋转副本 vs 引擎真头（只读 include）
if [ -d /eng/src ]; then
  echo "[build] hadamard_crosscheck（引擎真头 /eng/src 只读 include）"
  nvcc -O2 -std=c++17 -arch=sm_86 -I/eng/src -o hcheck hadamard_crosscheck.cu
  ./hcheck | tee out/hadamard_crosscheck.txt
  sha256sum /eng/src/ops/kv_cache/hadamard_d256.cuh | tee -a out/hadamard_crosscheck.txt
else
  echo "[skip] 未挂载 /eng，跳过 hadamard crosscheck"
fi

echo "[done] 原始输出：out/run.txt（含 out/summary.txt / out/per_token.txt）"
