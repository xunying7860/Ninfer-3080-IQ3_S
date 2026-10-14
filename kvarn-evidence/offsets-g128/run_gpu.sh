#!/bin/bash
# run_gpu.sh —— 用引擎真头（只读 include）在 RTX 3080 上跑 offset 探针
#   docker run --rm --gpus all \
#     -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/offsets-g128:/w" \
#     -v "H:/Hermes工作区/Hermes/kvarn-任务A-20261012/engine/ninfer-all-kvarn/src:/eng:ro" \
#     -w /w nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run_gpu.sh"
# 说明：/eng 是**只读**挂载（:ro），只 include，不改引擎目录。
set -u
ENG=${ENG:-/eng}
echo "== nvcc =="
nvcc --version | tail -2
echo "include: $ENG/src/ops/kernel/paged_kv_address.cuh  (+ $ENG/include)"
ls -l "$ENG/src/ops/kernel/paged_kv_address.cuh" || exit 2
echo
nvcc -std=c++20 -O2 -arch=sm_86 -Xcompiler -w -I"$ENG/src" -I"$ENG/include" \
     -o probe_engine_offsets probe_engine_offsets.cu || exit 2
echo "build ok"
echo
./probe_engine_offsets
rc=$?
echo "run rc=$rc"
exit $rc
