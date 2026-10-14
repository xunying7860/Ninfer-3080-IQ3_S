#!/usr/bin/env bash
# GATE 1 复现脚本（在 docker 里跑；本机 RTX 3080 / sm_86）
# docker run --rm --gpus all -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/wht-gate:/w" -w /w \
#   nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run.sh"
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p out

{
  echo "== 工具链 =="
  g++ --version | head -1
  nvcc --version | tail -1
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
} 2>&1 | tee out/toolchain.txt

echo "== 逐字取证（两侧实现按行号抠出，往返校验）=="
if command -v python3 >/dev/null 2>&1; then
  python3 extract_verbatim.py | tee out/extract.txt
else
  echo "(容器内无 python3；使用宿主已生成的 *_verbatim.h)" | tee out/extract.txt
  grep -h '来源' *_verbatim.h | sed 's/^/  /' | tee -a out/extract.txt
fi

echo "== 编译 =="
nvcc -O2 -std=c++17 -arch=sm_86 -lineinfo -o gate gate.cu 2>&1 | tee out/build.txt
echo "编译完成 rc=$?"

echo "== 运行 =="
./gate 2>&1 | tee out/gate.txt
rc=${PIPESTATUS[0]}
echo "[exit code] $rc   (0 = 逐字判据 PASS；1 = FAIL，详见文章判定段)"

echo "== 三方对拍（oracle rotate_d256 vs A vs B）=="
if python3 -c "import torch" >/dev/null 2>&1; then
  python3 check_oracle.py | tee out/oracle_check.txt
else
  echo "(容器内无 torch -> 该步必须在宿主跑：cd wht-gate && python check_oracle.py | tee out/oracle_check.txt)" | tee out/oracle_check.txt
fi
exit 0
