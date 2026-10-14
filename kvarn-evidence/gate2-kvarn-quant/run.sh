#!/usr/bin/env bash
# run.sh —— GATE2 全量复现：KVarN G128 CUDA 量化/存储 kernel vs 标量 oracle record 逐字节对拍
# 用法（Windows 宿主 / git-bash）：
#   docker run --rm --gpus all -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/gate2-kvarn-quant:/w" -w /w \
#     nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run.sh"
# 依赖：./dump（= bee_oracle.cpp 逐字复制 + 追加 dumprec 子命令）与 tiles/real_g128.bin（oracle 产物，只读复制）
set -e
cd "$(dirname "$0")"
mkdir -p out data

echo "===== 0. 工具链 =====" | tee out/build.txt
g++ --version | head -1 | tee -a out/build.txt
nvcc --version | tail -1 | tee -a out/build.txt

echo "===== 1. 编译 =====" | tee -a out/build.txt
g++ -O2 -std=c++17 -w -o dump dump_oracle.cpp                                      2>&1 | tee -a out/build.txt
nvcc -O2 -std=c++17 -w -arch=sm_86                -o gate_def   kvarn_quant_gate.cu 2>&1 | tee -a out/build.txt
nvcc -O2 -std=c++17 -w -arch=sm_86 -fmad=false    -o gate_nofma kvarn_quant_gate.cu 2>&1 | tee -a out/build.txt
nvcc -O2 -std=c++17 -w -arch=sm_86 -use_fast_math -o gate_ff    kvarn_quant_gate.cu 2>&1 | tee -a out/build.txt
nvcc -O2 -std=c++17 -w -arch=sm_86                -o storegate kvarn_store_gate.cu  2>&1 | tee -a out/build.txt
echo "BUILD_OK" | tee -a out/build.txt

echo "===== 2. 生成 oracle record dump ====="
: > out/dumps.txt
for ds in --gauss --outlier; do
  tag=${ds#--}
  for kv in "4 4" "4 2" "2 2" "3 3" "5 5" "6 6"; do
    set -- $kv
    ./dump dumprec "data/${tag}_k$1v$2.bin" $ds --bits-k $1 --bits-v $2 --iters 16 | tee -a out/dumps.txt
  done
done
for kv in "4 4" "4 2" "2 2" "3 3" "5 5" "6 6"; do
  set -- $kv
  ./dump dumprec "data/real_k$1v$2.bin" --tiles tiles/real_g128.bin --bits-k $1 --bits-v $2 --iters 16 | tee -a out/dumps.txt
done

FULL=""
for n in gauss outlier real; do
  for kv in 4v4 4v2 2v2 3v3 5v5 6v6; do FULL="$FULL data/${n}_k${kv}.bin"; done
done

echo "===== 3. 全量逐字节对拍（flags=def，精确数学）====="
./gate_def -v $FULL | tee out/gate_def_full.txt | tail -4
echo "===== 4. 全量逐字节对拍（flags=-use_fast_math = 上游 ggml-cuda 真 flag）====="
./gate_ff  $FULL | tee out/gate_ff_full.txt | tail -4

echo "===== 5. iters 扫描 + 1-ulp 敏感性探针 ====="
for it in 0 1 2 4 8 16; do
  ./dump dumprec data/g_it$it.bin  --gauss   --bits-k 4 --bits-v 4 --iters $it > /dev/null
  ./dump dumprec data/o_it$it.bin  --outlier --bits-k 4 --bits-v 4 --iters $it > /dev/null
  ./dump dumprec data/r_it$it.bin  --tiles tiles/real_g128.bin --bits-k 4 --bits-v 4 --iters $it > /dev/null
done
SWEEP="data/g_it0.bin data/g_it1.bin data/g_it2.bin data/g_it4.bin data/g_it8.bin data/g_it16.bin"
SWEEP="$SWEEP data/o_it0.bin data/o_it1.bin data/o_it2.bin data/o_it4.bin data/o_it8.bin data/o_it16.bin"
SWEEP="$SWEEP data/r_it0.bin data/r_it1.bin data/r_it2.bin data/r_it4.bin data/r_it8.bin data/r_it16.bin"
./gate_def -s $SWEEP | tee out/diag_def.txt | tail -3
./gate_nofma  $SWEEP | grep -E '^\[file\]|^records=' | tee out/diag_nofma.txt | tail -3
./gate_ff  -s $SWEEP | tee out/diag_ff.txt | tail -3

echo "===== 6. 存储路径（旋转 + F16 stage）====="
./storegate | tee out/store.txt | tail -2
echo "===== 完成 ====="
