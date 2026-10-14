#!/usr/bin/env bash
# diag.sh —— 定位 CUDA kernel 与标量 oracle 差异的来源
#  (1) 迭代数扫描（iters=0 时 Sinkhorn 不迭代 => 只测布局/打包/量化）
#  (2) 三种编译 flag（默认 / -fmad=false / -use_fast_math=上游真 flag）
set -e
cd "$(dirname "$0")"
mkdir -p out data

echo "=== 编译 3 个 flag 变体 ==="
nvcc -O2 -std=c++17 -w -arch=sm_86                       -o gate_def   kvarn_quant_gate.cu
nvcc -O2 -std=c++17 -w -arch=sm_86 -fmad=false           -o gate_nofma kvarn_quant_gate.cu
nvcc -O2 -std=c++17 -w -arch=sm_86 -use_fast_math        -o gate_ff    kvarn_quant_gate.cu
echo "=== 生成 iters 扫描 dump（K4V4）==="
for it in 0 1 2 4 8 16; do
  ./dump dumprec data/g_it$it.bin  --gauss --bits-k 4 --bits-v 4 --iters $it > /dev/null
  ./dump dumprec data/o_it$it.bin  --outlier --bits-k 4 --bits-v 4 --iters $it > /dev/null
  ./dump dumprec data/r_it$it.bin  --tiles tiles/real_g128.bin --bits-k 4 --bits-v 4 --iters $it > /dev/null
done
L="data/g_it0.bin data/g_it1.bin data/g_it2.bin data/g_it4.bin data/g_it8.bin data/g_it16.bin"
L="$L data/o_it0.bin data/o_it1.bin data/o_it2.bin data/o_it4.bin data/o_it8.bin data/o_it16.bin"
L="$L data/r_it0.bin data/r_it1.bin data/r_it2.bin data/r_it4.bin data/r_it8.bin data/r_it16.bin"
for tag in def nofma ff; do
  echo ""
  echo "######## flags=$tag ########"
  ./gate_$tag $L | grep -E '^\[file\]|^records=|^判定' | tee out/diag_$tag.txt
done
echo "=== 完成 ==="
