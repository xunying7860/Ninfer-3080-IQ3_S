#!/usr/bin/env bash
# run.sh —— 阶段 1 标量 oracle 的完整复现命令行（在 nvidia/cuda:13.1.2-devel-ubuntu24.04 里跑）
# 用法（Windows 宿主 / git-bash）：
#   docker run --rm -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/oracle:/w" -w /w \
#     nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run.sh"
# 先在宿主侧生成真实 tile（需要 torch）：
#   python make_tiles.py --group 128 ; python make_tiles.py --group 64
set -e
mkdir -p out
echo "===== 编译 ====="
g++ -O2 -std=c++17 -w -o bee bee_oracle.cpp
nvcc -O2 -std=c++17 -o f16check f16_check.cu

echo "===== 1. 36 组合字节账（doc02 / make_layout / record 三口径逐格） ====="
./bee table | tee out/table.txt

echo "===== 2. pack_bits / unpack_bits_value 自检（bits∈{2,3,4,5,6,8}） ====="
./bee packtest | tee out/packtest.txt

echo "===== 3. fp16 转换器 vs nvcc __float2half_rn 逐位对拍 ====="
./f16check 3000000 | tee out/f16check.txt

echo "===== 4. Sinkhorn：GPU 口径 vs CPU 口径 ====="
./bee sinktest data/real_g128.bin | tee out/sinktest.txt

echo "===== 5. 真实数据（beellama 旋转口径）误差 ====="
./bee quant --tiles data/real_g128.bin --bits-k 4 --bits-v 2 --group 128 --iters 16 | tee out/real_g128_k4v2.txt
./bee quant --tiles data/real_g128.bin --bits-k 4 --bits-v 4 --group 128 --iters 16 | tee out/real_g128_k4v4.txt
./bee quant --tiles data/real_g64.bin  --bits-k 4 --bits-v 2 --group 64  --iters 16 | tee out/real_g64_k4v2.txt
./bee quant --tiles data/real_g64.bin  --bits-k 4 --bits-v 4 --group 64  --iters 16 | tee out/real_g64_k4v4.txt

echo "===== 6. 合成数据（高斯 / 带离群通道）误差 ====="
for g in 128 64; do
  for bits in "4 2" "4 4" "2 2"; do
    set -- $bits
    ./bee quant --gauss   --group $g --bits-k $1 --bits-v $2 --iters 16 --seed 20261013 | tee out/synth_gauss_g${g}_k$1v$2.txt
    ./bee quant --outlier --group $g --bits-k $1 --bits-v $2 --iters 16 --seed 20261013 | tee out/synth_outlier_g${g}_k$1v$2.txt
  done
done

echo "===== 7. 迭代数敏感性（8 vs 16） ====="
./bee quant --tiles data/real_g128.bin --bits-k 4 --bits-v 4 --group 128 --iters 8 | tee out/real_g128_k4v4_iters8.txt

echo "===== 8. CPU 口径对照（确认两口径量化结果一致） ====="
./bee quant --tiles data/real_g128.bin --bits-k 4 --bits-v 4 --group 128 --iters 16 --cpu-order | tee out/real_g128_k4v4_cpuorder.txt
echo "全部完成"
