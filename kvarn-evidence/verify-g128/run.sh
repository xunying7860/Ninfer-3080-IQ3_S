#!/usr/bin/env bash
# run.sh —— KVarN G128 引擎级对拍工具 · 一条命令跑完
# 用法（Windows 宿主 / git-bash）：
#   docker run --rm -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/verify-g128:/w" -w /w \
#     nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run.sh"
# 说明：纯 host 计算，不需要 GPU、不需要 nvcc、不需要 python（只用 g++）。
#       独立容器/卷：不碰 kvarn-build / kvarnbuild，不碰别人的 gate/oracle 目录。
set -e
cd "$(dirname "$0")"
mkdir -p out data

echo "===== 0. 工具链 =====" | tee out/build.txt
g++ --version | head -1 | tee -a out/build.txt

echo "===== 1. 编译 verifier =====" | tee -a out/build.txt
g++ -O2 -std=c++17 -w -o verify kvarn_g128_verify.cpp 2>&1 | tee -a out/build.txt
echo "BUILD_OK" | tee -a out/build.txt

echo "===== 2. 生成 GATE2 dumprec（用只读 import 的 oracle，产出真实 tile 的 record）====="
mkdir -p data
g++ -O2 -std=c++17 -w -o dump import/dump_oracle.cpp 2>&1 | tee -a out/build.txt
# 真实 tile（layer35/63 k/v_proj，已按 beellama D256 口径旋转后切片）
./dump dumprec data/real_k4v4.bin --tiles data/real_g128.bin --bits-k 4 --bits-v 4 --iters 16 | tee out/dumps.txt
./dump dumprec data/real_k4v2.bin --tiles data/real_g128.bin --bits-k 4 --bits-v 2 --iters 16 | tee -a out/dumps.txt

echo "===== 3. selftest：import 的量化器副本 vs GATE2 dump ====="
# iters=0 与 iters=16 都应逐字节 0 差异（副本就是 oracle 的同一套代码）。
./dump dumprec data/real_k4v4_it0.bin --tiles data/real_g128.bin --bits-k 4 --bits-v 4 --iters 0 > /dev/null
./verify --selftest data/real_k4v4_it0.bin | tee out/selftest_it0.txt
./verify --selftest data/real_k4v4.bin    | tee out/selftest_it16.txt

echo "===== 4. 真实 tile 臂：--oracle（K4V4 / K4V2）====="
./verify --oracle data/real_k4v4.bin | tee out/oracle_real_k4v4.txt | tail -12
./verify --oracle data/real_k4v2.bin | tee out/oracle_real_k4v2.txt | tail -12

echo "===== 5. 合成全矩阵臂：--sweep（K4V4/K4V2 × 128/127/51/1 token × gauss/outlier）====="
./verify --sweep --verbose | tee out/sweep.txt

echo "===== 6. dump 入口自证 ====="
echo "--- 6a. 逐字段手写最小 dump + 引擎解码文件 -> --from-dump --decoded ---"
./verify --gen-fake data/fake_min.dump | tee out/fake_min.txt
echo "### 6a-1 正确引擎解码（应 PASS）" | tee -a out/fake_min.txt
./verify --from-dump data/fake_min.dump --decoded data/fake_min.dump.dec --verbose | tee -a out/fake_min.txt
echo "### 6a-2 投毒引擎解码（K 的 token7 取反，应 FAIL 且指向解码端）" | tee -a out/fake_min.txt
if ./verify --from-dump data/fake_min.dump --decoded data/fake_min.dump.bad --verbose >> out/fake_min.txt 2>&1; then
  echo "[!!] 投毒样本没被判 FAIL —— 工具判据失效" | tee -a out/fake_min.txt
else
  echo "[ok] 投毒样本被判 FAIL（判据有效）" | tee -a out/fake_min.txt
fi

echo "--- 6b. 真 dump：sweep 产物落成 KVD1 -> --from-dump ---"
./verify --gen-dump data/gen.dump | tee out/from_dump.txt
./verify --from-dump data/gen.dump | tee -a out/from_dump.txt

echo "--- 6c. 真 dump：GATE2 dumprec 转 KVD1 -> --from-dump（真实 tile）---"
./verify --make-dump data/real_k4v4.bin data/real_k4v4.dump | tee -a out/from_dump.txt
./verify --from-dump data/real_k4v4.dump | tee -a out/from_dump.txt

echo "===== 7. 汇总 ====="
{ echo "### selftest"; cat out/selftest_it0.txt out/selftest_it16.txt
  echo "### oracle 真实 tile"; grep -h "min cos_ENG_ADQ\|总判定\|病灶" out/oracle_real_k4v4.txt out/oracle_real_k4v2.txt
  echo "### from-dump 真实性"; grep -h "min cos_ENG_ADQ\|总判定\|病灶" out/from_dump.txt
} | tee out/summary.txt
echo "===== 完成 ====="
