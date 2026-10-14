#!/usr/bin/env bash
# run_llama.sh —— llama.cpp 原生 KV 档 vs KVarN 的全档位对拍（一条命令跑完）
# 用法（Windows 宿主 / git-bash）：
#   docker run --rm -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/llama原档对照-20261014/oracle:/w" \
#     -w /w nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run_llama.sh"
# 注意：不带 --gpus，不占 GPU。
set -e
mkdir -p out

echo "===== 编译 ====="
g++ -O2 -std=c++17 -w -o llama_oracle llama_oracle.cpp

echo "===== 0. 自检 / 字节账 / 交叉校验样本 ====="
./llama_oracle selftest data/real_g128.bin | tee out/selftest.txt
./llama_oracle bytes | tee out/bytes.txt
./llama_oracle dump out/dump.txt | tee out/dump_log.txt

echo "===== 0b. 归档口径复刻（128-dim 片测，应与 out/real_g128_k4v4.txt 逐位相同）====="
./llama_oracle quant --mode kvarn --tiles data/real_g128.bin --bits-k 4 --bits-v 4 \
  --group 128 --iters 16 --stage-f16 --ref-h16 --slice --tag real_g128 \
  | tee out/archive_check_slice.txt | grep -c "^SLICE"
echo "（把上面 SLICE 行与 ../../oracle/out/real_g128_k4v4.txt 比：cos_glob/cos_vec_mean/cos_vec_min/relL2/maxAbs 应完全一致）"

# 数据规格：name|data-args
DATA_SPECS=(
  "real_g128|--tiles data/real_g128.bin --group 128"
  "real_g64|--tiles data/real_g64.bin --group 64"
  "synth_gauss_g128|--gauss --group 128"
  "synth_gauss_g64|--gauss --group 64"
  "synth_outlier_g128|--outlier --group 128"
  "synth_outlier_g64|--outlier --group 64"
)

run() {   # run <data> <dataargs> <mode> <outfile> <extra...>
  local name="$1"; local dargs="$2"; local mode="$3"; local outf="$4"; shift 4
  echo "----- $name / $mode $* -----"
  ./llama_oracle quant --mode "$mode" $dargs --tag "$name" "$@" | tee "$outf"
}

for spec in "${DATA_SPECS[@]}"; do
  name="${spec%%|*}"; dargs="${spec#*|}"
  echo "############################## $name ##############################"

  # --- 1. KVarN 档（iters=16，beellama 生产口径）---
  for kv in "4 4" "4 3" "4 2" "3 3" "5 5" "6 6" "8 8"; do
    set -- $kv
    run "$name" "$dargs" kvarn "out/${name}_kvarn_k$1v$2_i16.txt" --bits-k $1 --bits-v $2 --iters 16
  done

  # --- 2. 消融：只旋转不平衡（Hadamard + 逐通道 uniform，iters=0）---
  for kv in "4 4" "4 3" "4 2" "3 3" "5 5" "6 6" "8 8"; do
    set -- $kv
    run "$name" "$dargs" rot-only "out/${name}_rotonly_k$1v$2.txt" --bits-k $1 --bits-v $2
  done

  # --- 3. 消融：只平衡不旋转 ---
  run "$name" "$dargs" bal-only "out/${name}_balonly_k4v4.txt" --bits-k 4 --bits-v 4 --iters 16

  # --- 4. 消融：都不（纯 RTN 逐通道 affine）---
  run "$name" "$dargs" neither "out/${name}_neither_k4v4.txt" --bits-k 4 --bits-v 4

  # --- 5. llama.cpp 原生档（不旋转）---
  for t in q4_0 q4_1 q5_0 q5_1 q8_0 q4_K; do
    run "$name" "$dargs" llama "out/${name}_llama_${t}.txt" --lk $t --lv $t
  done

  # --- 6. llama.cpp 原生档 + 先旋转 ---
  for t in q4_0 q4_1 q5_0 q5_1 q8_0 q4_K; do
    run "$name" "$dargs" llama-rot "out/${name}_llamarot_${t}.txt" --lk $t --lv $t
  done
done

echo "===== 全部完成 ====="
