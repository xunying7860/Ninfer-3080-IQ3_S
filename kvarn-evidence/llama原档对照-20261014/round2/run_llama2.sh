#!/usr/bin/env bash
# run_llama2.sh —— round2：把 llama.cpp KV 允许集合内的真实对手补进来重跑
# 相对 oracle/run_llama.sh 的差异：llama 档集合加入 iq4_nl / q3_0 / q3_1（集合内真实档），
# 并把 llama 与 llama-rot 的类型列表统一为 9 档。其余口径、模式、数据规格逐行相同。
# 用法（Windows 宿主 / git-bash，注意：不带 --gpus，不占 GPU）：
#   docker run --rm -v "<...>/llama原档对照-20261014/round2:/w" -w /w \
#     nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run_llama2.sh"
set -e
mkdir -p out

echo "===== 编译 ====="
g++ -O2 -std=c++17 -w -o llama_oracle llama_oracle.cpp

echo "===== 0. 自检 / 字节账 / 交叉校验样本 ====="
./llama_oracle selftest data/real_g128.bin | tee out/selftest.txt
./llama_oracle bytes | tee out/bytes.txt
./llama_oracle dump out/dump.txt | tee out/dump_log.txt

echo "===== 0b. 归档口径复刻（128-dim 片测，应与旧 oracle 的 real_g128_k4v4.txt 逐位相同）====="
./llama_oracle quant --mode kvarn --tiles data/real_g128.bin --bits-k 4 --bits-v 4 \
  --group 128 --iters 16 --stage-f16 --ref-h16 --slice --tag real_g128 \
  | tee out/archive_check_slice.txt | grep -c "^SLICE"

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

LTYPES="q4_0 q4_1 iq4_nl q5_0 q5_1 q8_0 q3_0 q3_1 q4_K"

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
  for t in $LTYPES; do
    run "$name" "$dargs" llama "out/${name}_llama_${t}.txt" --lk $t --lv $t
  done

  # --- 6. llama.cpp 原生档 + 先旋转 ---
  for t in $LTYPES; do
    run "$name" "$dargs" llama-rot "out/${name}_llamarot_${t}.txt" --lk $t --lv $t
  done
done

echo "===== 全部完成 ====="
