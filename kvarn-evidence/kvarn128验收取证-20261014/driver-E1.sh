#!/bin/bash
# driver-E1.sh —— 实验1 引擎侧 PPL 四臂 × 两语料
# 口径：--device 0 --context 8192 --disjoint --gdn-state-fp16，仅 --kv-dtype 与语料变。
# 一次只跑一个 GPU 进程（串行）。
set -u
ROOT=/data/kvarn/g128-验收-20261014
LOADER=/data/ninfer/bundle/libs/lib64/ld-linux-x86-64.so.2
LIBPATH=/data/ninfer/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu
MODEL=/data/nvme/ninfer-models/Qwen3.8-27B-Coder390-EfficientThink-Q2LynnStyle-MTP.ninfer
export CUDA_VISIBLE_DEVICES=0
CTX=${CTX:-8192}
ARMS=${ARMS:-"kvarn128 rk4v4 int8 bf16"}
# 语料：A=ppl_doc.txt  B=ppl_doc2.txt
declare -A CORP=( [A]=/data/kvarn/ppl_doc.txt [B]=/data/kvarn/ppl_doc2.txt )
mkdir -p "$ROOT/ppl" "$ROOT/logs"
SAMPLES=$ROOT/logs/e1-mem-samples.txt
: > "$SAMPLES"
( while :; do nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null >> "$SAMPLES"; sleep 1; done ) &
SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT

for C in A B; do
  CF=${CORP[$C]}
  for ARM in $ARMS; do
    TS=$(date +%Y%m%d-%H%M%S)
    OUT=$ROOT/ppl/${ARM}-${C}-ctx${CTX}-${TS}
    mkdir -p "$OUT"
    LOG=$ROOT/logs/e1-${ARM}-${C}-${TS}.log
    echo "########## E1 arm=$ARM corpus=$C ctx=$CTX start $(date '+%F %T') ##########"
    echo "### cmd: ninfer-perplexity $MODEL --text $CF --device 0 --context $CTX --disjoint --gdn-state-fp16 --kv-dtype $ARM --output $OUT"
    t0=$(date +%s)
    "$LOADER" --library-path "$LIBPATH" /data/kvarn/g128-engine/bin/ninfer-perplexity "$MODEL" \
      --text "$CF" --device 0 --context "$CTX" --disjoint --gdn-state-fp16 \
      --kv-dtype "$ARM" --output "$OUT" > "$LOG" 2>&1
    rc=$?
    t1=$(date +%s)
    echo "### rc=$rc elapsed=$((t1-t0))s log=$LOG"
    grep -a -E "engine ready|corpus ready|scoring complete|Perplexity result|kv:|domain|custom|overall|score rate|report:|error|Error|FATAL|out of memory|shorter than" "$LOG" | sed 's/^/    /'
    echo
  done
done
echo "########## E1 ALL DONE $(date '+%F %T') ##########"
