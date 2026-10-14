#!/bin/bash
# smoke：确认 ninfer-perplexity 的 kvarn128 档可跑、输出格式、耗时与显存
set -u
ROOT=/data/kvarn/g128-验收-20261014
LOADER=/data/ninfer/bundle/libs/lib64/ld-linux-x86-64.so.2
LIBPATH=/data/ninfer/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu
MODEL=/data/nvme/ninfer-models/Qwen3.8-27B-Coder390-EfficientThink-Q2LynnStyle-MTP.ninfer
export CUDA_VISIBLE_DEVICES=0
TS=$(date +%Y%m%d-%H%M%S)
OUT=$ROOT/ppl/smoke-kvarn128-A-$TS
mkdir -p "$OUT"
echo "### smoke kvarn128 corpusA ctx8192 start $(date '+%F %T')"
echo "### before:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
t0=$(date +%s)
"$LOADER" --library-path "$LIBPATH" /data/kvarn/g128-engine/bin/ninfer-perplexity "$MODEL" \
  --text /data/kvarn/ppl_doc.txt --device 0 --context 8192 --disjoint --gdn-state-fp16 \
  --kv-dtype kvarn128 --output "$OUT" 2>&1
rc=$?
t1=$(date +%s)
echo "### exit=$rc elapsed=$((t1-t0))s end $(date '+%F %T')"
echo "### after:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "### files:"; ls -la "$OUT"
echo "### DONE $(date '+%F %T')"
