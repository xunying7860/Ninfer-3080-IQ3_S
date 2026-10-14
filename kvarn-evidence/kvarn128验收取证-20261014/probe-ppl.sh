#!/bin/bash
# probe：测 bf16 KV 在给定 context 下能否起来（显存最吃紧的臂），并顺带拿 corpus B 的 token 数
set -u
ROOT=/data/kvarn/g128-验收-20261014
LOADER=/data/ninfer/bundle/libs/lib64/ld-linux-x86-64.so.2
LIBPATH=/data/ninfer/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu
MODEL=/data/nvme/ninfer-models/Qwen3.8-27B-Coder390-EfficientThink-Q2LynnStyle-MTP.ninfer
export CUDA_VISIBLE_DEVICES=0
CTX=${CTX:-32768}
CORPUS=${CORPUS:-/data/kvarn/ppl_doc.txt}
ARM=${ARM:-bf16}
TS=$(date +%Y%m%d-%H%M%S)
OUT=$ROOT/ppl/probe-$ARM-ctx$CTX-$TS
mkdir -p "$OUT"
echo "### probe arm=$ARM ctx=$CTX corpus=$CORPUS start $(date '+%F %T')"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
t0=$(date +%s)
"$LOADER" --library-path "$LIBPATH" /data/kvarn/g128-engine/bin/ninfer-perplexity "$MODEL" \
  --text "$CORPUS" --device 0 --context "$CTX" --disjoint --gdn-state-fp16 \
  --kv-dtype "$ARM" --output "$OUT" 2>&1
rc=$?
t1=$(date +%s)
echo "### exit=$rc elapsed=$((t1-t0))s end $(date '+%F %T')"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "### DONE $(date '+%F %T')"
