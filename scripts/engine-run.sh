#!/bin/bash
# 参数化启动器（调优用）：所有可调项走环境变量，避免同名参数重复出现。
# 固定项（用户定案 2026-10-08）：rk4v4 / 256K 上下文 / 温度1 / top_k20 / top_p0.95 /
#   min_p0 / 存在惩罚0 / 重复惩罚1（引擎只接受默认 1.0）/ 不开 KVMem。
# 可调项（本文件变量）：SPEC DRAFT ADAPTIVE CHUNK ALLOWANCE DSS HOSTKV HOSTSLOTS
#   PREFILL_CUBLAS FAST_PREFILL LOOKUP_MLP EXTRA VISION
set -u
ROOT=/data/ninfer-fusion-kvmem
MODEL=${MODEL:-/data/ninfer/models/gsq_rco_iq3_s_dflash2_prop.ninfer}
PORT=${PORT:-18084}
GPU=${GPU:-0}

SPEC=${SPEC:-mtp}
DRAFT=${DRAFT:-3}
ADAPTIVE=${ADAPTIVE:-1}
CHUNK=${CHUNK:-1024}
ALLOWANCE=${ALLOWANCE:-0}
DSS=${DSS:-8}
HOSTKV=${HOSTKV:-512}
HOSTSLOTS=${HOSTSLOTS:-2}
VISION=${VISION:-1}
EXTRA=${EXTRA:-}

LOADER=$ROOT/bundle/libs/lib64/ld-linux-x86-64.so.2
LIBPATH=$ROOT/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$ROOT/logs"

ARGS=(--host 0.0.0.0 --port "$PORT" --model-id qwen3.8-27b
      --max-context 262144 --kv-capacity 262144 --kv-dtype rk4v4
      --temperature 1 --top-k 20 --top-p 0.95 --min-p 0
      --presence-penalty 0 --frequency-penalty 0
      --max-concurrency 1 --max-pending-requests 16 --prefill-chunk "$CHUNK"
      --device-state-slots "$DSS" --host-state-slots "$HOSTSLOTS" --host-kv-mib "$HOSTKV"
      --context-cache-policy rolling --gdn-state-fp16
      --spec "$SPEC" --draft-tokens "$DRAFT"
      --chat-template /data/ninfer/chat_template.jinja
      --default-reasoning-effort max --default-max-tokens 131072
      --cors)
[ "$VISION" = "1" ] && ARGS+=(--vision --vision-residency overlay --vision-max-merged 12288)
[ "$ADAPTIVE" = "1" ] && ARGS+=(--adaptive-mtp)
[ "$ALLOWANCE" != "0" ] && ARGS+=(--cuda-graph-allowance-mib "$ALLOWANCE")
# EXTRA 用空格分隔的额外旗标
if [ -n "$EXTRA" ]; then for f in $EXTRA; do ARGS+=("$f"); done; fi

echo "ENV: SPEC=$SPEC DRAFT=$DRAFT ADAPTIVE=$ADAPTIVE CHUNK=$CHUNK ALLOWANCE=$ALLOWANCE DSS=$DSS HOSTKV=$HOSTKV VISION=$VISION EXTRA=$EXTRA"
exec "$LOADER" --library-path "$LIBPATH" "$ROOT/bundle/bin/ninfer-serve" "$MODEL" "${ARGS[@]}"
