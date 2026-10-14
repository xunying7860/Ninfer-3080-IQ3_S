#!/bin/bash
# armgen.sh —— 验收用启动骨架：与 /data/kvarn/g128-rk4v4-test/armgen.sh 逐字同 flag，
# 唯一差别：--kv-dtype 从 ARM_KV 取（不是写死 rk4v4），窗口从 ARM_MAXCTX 取（默认 65536）。
set -u
ROOT=/data/kvarn/g128-验收-20261014
LOADER=/data/ninfer/bundle/libs/lib64/ld-linux-x86-64.so.2
LIBPATH=/data/ninfer/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu
MODEL=/data/nvme/ninfer-models/Qwen3.8-27B-Coder390-EfficientThink-Q2LynnStyle-MTP.ninfer
export CUDA_VISIBLE_DEVICES=0
CTX=${ARM_MAXCTX:-65536}
exec "$LOADER" --library-path "$LIBPATH" "$ARM_BIN" "$MODEL" \
  --host 127.0.0.1 --port "$ARM_PORT" --model-id qwen3.8-27b-coder390-q2lynn \
  --max-context "$CTX" --kv-capacity "$CTX" --kv-dtype "$ARM_KV" \
  --device-profile-path "$ARM_PROFILE" \
  --cuda-graph-allowance-mib "${ARM_CG_MIB:-1024}" \
  --vision --vision-residency overlay --vision-max-merged 4096 \
  --media-cache-mib 512 --media-live-mib 1024 \
  --max-concurrency 1 --max-pending-requests 4 --prefill-chunk 1024 \
  --device-state-slots 1 --host-state-slots 1 --host-kv-mib 256 --context-cache-policy rolling \
  --gdn-state-fp16 --spec mtp --draft-tokens 3 --adaptive-mtp \
  --chat-template /data/ninfer/chat_template.jinja \
  --default-reasoning-effort "${ARM_EFFORT:-high}" --default-max-tokens 64 \
  ${ARM_EXTRA:-} "$@"
