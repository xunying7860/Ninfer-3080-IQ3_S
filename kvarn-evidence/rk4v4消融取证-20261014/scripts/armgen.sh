#!/bin/bash
# armgen.sh —— 通用启动骨架：与 armA.sh 逐字同 flag，只从环境取 BIN/PORT/PROFILE/EXTRA。
# 环境（由 armctl3.sh 从 run/<NAME>.env 导出）：
#   ARM_BIN  引擎二进制   ARM_PORT 端口   ARM_PROFILE profile 文件
#   ARM_EXTRA 追加 flag（可为空，按空格切分）
set -u
ROOT=/data/kvarn/g128-rk4v4-test
LOADER=/data/ninfer/bundle/libs/lib64/ld-linux-x86-64.so.2
LIBPATH=/data/ninfer/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu
MODEL=/data/nvme/ninfer-models/Qwen3.8-27B-Coder390-EfficientThink-Q2LynnStyle-MTP.ninfer
export CUDA_VISIBLE_DEVICES=0
exec "$LOADER" --library-path "$LIBPATH" "$ARM_BIN" "$MODEL" \
  --host 127.0.0.1 --port "$ARM_PORT" --model-id qwen3.8-27b-coder390-q2lynn \
  --max-context 65536 --kv-capacity 65536 --kv-dtype rk4v4 \
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
