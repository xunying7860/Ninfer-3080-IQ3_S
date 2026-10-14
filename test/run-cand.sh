#!/usr/bin/env bash
# 候选件启动器（任务 B 门禁测试用，临时，不入 systemd、不动生产 run.sh）
# 与生产 run.sh 逐字同参数，只改三处：
#   1) 二进制 → bundle/bin/ninfer-serve.cand（60f18c3 + 已取证补丁 #1~#6，md5 21bd4486…）
#   2) 加磁盘层 --disk-kv-path/--disk-kv-gib/--disk-kv-restore（生产是关的）
#   3) 日志 → logs/cand-serve.log（stderr 一并进去，才能看到 [diag] 落盘行）
set -u
ROOT=/data/ninfer
LOADER=$ROOT/bundle/libs/lib64/ld-linux-x86-64.so.2
LIBPATH=$ROOT/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$ROOT/logs" "$ROOT/test" /data/nvme/ninfer-disktier-test
echo "[launch] $(date '+%F %T') pid=$$" >> "$ROOT/logs/cand-serve.log"
exec "$LOADER" --library-path "$LIBPATH" "$ROOT/bundle/bin/ninfer-serve.cand" \
  "/data/nvme/ninfer-models/Qwen3.8-27B-Coder390-EfficientThink-Q2LynnStyle-MTP.ninfer" \
  --host 0.0.0.0 --port 18082 --model-id qwen3.8-27b-coder390-q2lynn \
  --max-context 317440 --kv-capacity 317440 --kv-dtype rk4v4 \
  --cuda-graph-allowance-mib 256 \
  --vision --vision-residency overlay --vision-max-merged 4096 --media-cache-mib 512 --media-live-mib 1024 \
  --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 \
  --device-state-slots 1 --host-state-slots 2 --host-kv-mib 512 --context-cache-policy rolling \
  --gdn-state-fp16 --spec mtp --draft-tokens 3 --adaptive-mtp \
  --chat-template "$ROOT/chat_template.jinja" --default-reasoning-effort max --default-max-tokens 131072 \
  --disk-kv-path /data/nvme/ninfer-disktier-test --disk-kv-gib 32 --disk-kv-restore
