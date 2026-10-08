#!/bin/bash
# ============================================================================
# ninfer-fusion-kvmem（下游衍生引擎 · KVMem 环能力）· x99 / GPU0 启动脚本
# 位置：/data/ninfer-fusion-kvmem/run.sh
# 盘位：/data（= ubuntu--vg-ubuntu--lv ← /dev/sda，SATA 盘）
# 端口：18084（避开 8080 strata / 18082 ninfer / 18083 面板）
#
# 用户定案口径（2026-10-08）：**不开 KVMem**、`rk4v4`、**256K 上下文**、
#   采样 temperature 1 / top_k 20 / top_p 0.95 / min_p 0 / presence_penalty 0 /
#   repetition_penalty 1（引擎只接受默认值 1.0）；其余参数可参与调优。
#
# 起服务前提（现场实测，2026-10-08）：
#   1) GPU0 被生产 ninfer 占满 ⇒ 先 `sudo systemctl stop ninfer-watchdog ninfer`
#      （**必须先停 watchdog**，否则它探活失败会把 ninfer 拉起来抢卡）。
#   2) 主机内存：strata(GPU1) 常驻 ~57 GiB ⇒ 起本引擎前先 `free -m` 看 MemAvailable，
#      不足就把 --host-kv-mib 再调小（本脚本默认 512）。
#
# 用法：./run.sh [额外 argv...]        # 额外参数追加在末尾，可覆盖同名项
# ============================================================================
set -u
ROOT=/data/ninfer-fusion-kvmem
MODEL=${MODEL:-/data/ninfer/models/gsq_rco_iq3_s_dflash2_prop.ninfer}
PORT=${PORT:-18084}
GPU=${GPU:-0}

# 运行铁律：用 bundle 自带加载器；**绝不全局 export LD_LIBRARY_PATH**
LOADER=$ROOT/bundle/libs/lib64/ld-linux-x86-64.so.2
LIBPATH=$ROOT/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$ROOT/logs"

exec "$LOADER" --library-path "$LIBPATH" \
  "$ROOT/bundle/bin/ninfer-serve" "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --model-id qwen3.8-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk4v4 \
  --temperature 1 --top-k 20 --top-p 0.95 --min-p 0 \
  --presence-penalty 0 --frequency-penalty 0 \
  --vision --vision-residency overlay --vision-max-merged 12288 \
  --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 \
  --device-state-slots 8 --host-state-slots 2 --host-kv-mib 512 \
  --context-cache-policy rolling --gdn-state-fp16 \
  --spec mtp --draft-tokens 3 --adaptive-mtp \
  --chat-template /data/ninfer/chat_template.jinja \
  --default-reasoning-effort max --default-max-tokens 131072 \
  --cors "$@"
