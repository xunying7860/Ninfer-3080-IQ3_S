#!/usr/bin/env bash
# 候选件启动器（任务 B 门禁测试用；临时，不入 systemd、不改生产 run.sh、不动生产二进制）
# 与生产 run.sh 逐字同参数，只改三处：
#   1) 二进制 → $CAND_BIN（默认 bundle/bin/ninfer-serve.cand = 60f18c3 + 已取证补丁 #1~#6，md5 21bd4486…）
#   2) 加磁盘层 --disk-kv-path/--disk-kv-gib/--disk-kv-restore（生产当前是关的）
#   3) 日志与磁盘层目录由 env 指定；stderr 一并进日志，才能看到 [diag] 落盘行
set -u
ROOT=/data/ninfer
LOADER=$ROOT/bundle/libs/lib64/ld-linux-x86-64.so.2
LIBPATH=$ROOT/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu
CAND_BIN="${CAND_BIN:-$ROOT/bundle/bin/ninfer-serve.cand}"
CAND_DISK="${CAND_DISK:-/data/nvme/ninfer-disktier-test}"
CAND_EXTRA="${CAND_EXTRA:-}"
# CAND_NO_DISK=1 ⇒ 不加磁盘层参数（测上游 slot-save 时要把它关掉，才知道是谁在起作用）
CAND_NO_DISK="${CAND_NO_DISK:-0}"
# CUDA graph 额度（生产 256）：换引擎时这块预算会变，可用 CAND_ALLOWANCE 单变量调
CAND_ALLOWANCE="${CAND_ALLOWANCE:-256}"
CAND_DISK_FLAGS=(--disk-kv-path "$CAND_DISK" --disk-kv-gib 32 --disk-kv-restore)
[ "$CAND_NO_DISK" = "1" ] && CAND_DISK_FLAGS=()
export CUDA_VISIBLE_DEVICES=0
# /data/nvme 属 root，普通用户不能直接在其下建目录 ⇒ 磁盘层目录用 sudo 预建并交回所有权
# （2026-10-12 实测：不加这两行候选件直接 FATAL「disk KV tier failed to open … Permission denied」）
sudo -n mkdir -p "$CAND_DISK" 2>/dev/null || mkdir -p "$CAND_DISK"
sudo -n chown "$(id -un)": "$(id -un):" "$CAND_DISK" 2>/dev/null || true
mkdir -p "$ROOT/logs" "$ROOT/test" "$CAND_DISK"
echo "[launch] $(date '+%F %T') pid=$$ bin=$CAND_BIN disk=$CAND_DISK extra=$CAND_EXTRA" >> "$ROOT/logs/cand-serve.log"
# shellcheck disable=SC2086
exec "$LOADER" --library-path "$LIBPATH" "$CAND_BIN" \
  "/data/nvme/ninfer-models/Qwen3.8-27B-Coder390-EfficientThink-Q2LynnStyle-MTP.ninfer" \
  --host 0.0.0.0 --port 18082 --model-id qwen3.8-27b-coder390-q2lynn \
  --max-context 317440 --kv-capacity 317440 --kv-dtype rk4v4 \
  --cuda-graph-allowance-mib "$CAND_ALLOWANCE" \
  --vision --vision-residency overlay --vision-max-merged 4096 --media-cache-mib 512 --media-live-mib 1024 \
  --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 \
  --device-state-slots 1 --host-state-slots 2 --host-kv-mib 512 --context-cache-policy rolling \
  --gdn-state-fp16 --spec mtp --draft-tokens 3 --adaptive-mtp \
  --chat-template "$ROOT/chat_template.jinja" --default-reasoning-effort max --default-max-tokens 131072 \
  "${CAND_DISK_FLAGS[@]}" \
  $CAND_EXTRA
