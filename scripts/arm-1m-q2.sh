#!/bin/bash
# 1M 档补测：只跑"定位/计数/新会话复用"三项（修好判据后），195K 单针
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
LOG=$ROOT/logs/kv-1M-q2.log
PIDF=$ROOT/logs/kv-1M-q2.pid
cd "$ROOT"

sudo systemctl stop ninfer-fusion.service 2>/dev/null
sleep 4
pkill -f "engine-ru[n]" 2>/dev/null
sleep 2

: > "$LOG"
setsid nohup env NINFER_KV_WINDOW=16384 NINFER_KV_RETRIEVE=8192 NINFER_KV_RING=1 \
  NINFER_HOST_PAGEABLE=1 NINFER_KV_REUSE_HOSTBACKED=1 \
  POOL=17920 CTX=1048576 HOSTKV=24576 ROPEYARN=1 \
  SPEC=dflash2 DRAFT=12 LMHEAD=1 ADAPTIVE=0 DSS=1 VISION=1 \
  "$ROOT/scripts/engine-run.sh" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"

for i in $(seq 1 120); do
  code=$(curl -s -m 180 -o /dev/null -w "%{http_code}" http://127.0.0.1:18084/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"say OK"}],"max_tokens":16,"temperature":0,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}}' || true)
  [ "$code" = "200" ] && { echo "READY after ${i}x10s"; break; }
  sleep 10
done
grep -aE "capacity \||pinning host KV|engine ready" "$LOG" | tail -3

echo "=== 195K 单针 + 定位 + 计数 + 新会话复用（修好判据）==="
python3 "$ROOT/scripts/quality-probe2.py" 18084 qwen3.8-27b Q195K2 9300 50 2>&1 | tee "$ROOT/logs/orch-q195k2.txt"
echo "=== 引擎权威行 ==="
grep -aE "req#[0-9]+ done" "$LOG" | tail -8

kill "$(cat "$PIDF")" 2>/dev/null; sleep 6
pkill -f "engine-ru[n]" 2>/dev/null; sleep 3
echo "### 补测结束"
