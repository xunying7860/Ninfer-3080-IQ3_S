#!/bin/bash
# KVMem 环单臂：起引擎（5 个 NINFER_KV_* 走环境变量）→ 真生成判活 → kvmem-probe → 权威行 → 收工
# 用法: arm-kv.sh <标签> [KEY=VAL ...]        （KEY=VAL 直接进 env，含 NINFER_KV_* / POOL / CTX / HOSTKV …）
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
ARM=${1:?arm 名}; shift || true
LOG=$ROOT/logs/kv-$ARM.log
PIDF=$ROOT/logs/kv-$ARM.pid
SIZES=${SIZES:-"3100 6100 9300"}
cd "$ROOT"

sudo systemctl stop ninfer-fusion.service 2>/dev/null
sleep 4
pkill -f "engine-run.sh" 2>/dev/null
sleep 2

: > "$LOG"
setsid nohup env "$@" "$ROOT/scripts/engine-run.sh" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "started pid=$(cat "$PIDF") arm=$ARM args=$*"

for i in $(seq 1 90); do
  code=$(curl -s -m 120 -o /dev/null -w "%{http_code}" http://127.0.0.1:18084/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"say OK"}],"max_tokens":16,"temperature":0,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}}' || true)
  [ "$code" = "200" ] && { echo "READY after ${i}x10s"; break; }
  sleep 10
done
grep -aE "capacity \||pinning host KV|context cache \||reuse host-backed|listening on|ERROR|FATAL|requires|\[ring\]" "$LOG" | tail -8

off=$(stat -c %s "$LOG")
python3 "$ROOT/scripts/kvmem-probe.py" 18084 qwen3.8-27b "$ARM" $SIZES 2>&1 | tee "$ROOT/logs/orch-kv-$ARM.txt"
{ echo "--- 权威行（$ARM）---"
  tail -c +$((off+1)) "$LOG" | grep -aE "req#[0-9]+ done|capacity \||\[ring\]|retriev|scoring" | tail -20
} | tee -a "$ROOT/logs/orch-kv-$ARM.txt"

kill "$(cat "$PIDF")" 2>/dev/null; sleep 6
pkill -f "engine-run.sh" 2>/dev/null; sleep 3
echo "### arm $ARM 结束"
