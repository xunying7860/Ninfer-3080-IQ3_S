#!/bin/bash
# 长任务单臂（新引擎）：起引擎 → 真生成判活 → 跑 65K+2000 长任务 → 回抄权威行
# 用法: arm-longtask.sh <标签> [KEY=VAL ...]      （KEY=VAL 透传给 engine-run.sh）
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
ARM=${1:?arm 名}; shift || true
LOG=$ROOT/logs/arm-$ARM.log
PIDF=$ROOT/logs/arm-$ARM.pid
cd "$ROOT"

sudo systemctl stop ninfer-fusion.service 2>/dev/null
sleep 4
pkill -f "engine-run.sh" 2>/dev/null
sleep 2

: > "$LOG"
setsid nohup env "$@" "$ROOT/scripts/engine-run.sh" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "started pid=$(cat "$PIDF") arm=$ARM args=$*"

for i in $(seq 1 60); do
  code=$(curl -s -m 90 -o /dev/null -w "%{http_code}" http://127.0.0.1:18084/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"say OK"}],"max_tokens":16,"temperature":0,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}}' || true)
  [ "$code" = "200" ] && { echo "READY after ${i}x10s"; break; }
  sleep 10
done
grep -E "capacity \||listening on" "$LOG" | tail -2

off=$(stat -c %s "$LOG")
python3 "$ROOT/scripts/longtask-probe.py" 18084 qwen3.8-27b "$ARM" 3100 2000 2>&1 | tee "$ROOT/logs/orch-long-$ARM.log"
{ echo "--- 权威行（$ARM）---"
  tail -c +$((off+1)) "$LOG" | grep -E "req#[0-9]+ done|capacity \|" | tail -8
} | tee -a "$ROOT/logs/orch-long-$ARM.log"

kill "$(cat "$PIDF")" 2>/dev/null; sleep 6
pkill -f "engine-run.sh" 2>/dev/null; sleep 3
echo "### arm $ARM 结束"
