#!/bin/bash
# 单臂跑测（x99 本机）：起引擎 → 等就绪 → 跑探针 → 回抄引擎权威行
# 用法: x99-arm2.sh <arm名> [KEY=VAL ...]   （KEY=VAL 直接透传给 engine-run.sh）
set -u
ARM=${1:?arm 名}; shift || true
ROOT=/data/ninfer-fusion-kvmem
LOG=$ROOT/logs/arm-$ARM.log
PIDF=$ROOT/logs/arm-$ARM.pid

if [ -f "$PIDF" ]; then
  kill "$(cat "$PIDF")" 2>/dev/null || true
  sleep 8
  kill -9 "$(cat "$PIDF")" 2>/dev/null || true
fi
pkill -f "port 18084" 2>/dev/null || true
sleep 2

: > "$LOG"
setsid nohup env "$@" "$ROOT/scripts/engine-run.sh" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "started pid=$(cat "$PIDF") arm=$ARM args=$*"

for i in $(seq 1 150); do
  if grep -q "listening on http" "$LOG" 2>/dev/null; then echo "READY after ${i}x3s"; break; fi
  if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "ENGINE DIED"; tail -25 "$LOG"; exit 1; fi
  sleep 3
done
grep -E "capacity \||engine ready|listening on|pinning host KV|context cache \|" "$LOG" | tail -6

python3 "$ROOT/scripts/tp-probe.py" 18084 qwen3.8-27b "$ARM"
sleep 3

echo "=== 引擎权威行（arm=$ARM）==="
grep -E "req#[0-9]+ done" "$LOG" | tail -20
