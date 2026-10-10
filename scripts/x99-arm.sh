#!/bin/bash
# 单臂跑测（在 x99 本机执行）：起引擎 → 等就绪 → 跑探针 → 回抄引擎权威行
# 用法: x99-arm.sh <arm名> [run.sh 的额外参数...]
set -u
ARM=${1:?arm 名}; shift || true
ROOT=/data/ninfer-fusion-kvmem
LOG=$ROOT/logs/arm-$ARM.log
PIDF=$ROOT/logs/arm-$ARM.pid

# 1) 清掉上一臂（只清我们自己起的 18084 实例，不动生产）
if [ -f "$PIDF" ]; then
  kill "$(cat "$PIDF")" 2>/dev/null || true
  sleep 8
  kill -9 "$(cat "$PIDF")" 2>/dev/null || true
fi
pkill -f "port 18084" 2>/dev/null || true
sleep 2

# 2) 起本臂
: > "$LOG"
echo "ARGS: $*" >> "$LOG"
setsid nohup "$ROOT/run.sh" "$@" >> "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "started pid=$(cat "$PIDF") arm=$ARM"

# 3) 等就绪（listening on http://）最多 420 s
for i in $(seq 1 140); do
  if grep -q "listening on http" "$LOG" 2>/dev/null; then echo "READY after ${i}x3s"; break; fi
  if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "ENGINE DIED"; tail -30 "$LOG"; exit 1; fi
  sleep 3
done
grep -E "loading weights|capacity \||free |listening on|pinning host KV|context cache \|" "$LOG" | tail -8

# 4) 探针
python3 "$ROOT/scripts/tp-probe.py" 18084 qwen3.8-27b "$ARM"
sleep 3

# 5) 引擎权威行回抄
echo "=== 引擎权威行（arm=$ARM）==="
grep -E "req#[0-9]+ (started|done)|capacity \||device profile|calibrating routes" "$LOG" | tail -40
