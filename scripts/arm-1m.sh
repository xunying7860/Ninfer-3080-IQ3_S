#!/bin/bash
# 1M 档 + 质量判据（KVMem 环开着）
#   引擎：CTX=1048576（= 四倍原生上限，恰好 1M）+ --rope-yarn（factor = 1048576/262144 = 4.0）
#         POOL=17920（环，设备池 280 页）· HOSTKV=24576 MiB（1M token ≈ 24 GiB 主机层）· dflash2 draft12
#   测试：① 195K 五针召回 + 定位 + 计数（质量判据）
#         ② 900K 两针（30%/80% 深度）→ 极端长度下的召回与 prefill 吞吐
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
LOG=$ROOT/logs/kv-1M.log
PIDF=$ROOT/logs/kv-1M.pid
cd "$ROOT"

sudo systemctl stop ninfer-fusion.service 2>/dev/null
sleep 4
pkill -f "engine-run.sh" 2>/dev/null
sleep 2

: > "$LOG"
setsid nohup env NINFER_KV_WINDOW=16384 NINFER_KV_RETRIEVE=8192 NINFER_KV_RING=1 \
  NINFER_HOST_PAGEABLE=1 NINFER_KV_REUSE_HOSTBACKED=1 \
  POOL=17920 CTX=1048576 HOSTKV=24576 ROPEYARN=1 \
  SPEC=dflash2 DRAFT=12 LMHEAD=1 ADAPTIVE=0 DSS=1 VISION=1 \
  "$ROOT/scripts/engine-run.sh" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "started 1M pid=$(cat "$PIDF")"

for i in $(seq 1 120); do
  code=$(curl -s -m 180 -o /dev/null -w "%{http_code}" http://127.0.0.1:18084/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"say OK"}],"max_tokens":16,"temperature":0,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}}' || true)
  [ "$code" = "200" ] && { echo "READY after ${i}x10s"; break; }
  sleep 10
done
grep -aE "capacity \||pinning host KV|engine ready|listening on|rope|ERROR|FATAL|requires" "$LOG" | tail -8

echo "=== ① 195K 质量判据（五针 + 定位 + 计数，追加式会话）==="
python3 "$ROOT/scripts/quality-probe2.py" 18084 qwen3.8-27b Q195K 9300 5,25,50,75,95 2>&1 | tee "$ROOT/logs/orch-q195k.txt"

echo "=== ② 900K 极端长度（两针 30%/80%）==="
python3 "$ROOT/scripts/quality-probe2.py" 18084 qwen3.8-27b Q900K 43000 30,80 2>&1 | tee "$ROOT/logs/orch-q900k.txt"

echo "=== 引擎权威行 ==="
grep -aE "req#[0-9]+ done|capacity \|" "$LOG" | tail -14

kill "$(cat "$PIDF")" 2>/dev/null; sleep 6
pkill -f "engine-run.sh" 2>/dev/null; sleep 3
echo "### 1M 档结束"
