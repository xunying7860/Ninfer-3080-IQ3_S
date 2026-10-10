#!/bin/bash
# ① 1M 上下文的内存下限测试：host-kv 12 GiB（期望被启动守卫拒并报出所需页数）
# ② dflash2 draft 7 + 1M 环（YaRN 4.0）：能否起 + 65K/195K 吞吐与针
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
cd "$ROOT"
export SIZES="3100 9300"

echo "=== ① host-kv 12 GiB + 1M 逻辑上下文 ==="
sudo systemctl stop ninfer-fusion.service 2>/dev/null
sleep 4; pkill -f "engine-ru[n]" 2>/dev/null; sleep 2
: > logs/arm-1m-host12g.log
setsid nohup env NINFER_KV_WINDOW=16384 NINFER_KV_RETRIEVE=8192 NINFER_KV_RING=1 NINFER_HOST_PAGEABLE=1 \
  NINFER_KV_REUSE_HOSTBACKED=1 POOL=17920 CTX=1048576 HOSTKV=12288 ROPEYARN=1 \
  SPEC=dflash2 DRAFT=7 LMHEAD=1 ADAPTIVE=0 DSS=1 VISION=1 \
  "$ROOT/scripts/engine-run.sh" >> logs/arm-1m-host12g.log 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 50); do
  if grep -qaE "FATAL|requires|listening on|host_pages" logs/arm-1m-host12g.log; then break; fi
  sleep 5
done
grep -aE "FATAL|requires|listening on|host KV|host_pages|logical" logs/arm-1m-host12g.log | head -6
kill $PID 2>/dev/null; sleep 5; pkill -f "engine-ru[n]" 2>/dev/null; sleep 3

echo "=== ② dflash2 draft 7 + 1M 环（host-kv 24 GiB）==="
./scripts/arm-kv.sh df2d7-1M NINFER_KV_WINDOW=16384 NINFER_KV_RETRIEVE=8192 NINFER_KV_RING=1 \
  NINFER_HOST_PAGEABLE=1 NINFER_KV_REUSE_HOSTBACKED=1 \
  POOL=17920 CTX=1048576 HOSTKV=24576 ROPEYARN=1 \
  SPEC=dflash2 DRAFT=7 LMHEAD=1 ADAPTIVE=0 DSS=1 VISION=1
echo "### 两测结束"
