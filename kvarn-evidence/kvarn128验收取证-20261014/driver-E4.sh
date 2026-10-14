#!/bin/bash
# driver-E4.sh —— 实验4 吞吐/一致性（65536 窗口）：三臂同题面 F1,F2,C1,C2,Z1,Z2,Q1,Q2 各 2 遍；
#                 另跑三针（原 needle-test.py + reasoning 版副本）。
set -u
ROOT=/data/kvarn/g128-验收-20261014
cd "$ROOT" || exit 9
IDS="F1,F2,C1,C2,Z1,Z2,Q1,Q2"
wait_ready() {
  local L=$1
  for i in $(seq 1 120); do
    grep -qa "api base" "$L" && { echo "[wait] api base 出现在 ${i}s"; return 0; }
    kill -0 "$(cat run/pid-$2)" 2>/dev/null || { echo "[wait] 进程已退出"; return 1; }
    sleep 1
  done
  echo "[wait] TIMEOUT"; return 1
}
for A in KV128 RK4V4 INT8; do
  set -a; . "run/$A.env"; set +a
  echo "########## E4 $A kv=$ARM_KV port=$ARM_PORT start $(date '+%F %T') ##########"
  bash run/armctl.sh "$A" startfresh
  L=$(cat "run/log-$A.path")
  wait_ready "$L" "$A" || echo "### NOT READY"
  grep -a -E "engine ready|capacity \||api base" "$L" | sed 's/^/    /'
  echo "### nvidia-smi: $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)"
  echo "--- E4 $A runq3 ids=$IDS ---"
  python3 run/runq3.py "$ARM_PORT" "$L" "$ROOT/ev/$A-runq3.json" "$A" "$IDS"
  echo "--- E4 $A needle (原 /data/ninfer/needle-test.py) ---"
  python3 /data/ninfer/needle-test.py "$ARM_PORT" qwen3.8-27b-coder390-q2lynn 2>&1
  echo "--- E4 $A needle2 (reasoning 版副本) ---"
  python3 run/needle2.py "$ARM_PORT" qwen3.8-27b-coder390-q2lynn 2>&1
  bash run/armctl.sh "$A" stop
  echo "### $A stopped $(date '+%F %T')"
done
echo "########## E4 ALL DONE $(date '+%F %T') ##########"
