#!/bin/bash
# driver-E2.sh —— 实验2 门禁：kvarn128 / rk4v4 / int8 各一轮 9 套件（ngram 默认开）；
#                 另加 kvarn128 与 rk4v4 两臂的 --ngram-draft-tokens 0 档（只跑门禁）。
set -u
ROOT=/data/kvarn/g128-验收-20261014
cd "$ROOT" || exit 9
wait_ready() {
  local L=$1
  for i in $(seq 1 120); do
    grep -qa "api base" "$L" && { echo "[wait] api base 出现在 ${i}s"; return 0; }
    kill -0 "$(cat run/pid-$2)" 2>/dev/null || { echo "[wait] 进程已退出"; return 1; }
    sleep 1
  done
  echo "[wait] TIMEOUT 120s"; return 1
}
run_arm() {
  local A=$1 TAG=$2
  set -a; . "run/$A.env"; set +a
  echo "########## E2 $A tag=$TAG kv=$ARM_KV extra='${ARM_EXTRA:-}' port=$ARM_PORT start $(date '+%F %T') ##########"
  echo "### profile md5 BEFORE: $(md5sum "$ARM_PROFILE")"
  bash run/armctl.sh "$A" startfresh
  local L; L=$(cat "run/log-$A.path")
  wait_ready "$L" "$A" || echo "### NOT READY (继续尝试)"
  echo "### boot lines:"; grep -a -E "build |device profile|engine ready|capacity \||api base" "$L" | sed 's/^/    /'
  echo "### nvidia-smi boot: $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)"
  GATE_DOC_LINES=1200 GATE_NONCE=gate-prod bash run/gates.sh "$A" "$TAG"
  echo "### profile md5 AFTER: $(md5sum "$ARM_PROFILE")"
  echo "### nvidia-smi post: $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)"
  bash run/armctl.sh "$A" stop
  echo "### $A stopped $(date '+%F %T')"
}
run_arm KV128  r1
run_arm RK4V4  r1
run_arm INT8   r1
run_arm KV128NG ng0
run_arm RK4V4NG ng0
echo "########## E2 ALL DONE $(date '+%F %T') ##########"
