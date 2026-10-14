#!/bin/bash
# driver-E3.sh —— 实验3 容量阶梯：每臂 65536→317440，记录最大可起窗口与失败原文。
set -u
ROOT=/data/kvarn/g128-验收-20261014
cd "$ROOT" || exit 9
for A in KV128 RK4V4 INT8; do
  set -a; . "run/$A.env"; set +a
  for CTX in 65536 131072 196608 262144 317440; do
    echo "########## E3 $A ctx=$CTX start $(date '+%F %T') ##########"
    export ARM_MAXCTX=$CTX
    bash run/armctl.sh "$A" startfresh
    L=$(cat "run/log-$A.path")
    ok=0
    for i in $(seq 1 90); do
      grep -qa "api base" "$L" && { ok=1; break; }
      kill -0 "$(cat run/pid-$A)" 2>/dev/null || break
      sleep 1
    done
    if [ $ok -eq 1 ]; then
      echo "### ctx=$CTX RESULT=OK (ready in ${i}s)"
      grep -a -E "engine ready|capacity \||api base" "$L" | sed 's/^/    /'
    else
      echo "### ctx=$CTX RESULT=FAIL"
      grep -a -iE "error|fatal|out of memory|oom|resize|exceed|cannot|fail|invalid|capacity" "$L" | tail -25 | sed 's/^/    /'
      echo "    --- last 12 lines ---"; tail -12 "$L" | sed 's/^/    /'
    fi
    echo "### nvidia-smi: $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)"
    bash run/armctl.sh "$A" stop
    echo "### $A ctx=$CTX stopped $(date '+%F %T')"
  done
done
echo "########## E3 ALL DONE $(date '+%F %T') ##########"
