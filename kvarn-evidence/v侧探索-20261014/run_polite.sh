#!/bin/bash
# 礼貌跑：只在 GPU0 确认空闲（无引擎进程、/data/kvarn 60s 无写入）时起跑；
# 一旦别的子代理起引擎，vside_run.py 的看门狗 3s 内自杀让出 GPU0。最多 8 次尝试。
cd /data/kvarn/vside-20261014 || exit 1
export PYTHONPATH=/data/kvarn/pylibs19:/data/kvarn/pylibs:/data/kvarn:/data/kvarn/g128
export PPL_DOC='/data/kvarn/ppl_doc.txt;/data/kvarn/ppl_doc2.txt'
export VSIDE_TOKS=2944,6016
export PPL_CHARS=20000
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VSIDE_WATCH=1
export VSIDE_RESULTS=/data/kvarn/vside-20261014/results.tsv
A='K4Vprom_k16,K4Vprom_k32,K4Vprom_k64,K4Vprom_k128'
B='K4V3chan_lsq01r1,K4V3chan_ls1,K4Vprom_k32,K4Vprom_k64,K4Vprom_k128'
export VSIDE_ARMS_A="$A"
export VSIDE_ARMS_B="$B"

need_more() {
  for a in $(echo "$A" | tr ',' ' '); do
    awk -F'\t' -v f=ppl_doc.txt -v a="$a" '$1==f && $2==a{ok=1} END{exit !ok}' results.tsv 2>/dev/null || return 0
  done
  for a in $(echo "$B" | tr ',' ' '); do
    awk -F'\t' -v f=ppl_doc2.txt -v a="$a" '$1==f && $2==a{ok=1} END{exit !ok}' results.tsv 2>/dev/null || return 0
  done
  return 1
}

DEADLINE=$(( $(date +%s) + 2700 ))   # 45 分钟后自动收工，别在别人机器上长挂
for att in $(seq 1 8); do
  if ! need_more; then echo "[$(date +%T)] 全部臂已完成"; break; fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then echo "[$(date +%T)] 到点收工（还有臂未跑）"; break; fi
  while true; do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F', ' '$1=="0"{print $2}' | tr -d ' MiB')
    procs=$(ps -eo args | grep -E 'ld-linux-x86-64|ninfer-perplexity|ninfer-serve' | grep -v grep | wc -l)
    writes=$(find /data/kvarn -newermt '-60 seconds' -type f 2>/dev/null | wc -l)
    if [ "${used:-9999}" -lt 200 ] && [ "$procs" -eq 0 ] && [ "$writes" -eq 0 ]; then
      sleep 25
      used2=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F', ' '$1=="0"{print $2}' | tr -d ' MiB')
      procs2=$(ps -eo args | grep -E 'ld-linux-x86-64|ninfer-perplexity|ninfer-serve' | grep -v grep | wc -l)
      writes2=$(find /data/kvarn -newermt '-25 seconds' -type f 2>/dev/null | wc -l)
      if [ "${used2:-9999}" -lt 200 ] && [ "$procs2" -eq 0 ] && [ "$writes2" -eq 0 ]; then
        echo "[$(date +%T)] 窗口连续空闲 (used=${used2}MiB procs=$procs2 writes=$writes2) → 起跑"
        break
      fi
      echo "[$(date +%T)] 二次检查仍忙 (used=${used2}MiB procs=$procs2 writes=$writes2)"
      sleep 60
      continue
    fi
    echo "[$(date +%T)] 忙 (used=${used}MiB procs=$procs writes=$writes)，等 60s"
    sleep 60
  done
  TS=$(date +%Y%m%d-%H%M%S)
  /data/comfy/venv/bin/python -u vside_run.py > "logs/vsab-$TS.log" 2>&1
  rc=$?
  echo "[$(date +%T)] 第 $att 次结束 rc=$rc  log=logs/vsab-$TS.log"
  if [ $rc -ne 0 ]; then echo "  非零退出（看门狗/异常），等 120s"; sleep 120; fi
done
echo "POLITE_DONE $(date +%T)"
