#!/bin/bash
# 礼貌跑 round2（2026-10-14）：
#   语料 A（ppl_doc.txt, 2944 tok）  ：锚点复核 + K4Vprom_k8
#   语料 B（ppl_doc2.txt, 6016 tok） ：锚点 + K4Vprom_k8 / k16
#   语料 C（ppl_doc3.txt, 2432 tok） ：锚点 + 六臂（chan_lsq01r1 / chan_ls1 / prom_k8/k16/k32/k64/k128）
# 只在 GPU0 连续空闲（<200MiB、无引擎进程、/data/kvarn 60s 无写入）时起跑；
# vside_run.py 进程内看门狗 3s 内自杀让位。结果 append 到 results.tsv，可续跑。
cd /data/kvarn/vside-20261014 || exit 1
export PYTHONPATH=/data/kvarn/pylibs19:/data/kvarn/pylibs:/data/kvarn:/data/kvarn/g128
export PPL_DOC='/data/kvarn/ppl_doc.txt;/data/kvarn/ppl_doc2.txt;/data/kvarn/ppl_doc3.txt'
export VSIDE_TOKS='2944,6016,2432'
export PPL_CHARS=20000
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VSIDE_WATCH=1
export VSIDE_RESULTS=/data/kvarn/vside-20261014/results.tsv
export VSIDE_ARMS_0='baseline_fp16,rk4v4_post,K4V4_g128,K4Vprom_k8'
export VSIDE_ARMS_1='baseline_fp16,rk4v4_post,K4V4_g128,K4Vprom_k8,K4Vprom_k16'
export VSIDE_ARMS_2='baseline_fp16,rk4v4_post,K4V4_g128,K4V3chan_lsq01r1,K4V3chan_ls1,K4Vprom_k8,K4Vprom_k16,K4Vprom_k32,K4Vprom_k64,K4Vprom_k128'

# 需要跑的臂（tag, arm）——与上面三个列表一致
NEED="ppl_doc.txt|baseline_fp16 ppl_doc.txt|rk4v4_post ppl_doc.txt|K4V4_g128 ppl_doc.txt|K4Vprom_k8
ppl_doc2.txt|baseline_fp16 ppl_doc2.txt|rk4v4_post ppl_doc2.txt|K4V4_g128 ppl_doc2.txt|K4Vprom_k8 ppl_doc2.txt|K4Vprom_k16
ppl_doc3.txt|baseline_fp16 ppl_doc3.txt|rk4v4_post ppl_doc3.txt|K4V4_g128 ppl_doc3.txt|K4V3chan_lsq01r1 ppl_doc3.txt|K4V3chan_ls1 ppl_doc3.txt|K4Vprom_k8 ppl_doc3.txt|K4Vprom_k16 ppl_doc3.txt|K4Vprom_k32 ppl_doc3.txt|K4Vprom_k64 ppl_doc3.txt|K4Vprom_k128"

need_more() {
  for pair in $NEED; do
    f="${pair%%|*}"; a="${pair##*|}"
    awk -F'\t' -v f="$f" -v a="$a" '$1==f && $2==a{ok=1} END{exit !ok}' results.tsv 2>/dev/null || return 0
  done
  return 1
}

ts() { while IFS= read -r l; do printf '[%s] %s\n' "$(date +%T)" "$l"; done; }

DEADLINE=$(( $(date +%s) + 2700 ))
echo "[$(date +%T)] 礼貌跑 round2 启动；缺口：$(need_more && echo 有 || echo 无)"
for att in $(seq 1 6); do
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
      sleep 60; continue
    fi
    echo "[$(date +%T)] 忙 (used=${used}MiB procs=$procs writes=$writes)，等 60s"
    sleep 60
  done
  TS=$(date +%Y%m%d-%H%M%S)
  /data/comfy/venv/bin/python -u vside_run.py 2>&1 | ts > "logs/round2-$TS.log"
  rc=${PIPESTATUS[0]}
  echo "[$(date +%T)] 第 $att 次结束 rc=$rc log=logs/round2-$TS.log"
  if [ $rc -ne 0 ]; then echo "[$(date +%T)]   非零退出（看门狗/异常），等 120s"; sleep 120; fi
done
echo "[$(date +%T)] ROUND2_DONE"
