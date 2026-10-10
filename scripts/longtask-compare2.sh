#!/bin/bash
# 长任务真实吞吐对比 v2（干净版）
# 修正 v1 的两个坑：
#   1) v1 的 ready 判据只等 curl 有响应 ⇒ 生产引擎还在 loading 就发题，拿到 503
#      ⇒ v2 用**真生成**（16 token，60 s 超时）判活，失败即继续等（同 ninfer-3080x2 的"常驻服务假活"判例）
#   2) v1 跑的实例是臂脚本手动起的引擎，权威行不在服务日志里 ⇒ v2 一律走 systemd 服务，日志切片才准
# 用法: longtask-compare2.sh [prefill_lines] [out_tokens]
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
NLINES=${1:-3100}; OUT=${2:-2000}
OUTDIR=$ROOT/logs/longtask2-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUTDIR"
cd "$ROOT"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/run.log"; }

kill_stray(){  # 只杀"非 systemd 管理"的引擎实例（臂脚本留下的）
  local svc_pids
  svc_pids=$(systemctl show -p MainPID --value ninfer-fusion.service ninfer.service 2>/dev/null | tr '\n' ' ')
  for p in $(pgrep -f 'bundle/bin/ninfer-serve' 2>/dev/null); do
    case " $svc_pids " in *" $p "*) ;; *) log "杀游离引擎 pid=$p"; kill "$p" 2>/dev/null ;; esac
  done
  sleep 4
}

alive(){  # $1=端口；真生成 16 token 判活
  local code
  code=$(curl -s -m 60 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$1/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"say OK"}],"max_tokens":16,"temperature":0,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}}' || true)
  [ "$code" = "200" ]
}

wait_alive(){  # $1=端口 $2=最多等秒
  local port=$1 lim=${2:-600} t=0
  while [ $t -lt $lim ]; do
    if alive "$port"; then log "port $port 真生成 OK（等了 ${t}s）"; return 0; fi
    sleep 10; t=$((t+10))
  done
  log "port $port 超时未就绪"; return 1
}

run_side(){  # $1=标签 $2=端口 $3=引擎日志
  local tag=$1 port=$2 logf=$3 off=0
  [ -f "$logf" ] && off=$(stat -c %s "$logf")
  log "### 长任务：$tag（端口 $port）"
  python3 "$ROOT/scripts/longtask-probe.py" "$port" qwen3.8-27b "$tag" "$NLINES" "$OUT" 2>&1 | tee "$OUTDIR/probe-$tag.txt"
  { echo "--- 引擎权威行（$tag，本轮新增）---"
    tail -c +$((off+1)) "$logf" 2>/dev/null | grep -E "req#[0-9]+ done|capacity \||engine ready|pinning host KV|context cache \|" | tail -12
  } | tee "$OUTDIR/authoritative-$tag.txt"
}

log "=== 清理游离引擎 ==="
kill_stray

log "=== 第一侧：新引擎（ninfer-fusion，调优档）==="
sudo systemctl start ninfer-fusion.service 2>/dev/null
wait_alive 18084 600 && run_side new-fusion 18084 "$ROOT/logs/ninfer-fusion.log"

log "=== 切到生产 ninfer ==="
sudo bash "$ROOT/scripts/switch-gpu0.sh" ninfer >/dev/null 2>&1
kill_stray
wait_alive 18082 600 && run_side prod-ninfer 18082 /data/ninfer/logs/ninfer-serve.log

log "=== 收尾：切回新引擎 ==="
sudo bash "$ROOT/scripts/switch-gpu0.sh" fusion >/dev/null 2>&1
wait_alive 18084 600 || true
log "=== 结果目录：$OUTDIR ==="
