#!/bin/bash
# 长任务真实吞吐对比：新引擎（ninfer-fusion, 18084）vs 生产 ninfer（ninfer.service, 18082）
# 同一份长任务、同一采样口径、同一卡（GPU0 串行），只比引擎自己 req#N done 行。
# 用法: longtask-compare.sh [prefill_lines] [out_tokens]     默认 3100 行（≈32K token）+ 2000 输出
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
NLINES=${1:-3100}; OUT=${2:-2000}
OUTDIR=$ROOT/logs/longtask-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUTDIR"
cd "$ROOT"

run_side () {  # $1=标签 $2=端口 $3=model-id $4=引擎日志路径
  local tag=$1 port=$2 mid=$3 logf=$4
  echo "### $tag（端口 $port）"
  local off=0
  [ -f "$logf" ] && off=$(stat -c %s "$logf")
  python3 "$ROOT/scripts/longtask-probe.py" "$port" "$mid" "$tag" "$NLINES" "$OUT" 2>&1 | tee "$OUTDIR/probe-$tag.txt"
  {
    echo "--- 引擎权威行（$tag，只取本轮新增）---"
    tail -c +$((off + 1)) "$logf" 2>/dev/null | grep -E "req#[0-9]+ done|capacity \||engine ready|pinning host KV|context cache \|" | tail -12
  } | tee "$OUTDIR/authoritative-$tag.txt"
}

# 1) 新引擎
sudo systemctl start ninfer-fusion.service 2>/dev/null
for i in $(seq 1 100); do curl -s -m 3 -o /dev/null http://127.0.0.1:18084/v1/models && break; sleep 3; done
run_side new-fusion 18084 qwen3.8-27b "$ROOT/logs/ninfer-fusion.log"

# 2) 生产 ninfer（让出/收回 GPU0 由 switch-gpu0.sh 统一管）
sudo bash "$ROOT/scripts/switch-gpu0.sh" ninfer >/dev/null 2>&1
for i in $(seq 1 100); do curl -s -m 3 -o /dev/null http://127.0.0.1:18082/v1/models && break; sleep 3; done
run_side prod-ninfer 18082 qwen3.8-27b /data/ninfer/logs/ninfer-serve.log

echo "### 结果目录：$OUTDIR"
echo "### 收尾：切回新引擎"
sudo bash "$ROOT/scripts/switch-gpu0.sh" fusion >/dev/null 2>&1
