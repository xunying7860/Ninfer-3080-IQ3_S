#!/bin/bash
# 启动期守卫 A/B：新引擎（本仓衍生）vs 生产 bundle（上游）
# 用例（fork 自称已修的"启动期"缺陷，正是老引擎的形态）：
#   B06 打分器并发门：KVMem 打分器开 + --max-concurrency 2 ⇒ 新引擎应拒启并点名；老引擎应照常起
#   B21 启动守卫：NINFER_KV_SINK 页数 + 4 > 池页数 ⇒ 新引擎应拒启并给可读报文；老引擎应照常起
# 判据：看日志出现 listening on（STARTED）/ FATAL·ERROR（REFUSED）/ 进程消失（DIED）
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
NEW_BIN=$ROOT/bundle/bin/ninfer-serve; NEW_LD=$ROOT/bundle/libs/lib64/ld-linux-x86-64.so.2
NEW_LIB=$ROOT/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu
OLD_BIN=/data/ninfer/bundle/bin/ninfer-serve; OLD_LD=/data/ninfer/bundle/libs/lib64/ld-linux-x86-64.so.2
OLD_LIB=/data/ninfer/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu
MODEL=/data/ninfer/models/gsq_rco_iq3_s_dflash2_prop.ninfer
OUT=$ROOT/logs/startup-guards-$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
PORT=18086
cd "$ROOT"

try () {  # $1=标签 $2=用例 $3=LD $4=LIB $5=BIN ; 其余 KEY=VAL 之后 "--" 之后为 argv
  local tag=$1 case=$2 ld=$3 lib=$4 bin=$5; shift 5
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local log="$OUT/$case-$tag.log"
  : > "$log"
  setsid nohup env "${envs[@]}" CUDA_VISIBLE_DEVICES=0 "$ld" --library-path "$lib" "$bin" "$MODEL" "$@" >> "$log" 2>&1 < /dev/null &
  local pid=$!
  local verdict="DIED" t=0
  while [ $t -lt 300 ]; do
    if grep -qa "listening on http" "$log"; then verdict="STARTED"; break; fi
    if grep -qaE "FATAL|requires|must hold|throw|unknown argument|KVMem content scoring" "$log"; then verdict="REFUSED"; break; fi
    if ! kill -0 $pid 2>/dev/null; then verdict="DIED"; break; fi
    sleep 5; t=$((t+5))
  done
  sleep 2
  echo "### [$case / $tag] verdict=$verdict  (${t}s)"
  grep -aE "FATAL|ERROR|requires|must hold|KVMem content scoring|listening on|engine ready" "$log" | head -4
  kill $pid 2>/dev/null; sleep 4; pkill -f "port $PORT" 2>/dev/null; sleep 2
}

echo "=== B06 打分器并发门（KVMem 打分器开 + 并发 2）==="
try new B06 "$NEW_LD" "$NEW_LIB" "$NEW_BIN" NINFER_TERNARY_KVMEM=1 -- --host 127.0.0.1 --port $PORT --max-context 262144 --kv-capacity 262144 --kv-dtype rk4v4 --max-concurrency 2 --prefill-chunk 1024 --spec mtp --draft-tokens 3
try old B06 "$OLD_LD" "$OLD_LIB" "$OLD_BIN" NINFER_TERNARY_KVMEM=1 -- --host 127.0.0.1 --port $PORT --max-context 262144 --kv-capacity 262144 --kv-dtype rk4v4 --max-concurrency 2 --prefill-chunk 1024 --spec mtp --draft-tokens 3

echo "=== B21 启动守卫（sink 33 页 + 池 34 页 ⇒ 应拒启）==="
try new B21 "$NEW_LD" "$NEW_LIB" "$NEW_BIN" NINFER_KV_SINK=2112 -- --host 127.0.0.1 --port $PORT --max-context 2176 --kv-capacity 2176 --kv-dtype rk4v4 --max-concurrency 1 --prefill-chunk 256 --spec mtp --draft-tokens 3
try old B21 "$OLD_LD" "$OLD_LIB" "$OLD_BIN" NINFER_KV_SINK=2112 -- --host 127.0.0.1 --port $PORT --max-context 2176 --kv-capacity 2176 --kv-dtype rk4v4 --max-concurrency 1 --prefill-chunk 256 --spec mtp --draft-tokens 3

echo "=== 结果目录：$OUT ==="
