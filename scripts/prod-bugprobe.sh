#!/bin/bash
# 旧引擎（生产 bundle）bug 对照：手动起临时实例（端口 18085，与生产 run.sh 同形状），跑 bug-probe
# 生产脚本 /data/ninfer/run.sh 与 unit 一字节不动。
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
P=/data/ninfer/bundle
LOG=$ROOT/logs/prod-bugprobe.log
PIDF=$ROOT/logs/prod-bugprobe.pid
cd "$ROOT"

sudo systemctl stop ninfer-fusion.service 2>/dev/null
sleep 4
pkill -f "engine-run.sh" 2>/dev/null
sleep 2

: > "$LOG"
setsid nohup env CUDA_VISIBLE_DEVICES=0 \
  "$P/libs/lib64/ld-linux-x86-64.so.2" --library-path "$P/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu" \
  "$P/bin/ninfer-serve" /data/ninfer/models/gsq_rco_iq3_s_dflash2_prop.ninfer \
  --host 0.0.0.0 --port 18085 --model-id qwen3.8-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk4v4 \
  --cuda-graph-allowance-mib 1792 \
  --temperature 1 --top-k 20 --top-p 0.95 --min-p 0 \
  --presence-penalty 0 --frequency-penalty 0 \
  --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 \
  --device-state-slots 8 --host-state-slots 2 --host-kv-mib 512 \
  --context-cache-policy rolling --gdn-state-fp16 \
  --spec mtp --draft-tokens 3 --adaptive-mtp \
  --vision --vision-residency overlay --vision-max-merged 12288 \
  --media-cache-mib 512 --media-live-mib 1024 \
  --chat-template /data/ninfer/chat_template.jinja \
  --default-reasoning-effort max --default-max-tokens 131072 --cors \
  >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "started prod-bugprobe pid=$(cat "$PIDF")"

for i in $(seq 1 60); do
  code=$(curl -s -m 90 -o /dev/null -w "%{http_code}" http://127.0.0.1:18085/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"say OK"}],"max_tokens":16,"temperature":0,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}}' || true)
  [ "$code" = "200" ] && { echo "READY after ${i}x10s"; break; }
  sleep 10
done
grep -E "capacity \||listening on|ERROR|FATAL|requires" "$LOG" | tail -3

off=$(stat -c %s "$LOG")
python3 "$ROOT/scripts/bug-probe.py" 18085 qwen3.8-27b OLD 13500 2>&1 | tee "$ROOT/logs/bugprobe-OLD.txt"
{ echo "--- 权威行（OLD）---"
  tail -c +$((off+1)) "$LOG" | grep -aE "req#[0-9]+ (started|done)|429|exhausted|ERROR|crash" | tail -12
} | tee -a "$ROOT/logs/bugprobe-OLD.txt"

kill "$(cat "$PIDF")" 2>/dev/null; sleep 6
pkill -f "bundle/bin/ninfer-serve" 2>/dev/null; sleep 3
echo "### prod-bugprobe 结束"
