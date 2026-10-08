#!/bin/bash
# 生产 bundle + mtp3（与生产 run.sh 同形状，只改端口/加采样旗标）长任务臂
# 目的：补齐 2×2（两个引擎 × {mtp3, dflash2}），与"新引擎 mtp3"逐字同口径
# 不改 /data/ninfer/run.sh，只手动起一次临时实例。
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
P=/data/ninfer/bundle
LOG=$ROOT/logs/prod-mtp3.log
PIDF=$ROOT/logs/prod-mtp3.pid
cd "$ROOT"

sudo systemctl stop ninfer-fusion.service 2>/dev/null
sleep 4
pkill -f "engine-run.sh" 2>/dev/null
sleep 2

: > "$LOG"
setsid nohup env CUDA_VISIBLE_DEVICES=0 \
  "$P/libs/lib64/ld-linux-x86-64.so.2" --library-path "$P/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu" \
  "$P/bin/ninfer-serve" /data/ninfer/models/gsq_rco_iq3_s_dflash2_prop.ninfer \
  --host 0.0.0.0 --port 18084 --model-id qwen3.8-27b \
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
echo "started prod-mtp3 pid=$(cat "$PIDF")"

for i in $(seq 1 60); do
  code=$(curl -s -m 90 -o /dev/null -w "%{http_code}" http://127.0.0.1:18084/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"say OK"}],"max_tokens":16,"temperature":0,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}}' || true)
  [ "$code" = "200" ] && { echo "READY after ${i}x10s"; break; }
  sleep 10
done
grep -E "capacity \||listening on|ERROR|FATAL|requires" "$LOG" | tail -3

off=$(stat -c %s "$LOG")
python3 "$ROOT/scripts/longtask-probe.py" 18084 qwen3.8-27b prod-mtp3 3100 2000 2>&1 | tee "$ROOT/logs/orch-long-prod-mtp3.log"
{ echo "--- 权威行（prod-mtp3）---"
  tail -c +$((off+1)) "$LOG" | grep -E "req#[0-9]+ done|capacity \|" | tail -8
} | tee -a "$ROOT/logs/orch-long-prod-mtp3.log"

kill "$(cat "$PIDF")" 2>/dev/null; sleep 6
pkill -f "bundle/bin/ninfer-serve" 2>/dev/null; sleep 3
echo "### prod-mtp3 结束"
