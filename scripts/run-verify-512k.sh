#!/bin/bash
# 512K 档回归测试编排：跑前后各记一次崩溃计数，测试期间 tail 日志看有无 worker crash
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
LOG=$ROOT/logs/ninfer-fusion.log
OUT=$ROOT/logs/verify-512k-$(date +%Y%m%d-%H%M%S).txt
cd "$ROOT"

echo "=== 前置检查 ===" | tee "$OUT"
grep -aE "capacity \|" "$LOG" | tail -1 | cut -c1-140 | tee -a "$OUT"
tr "\0" " " < /proc/$(systemctl show -p MainPID --value ninfer-fusion.service)/cmdline | tr " " "\n" | grep -A1 -E "default-max-tokens|kv-capacity|max-context" | grep -v "^--$" | tee -a "$OUT"
BEFORE_CRASH=$(grep -ac "worker crash" "$LOG")
echo "崩溃计数（跑前）= $BEFORE_CRASH" | tee -a "$OUT"

echo "=== 开始测试（约 15 分钟）===" | tee -a "$OUT"
python3 -u "$ROOT/scripts/verify-512k.py" 18084 qwen3.8-27b v512k 2>&1 | tee -a "$OUT"

AFTER_CRASH=$(grep -ac "worker crash" "$LOG")
echo "崩溃计数（跑后）= $AFTER_CRASH" | tee -a "$OUT"
echo "=== 测试期间新增崩溃 ===" | tee -a "$OUT"
tail -n +1 "$LOG" | grep -aA2 "worker crash" | tail -6 | tee -a "$OUT"
echo "=== 引擎存活 ===" | tee -a "$OUT"
curl -s -m 8 -o /dev/null -w "models=%{http_code} " http://127.0.0.1:18084/v1/models | tee -a "$OUT"
curl -s -m 60 -o /dev/null -w "chat=%{http_code}\n" http://127.0.0.1:18084/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"say ok"}],"max_tokens":4}' | tee -a "$OUT"
echo "=== 结果文件：$OUT ===" | tee -a "$OUT"
