#!/bin/bash
# bug 对照链：新引擎（常驻服务 18084）→ 旧引擎（生产 bundle 临时实例 18085）
# 用例：B 超池题面 / C 打砖回归 / D 思考吃满 / A2 活锁复现（51.5K 题面 + max_tokens 229376）
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
cd "$ROOT"

echo "=== 新引擎（18084）==="
python3 "$ROOT/scripts/bug-probe.py" 18084 qwen3.8-27b NEW2 12400 2>&1 | tee "$ROOT/logs/bugprobe-NEW2.txt"
echo "--- 新引擎权威行 ---" | tee -a "$ROOT/logs/bugprobe-NEW2.txt"
grep -aE "req#[0-9]+ (started|done)|429|exhausted" "$ROOT/logs/ninfer-fusion.log" | tail -10 | tee -a "$ROOT/logs/bugprobe-NEW2.txt"

echo "=== 旧引擎（生产 bundle，临时实例 18085）==="
"$ROOT/scripts/prod-bugprobe.sh"

echo "### 收尾：拉回新引擎常驻服务"
sudo systemctl start ninfer-fusion.service
echo "### 链结束"
