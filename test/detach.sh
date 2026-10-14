#!/usr/bin/env bash
# 分离进程跑一个 suite（抗 ssh 回收 SIGHUP）；日志 test/logs/<suite>.log，产物 test/ev/<out>
# 用法: detach.sh <suite> <evidence.json>
set -u
cd /data/ninfer/test || exit 9
mkdir -p logs ev
s="$1"; out="ev/$2"
GATE_LOG="${GATE_LOG:-/data/ninfer/logs/ninfer-serve.log}" \
GATE_RESTART_CMD="${GATE_RESTART_CMD:-sudo -n systemctl restart ninfer.service}" \
setsid nohup python3 gate_test.py run --suite "$s" --out "$out" </dev/null >"logs/$s.log" 2>&1 &
echo "detached suite=$s log=logs/$s.log out=$out pid=$!"
