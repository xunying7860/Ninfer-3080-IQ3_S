#!/bin/bash
# 速度优先臂 2：dflash2 + 关视觉（省 media cache/live 的显存）保住 250880
# 上一臂（VISION=1, DSS=1, HOSTKV=256, CTX=250880）被拒：requires 6,893,758,464 > available 6,444,220,416（差 449 MB）
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
cd "$ROOT"

echo "### 停常驻服务"
sudo systemctl stop ninfer-fusion.service
sleep 4
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "### 臂 df2nv（SPEC=dflash2 DRAFT=12 LMHEAD=1 ADAPTIVE=0 VISION=0 DSS=1 HOSTKV=256 CTX=250880）"
./scripts/x99-arm2.sh df2nv SPEC=dflash2 DRAFT=12 LMHEAD=1 ADAPTIVE=0 VISION=0 DSS=1 HOSTKV=256 CTX=250880 > logs/orch-df2nv.log 2>&1

echo "### 收尾"
P=$(cat logs/arm-df2nv.pid 2>/dev/null); [ -n "${P:-}" ] && kill "$P" 2>/dev/null
sleep 6
pkill -f "engine-run.sh" 2>/dev/null
sleep 3
pkill -f "bundle/bin/ninfer-serve" 2>/dev/null
sleep 4
sudo systemctl start ninfer-fusion.service
echo "### 链结束"
