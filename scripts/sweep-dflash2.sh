#!/bin/bash
# dflash2 对比臂链（用户追问"dflash2 你测过了？" → 现测）
#   df2-262k：256K + dflash2（预期装不下，取"拒启原文"作证据）
#   df2-250k：250880 + dflash2（历史该形状的最大 ctx）
# 收尾：把常驻服务 ninfer-fusion.service 拉回来（mtp draft7 / 256K 档）
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
cd "$ROOT"

echo "### 先停常驻服务（让出 GPU0 与 18084）"
sudo systemctl stop ninfer-fusion.service
sleep 4
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "### 臂 df2-250k（SPEC=dflash2 DRAFT=12 LMHEAD=1 ADAPTIVE=0 CTX=250880）"
./scripts/x99-arm2.sh df2-250k SPEC=dflash2 DRAFT=12 LMHEAD=1 ADAPTIVE=0 CTX=250880 > logs/orch-df2-250k.log 2>&1

echo "### 收尾：清掉臂实例，拉回常驻服务"
P=$(cat logs/arm-df2-250k.pid 2>/dev/null); [ -n "${P:-}" ] && kill "$P" 2>/dev/null
sleep 6
pkill -f "engine-run.sh" 2>/dev/null
sleep 3
pkill -f "bundle/bin/ninfer-serve" 2>/dev/null
sleep 4
sudo systemctl start ninfer-fusion.service
echo "### 链结束（常驻服务已发起）"
