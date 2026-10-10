#!/bin/bash
# 速度优先臂 3：dflash2 + 关视觉 + ctx 让到 225280（按 18,166 B/token 反算：250880 时超 420 MB ⇒ 需砍 ~23K token）
# 前两臂拒启原文：
#   VISION=1 CTX=250880 → requires 6,893,758,464 > 6,444,220,416（差 449 MB）
#   VISION=0 CTX=250880 → requires 6,883,235,840 > 6,463,094,784（差 420 MB）⇒ 视觉只省 ~19 MB，瓶颈是 ctx 本身
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
cd "$ROOT"

echo "### 停常驻服务"
sudo systemctl stop ninfer-fusion.service
sleep 4
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "### 臂 df2-225k（SPEC=dflash2 DRAFT=12 LMHEAD=1 ADAPTIVE=0 VISION=0 DSS=1 HOSTKV=256 CTX=225280）"
./scripts/x99-arm2.sh df2-225k SPEC=dflash2 DRAFT=12 LMHEAD=1 ADAPTIVE=0 VISION=0 DSS=1 HOSTKV=256 CTX=225280 > logs/orch-df2-225k.log 2>&1

echo "### 收尾"
P=$(cat logs/arm-df2-225k.pid 2>/dev/null); [ -n "${P:-}" ] && kill "$P" 2>/dev/null
sleep 6
pkill -f "engine-run.sh" 2>/dev/null
sleep 3
pkill -f "bundle/bin/ninfer-serve" 2>/dev/null
sleep 4
sudo systemctl start ninfer-fusion.service
echo "### 链结束"
