#!/bin/bash
# 交叉臂链：补齐"两个引擎 × 两个投机头"的 2×2（长任务 65K+2000）
#   已有：新引擎 dflash2@225280 = 213.9 · 生产 mtp3@262144 = 114.2
#   本轮补：新引擎 mtp3@262144（与生产同头同档）· 新引擎 mtp7@262144（调优 mtp）· 生产 bundle dflash2@250880（与新手同头）
# 收尾：把速度档常驻服务拉回来
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
cd "$ROOT"
./scripts/arm-longtask.sh new-mtp3 SPEC=mtp DRAFT=3 ADAPTIVE=1 CTX=262144
./scripts/arm-longtask.sh new-mtp7 SPEC=mtp DRAFT=7 ADAPTIVE=1 CTX=262144
./scripts/prod-dflash2-longtask.sh
echo "### 收尾：拉回速度档常驻服务"
pkill -f "bundle/bin/ninfer-serve" 2>/dev/null
sleep 4
sudo systemctl start ninfer-fusion.service
echo "### 链结束"
