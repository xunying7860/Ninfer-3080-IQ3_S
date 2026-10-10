#!/bin/bash
# 速度优先臂：dflash2（草稿头 13.4 GiB 权重，ctx 让到 250880）+ 精简状态槽
# 依据：2026-10-03 同模型同卡实测该形状 decode 250.2–254.0 tok/s（对照 mtp draft7 的 151.5）
#   dflash2 不能配 --adaptive-mtp（引擎：--adaptive-mtp requires --spec mtp）⇒ ADAPTIVE=0
#   状态槽从 8 压到 1、host KV 512→256 是让 13.4 GiB 权重装得下的代价（2026-10-03 用的就是这组）
# 收尾：把常驻服务拉回来
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
cd "$ROOT"

echo "### 停常驻服务"
sudo systemctl stop ninfer-fusion.service
sleep 4
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "### 臂 df2lean（SPEC=dflash2 DRAFT=12 LMHEAD=1 ADAPTIVE=0 DSS=1 HOSTKV=256 CTX=250880）"
./scripts/x99-arm2.sh df2lean SPEC=dflash2 DRAFT=12 LMHEAD=1 ADAPTIVE=0 DSS=1 HOSTKV=256 CTX=250880 > logs/orch-df2lean.log 2>&1

echo "### 收尾"
P=$(cat logs/arm-df2lean.pid 2>/dev/null); [ -n "${P:-}" ] && kill "$P" 2>/dev/null
sleep 6
pkill -f "engine-run.sh" 2>/dev/null
sleep 3
pkill -f "bundle/bin/ninfer-serve" 2>/dev/null
sleep 4
sudo systemctl start ninfer-fusion.service
echo "### 链结束"
