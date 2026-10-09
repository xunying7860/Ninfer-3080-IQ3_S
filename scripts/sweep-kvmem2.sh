#!/bin/bash
# KVMem 第二轮：补齐窗口轴中点（W=16384/R=8192，带长上下文 decode 用例）+ 环外参照（关环 @ctx 225280）
#   kvA2 WINDOW=16384 RETRIEVE=8192 POOL=17920  ← 与 kvA 同参，但用打补丁后的探针（含 longctx-decode）
#   kvE  关环（不设任何 NINFER_KV_*）CTX=225280 POOL=225280 ← 环的"代价参照"：同样 195K 题面不走主机层
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
cd "$ROOT"
export SIZES="3100 9300"

./scripts/arm-kv.sh kvA2 NINFER_KV_WINDOW=16384 NINFER_KV_RETRIEVE=8192 NINFER_KV_RING=1 NINFER_HOST_PAGEABLE=1 NINFER_KV_REUSE_HOSTBACKED=1 POOL=17920 CTX=262144 HOSTKV=16384 SPEC=dflash2 DRAFT=12 LMHEAD=1 ADAPTIVE=0 DSS=1 VISION=1
./scripts/arm-kv.sh kvE POOL=225280 CTX=225280 HOSTKV=1024 SPEC=dflash2 DRAFT=12 LMHEAD=1 ADAPTIVE=0 DSS=1 VISION=1
echo "### 第二轮结束"
