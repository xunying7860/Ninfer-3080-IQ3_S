#!/bin/bash
# "吃满显存"对照臂：1M 逻辑上下文 + 环，但设备池从 280 页抬到 3,584 页（229,376 token，吃掉那 4.25 GiB 余量）
# 依据：每页 1.146 MB（64 token），加一页就少要一页主机内存（主机需求 18.5 GiB → ~14.7 GiB）
set -uo pipefail
ROOT=/data/ninfer-fusion-kvmem
cd "$ROOT"
export SIZES="3100 9300"
./scripts/arm-kv.sh bigpool-1M NINFER_KV_WINDOW=16384 NINFER_KV_RETRIEVE=8192 NINFER_KV_RING=1 \
  NINFER_HOST_PAGEABLE=1 NINFER_KV_REUSE_HOSTBACKED=1 \
  POOL=229376 CTX=1048576 HOSTKV=16384 ROPEYARN=1 \
  SPEC=dflash2 DRAFT=7 LMHEAD=1 ADAPTIVE=0 DSS=1 VISION=1
echo "### 大池臂结束"
