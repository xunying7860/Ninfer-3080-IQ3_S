#!/bin/bash
# ============================================================================
# ninfer-fusion-kvmem（KVMem 环 + host-backed 复用）· x99 启动脚本
# 位置：/data/ninfer-fusion-kvmem/run-fusion.sh
# 落盘：/data（= ubuntu--vg-ubuntu--lv，sda = SATA 盘）
#
# 状态：**部署时未启动、未验证**（用户 2026-10-08 指示「不用测试」）。
# 本脚本只把参数按仓内 docs/12 §2/§5 的逐字 argv 写齐，供日后人工起服务用。
#
# 起服务前必读（两条硬约束，2026-10-08 现场实测）：
#   1) 两张卡都已被占满：GPU0 = ninfer 18,289 MiB / GPU1 = strata 19,989 MiB
#      ⇒ 必须先 `sudo systemctl stop ninfer`（让出 GPU0）才能起本引擎。
#   2) 主机内存被 strata 占到 MemAvailable ≈ 1.5 GiB
#      ⇒ `--host-kv-mib`（默认 16384）装不下；要么先停 strata，要么把它降到 1024 量级。
#   3) 端口：8080=strata、18082=ninfer、18083=面板 ⇒ 本脚本用 18084。
#
# 用法：./run-fusion.sh                （前台跑，日志同时进 logs/）
#      nohup ./run-fusion.sh > /dev/null 2>&1 &   （后台跑）
# ============================================================================
set -euo pipefail

ROOT=/data/ninfer-fusion-kvmem
MODEL=${MODEL:-/data/ninfer/models/gsq_rco_iq3_s_dflash2_prop.ninfer}
PORT=${PORT:-18084}
GPU=${GPU:-0}

# ---- 运行铁律：用 bundle 自带加载器，绝不全局 export LD_LIBRARY_PATH ----
# （全局 export 会让宿主的 head/ls 加载新 glibc ⇒ symbol lookup error ... GLIBC_PRIVATE, rc=127）
LOADER="$ROOT/bundle/libs/lib64/ld-linux-x86-64.so.2"
LIBPATH="$ROOT/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu"

# ---- KVMem 环的五个环境变量：缺一不可（docs/12 §5.1）----
# 少 NINFER_KV_RETRIEVE ⇒ 长题面中段答不出来，而且日志零错误（最危险的失效形态）
export NINFER_KV_WINDOW=16384        # 常驻窗口（设备上真留多少 token）= 环的开关
export NINFER_KV_RETRIEVE=8192       # 检索预算（缺它 = 静默答错）
export NINFER_KV_RING=1              # 历史开关（新件已不读）
export NINFER_HOST_PAGEABLE=1        # 宿主页可换入
export NINFER_KV_REUSE_HOSTBACKED=1  # 允许复用宿主驻留页

mkdir -p "$ROOT/logs"
exec env CUDA_VISIBLE_DEVICES=$GPU "$LOADER" --library-path "$LIBPATH" \
  "$ROOT/bundle/bin/ninfer-serve" "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --model-id qwen3.8-27b-kvmem \
  --max-context 262144 \
  --kv-capacity 17920 \
  --kv-dtype rk4v4 \
  --host-kv-mib 16384 \
  --max-shared-prefixes 0 \
  --prefill-chunk 1024 \
  --spec dflash2 --draft-tokens 12 --lm-head-draft \
  --vision --vision-residency overlay --vision-max-merged 12288 \
  --max-concurrency 1 --max-pending-requests 16 \
  --device-state-slots 1 --host-state-slots 2 \
  --gdn-state-fp16 \
  --chat-template /data/ninfer/chat_template.jinja \
  --default-reasoning-effort none \
  --default-max-tokens 32768 \
  --cors
