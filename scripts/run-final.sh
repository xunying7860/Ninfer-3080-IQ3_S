#!/bin/bash
# ============================================================================
# ninfer-fusion-kvmem（下游衍生引擎 · KVMem 环能力）· x99 / GPU0 生产脚本
# 位置：/data/ninfer-fusion-kvmem/run.sh   盘位：/data（SATA，sda）
# 端口：18084（避开 8080 strata / 18082 ninfer / 18083 面板）
#
# 【调优结论（2026-10-08 实测，见 report-x99-部署与吞吐-20261008.md）】
#   用户定案固定项：rk4v4 / 256K(262144) / 不开 KVMem /
#     temperature 1 · top_k 20 · top_p 0.95 · min_p 0 · presence 0 · frequency 0
#     （repetition_penalty 引擎只接受默认 1.0 ⇒ 1 = 不惩罚，等价默认）
#   调优落点：`--draft-tokens 3 → 7`（配 --adaptive-mtp）⇒ 数数字 decode 均值
#     139.0 → 151.6 tok/s（+9.1%，mtp 接受 93.6–99.7%）；32K prefill 1.37k tok/s。
#   未采纳（实测无效或装不下）：
#     · --lookup-ngram 64  ：decode 155.2/147.2，与不加逐位相同（无效）
#     · --mlp-a8-decode    ：decode 155.3/147.0，与不加逐位相同（无效）
#     · --draft-tokens 10  ：启动被拒（requires 10,031,256,832 > 8,709,144,576）
#     · --prefill-cublas + --prefill-chunk 4096/2048：4096 被拒；2048 需把
#       --device-state-slots 压到 2 才装得下，prefill 1.37k→1.42k（+3.6%），
#       但空闲显存 1.60 GiB→0.95 GiB ⇒ 作为可选项保留，不作默认。
#
# 【起服务前提】GPU0 必须先让出：`sudo systemctl stop ninfer-watchdog ninfer`
#   （先停 watchdog，否则它探活失败会拉起生产 ninfer 抢卡）
# ============================================================================
set -u
ROOT=/data/ninfer-fusion-kvmem
MODEL=${MODEL:-/data/ninfer/models/gsq_rco_iq3_s_dflash2_prop.ninfer}
PORT=${PORT:-18084}
GPU=${GPU:-0}

DRAFT=${DRAFT:-7}          # 调优值：3 → 7
CHUNK=${CHUNK:-1024}
DSS=${DSS:-8}
HOSTKV=${HOSTKV:-512}
EXTRA=${EXTRA:-}           # 可选："--prefill-cublas" 需同时 DSS=2 CHUNK=2048

LOADER=$ROOT/bundle/libs/lib64/ld-linux-x86-64.so.2
LIBPATH=$ROOT/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$ROOT/logs"

ARGS=(--host 0.0.0.0 --port "$PORT" --model-id qwen3.8-27b
      --max-context 262144 --kv-capacity 262144 --kv-dtype rk4v4
      --temperature 1 --top-k 20 --top-p 0.95 --min-p 0
      --presence-penalty 0 --frequency-penalty 0
      --vision --vision-residency overlay --vision-max-merged 12288
      --max-concurrency 1 --max-pending-requests 16 --prefill-chunk "$CHUNK"
      --device-state-slots "$DSS" --host-state-slots 2 --host-kv-mib "$HOSTKV"
      --context-cache-policy rolling --gdn-state-fp16
      --spec mtp --draft-tokens "$DRAFT" --adaptive-mtp
      --chat-template /data/ninfer/chat_template.jinja
      --default-reasoning-effort max --default-max-tokens 131072
      --cors)
if [ -n "$EXTRA" ]; then for f in $EXTRA; do ARGS+=("$f"); done; fi

exec "$LOADER" --library-path "$LIBPATH" "$ROOT/bundle/bin/ninfer-serve" "$MODEL" "${ARGS[@]}"
