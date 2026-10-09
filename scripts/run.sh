#!/bin/bash
# ============================================================================
# ninfer-fusion-kvmem（下游衍生引擎 · KVMem 环能力）· x99 / GPU0 生产脚本
# 位置：/data/ninfer-fusion-kvmem/run.sh   盘位：/data（SATA，sda）   端口：18084
#
# 【用户定案（2026-10-08/09）】先速度、上下文后议；采样固定
#   temperature 1 · top_k 20 · top_p 0.95 · min_p 0 · presence 0 · frequency 0
#   （repetition_penalty 引擎只接受默认 1.0 ⇒ 1 = 不惩罚）；不开 KVMem。
#
# 【速度调优结论（实测，引擎自己 req#N done 行；数数字 400 出、关思考）】
#   mtp  --draft-tokens 3（原生产档）   decode 139.0 tok/s   prefill(32K) 1.37k   空闲 1.65 GiB  ctx 262144
#   mtp  --draft-tokens 7（+adaptive）  decode 151.5 tok/s   prefill(32K) 1.37k   空闲 1.60 GiB  ctx 262144
#   **dflash2 --draft-tokens 12（本档）  decode 250.0 tok/s   prefill(32K) 1.36k   空闲 833 MiB   ctx 225280**
#   ⇒ 选 dflash2：**+65% decode**（对照 mtp draft7），接受率 91.2%。
#   代价（装得下的边界，全部为引擎拒启原文反解）：
#     · dflash2 草稿头权重 13.4 GiB（mtp 档 11.3 GiB）⇒ runtime 上限从 8,709,144,576 B 降到 6,463,094,784 B
#     · ctx 250880 装不下（差 420 MB）；**225280 可**（free 833 MiB）
#     · dflash2 **不能**配 --adaptive-mtp（引擎：--adaptive-mtp requires --spec mtp）
#     · 状态槽压到 1、host KV 256 MiB（让权重的代价；代价是设备检查点变少）
#   未采纳：--lookup-ngram 64 / --mlp-a8-decode（逐位相同，无效）；draft 10（拒启）
#   `--vision`：本轮测试臂 VISION=0；视觉开销实测只 ~19 MB，余量 833 MiB 够，恢复为 ON（要省显存可设 VISION=0）
#
# 【起服务前提】GPU0 必须先让出：`sudo bash scripts/switch-gpu0.sh fusion`
#   （先停 ninfer-watchdog，否则它探活失败会拉起生产 ninfer 抢卡）
# ============================================================================
set -u
ROOT=/data/ninfer-fusion-kvmem
MODEL=${MODEL:-/data/ninfer/models/gsq_rco_iq3_s_dflash2_prop.ninfer}
PORT=${PORT:-18084}
GPU=${GPU:-0}

# —— 档位（默认速度档 225280；512K/1M 走 systemd drop-in 覆盖这些变量）——
#   384K 档（2026-10-09 起为线上档）：CTX=393216 HOSTKV=5120 ROPEYARN=1 DSS=10
#     （512K 档的旧值 CTX=524288 HOSTKV=8192 见 git 历史；池一直是 POOL=158720）
#     + 环环境变量 NINFER_KV_WINDOW=16384 NINFER_KV_RETRIEVE=8192 NINFER_KV_RING=1
#       NINFER_HOST_PAGEABLE=1 NINFER_KV_REUSE_HOSTBACKED=1（引擎直接读环境，见 drop-in）
#     依据：主机层每页 1.146 MB（64 token/页）；512K=8192 页，池 280 页 ⇒ 需 7912 页 = 9.07 GiB。
CTX=${CTX:-225280}         # 逻辑上下文（线上档由 drop-in 覆盖为 393216 = 384K）
POOL=${POOL:-$CTX}         # --kv-capacity：**环语义要求 POOL < CTX**（默认相等=不开环）
SPEC=${SPEC:-dflash2}
DRAFT=${DRAFT:-12}
ADAPTIVE=${ADAPTIVE:-0}    # dflash2 必须 0（--adaptive-mtp 只配 mtp）
DSS=${DSS:-1}
HOSTSLOTS=${HOSTSLOTS:-2}  # --host-state-slots：主机侧常驻的会话检查点份数（多吃 RAM，不吃显存）
PRIVCONT=${PRIVCONT:-}     # --max-private-continuations：私有续写目录容量（默认 2×max-concurrency；空=用默认）
RECOVER=${RECOVER:-1}       # --recover-invariant-failures：**内部不变量破坏时只丢在途请求、引擎继续服务**
#                             ★ 2026-10-09 关键修复：今天 4 个崩溃签名（active KV snapshot page ownership /
#                             Paged KV single-page materialization / KV activation reservation is stale /
#                             KV activation capacity reservation changed）全是 failure_class.h 分类表里的
#                             `std::logic_error -> Invariant`（= 默认"崩掉整个引擎"）；该开关是引擎自带、
#                             help 原文 "fail the active requests and keep serving the queue instead of failing the engine"。
LEASE=${LEASE:-0}           # --kv-lease-growth=1：输出页只预留 4096 token 窗口按需扩展（替代"整份 max_tokens 一次预留"）
#                             ★ 依据 fork Bug手册 §1.2 的候选修法；不开时 15:54 实测崩于
#                             `Paged KV single-page materialization exceeds reservation`（paged_kv_cache.cpp:528）
HOSTKV=${HOSTKV:-256}
STATSMS=${STATSMS:-1000}    # --log-stats-interval-ms：引擎 throughput 统计行周期（默认 5000）
#                             收紧到 1000ms 是为了面板的 tg（每秒刷新）：该行是引擎唯一
#                             连续的解码速率信号（指标与计数器都只在请求完成时跳一次）。
MAXTOK=${MAXTOK:-131072}   # --default-max-tokens：**必须 ≤ 池 token 数**（Bug手册 §1.2：max_tokens ≫ 池 ⇒ worker 崩 + 全 503 不自愈）
CHUNK=${CHUNK:-1024}
VISION=${VISION:-1}
EXTRA=${EXTRA:-}

LOADER=$ROOT/bundle/libs/lib64/ld-linux-x86-64.so.2
LIBPATH=$ROOT/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$ROOT/logs"

ARGS=(--host 0.0.0.0 --port "$PORT" --model-id qwen3.8-27b
      --max-context "$CTX" --kv-capacity "$POOL" --kv-dtype rk4v4
      --temperature 1 --top-k 20 --top-p 0.95 --min-p 0
      --presence-penalty 0 --frequency-penalty 0
      --max-concurrency 1 --max-pending-requests 16 --prefill-chunk "$CHUNK"
      --device-state-slots "$DSS" --host-state-slots "$HOSTSLOTS" --host-kv-mib "$HOSTKV"
      --context-cache-policy rolling --gdn-state-fp16
      --spec "$SPEC" --draft-tokens "$DRAFT" --lm-head-draft
      --chat-template /data/ninfer/chat_template.jinja
      --default-reasoning-effort max --default-max-tokens "$MAXTOK"
      --log-stats-interval-ms "$STATSMS"
      --cors)
[ "$ADAPTIVE" = "1" ] && ARGS+=(--adaptive-mtp)
[ "${ROPEYARN:-0}" = "1" ] && ARGS+=(--rope-yarn)
[ -n "${ROPEYARN_FACTOR:-}" ] && ARGS+=(--rope-yarn-factor "$ROPEYARN_FACTOR")
[ -n "$PRIVCONT" ] && ARGS+=(--max-private-continuations "$PRIVCONT")
[ "$LEASE" = "1" ] && ARGS+=(--kv-lease-growth)
[ "$RECOVER" = "1" ] && ARGS+=(--recover-invariant-failures)
[ "$VISION" = "1" ] && ARGS+=(--vision --vision-residency overlay --vision-max-merged 12288)
if [ -n "$EXTRA" ]; then for f in $EXTRA; do ARGS+=("$f"); done; fi

echo "ENV: CTX=$CTX POOL=$POOL HOSTKV=$HOSTKV ROPEYARN=${ROPEYARN:-0} LEASE=${LEASE:-0} SPEC=$SPEC DRAFT=$DRAFT DSS=$DSS KV_RING=${NINFER_KV_RING:-0} KV_WINDOW=${NINFER_KV_WINDOW:-} KV_RETRIEVE=${NINFER_KV_RETRIEVE:-}"

exec "$LOADER" --library-path "$LIBPATH" "$ROOT/bundle/bin/ninfer-serve" "$MODEL" "${ARGS[@]}"
