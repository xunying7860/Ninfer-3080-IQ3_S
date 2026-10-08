#!/bin/bash
# ninfer-serve 在 x99(Ubuntu 22.04, glibc 2.35) 原生运行：
#   本机 Docker(noble) 编出 sm_86 Linux 版，运行时用自带 glibc/loader，CUDA 用 x99 的 13.3。
# 卡位：GPU0（用户定案：strata 用 GPU1，ninfer 用 GPU0）
#
# 2026-10-03 21:3x 换代（用户定案「ninfer 这个可以替换生产脚本了」）：
#   模型 → gsq_rco_iq3_s_dflash2_prop.ninfer（自带 dflash2 草稿头 + 视觉塔）
#   KV   → rk4v4（比 rk8v4 省 ~34% 显存/token）
#   投机 → --spec dflash2 --draft-tokens 7 --lm-head-draft（2026-10-03 23:24 由 12 改为 7：实测中文创作接受率仅 4-6%，窗口 12 白算过多；代价是数数字基线会低于窗口 12）
#   不开 kvmem：本脚本与环境里不设 NINFER_KV_WINDOW / NINFER_KV_RETRIEVE / NINFER_KV_RING /
#     NINFER_HOST_PAGEABLE / NINFER_KV_REUSE_HOSTBACKED 这 5 个变量（设了就会开内容打分环）
#   接口面：--host 0.0.0.0 --port 18082 --model-id qwen3.8-27b、--default-reasoning-effort max
#
# 2026-10-04 09:3x 输出上限解除（用户定案逐字「--default-max-tokens 这个不能关了吗，不设上限，
#   小模型本来就需要多思考提高质量」）：
#   加 --default-max-tokens 0 ⇒ 生成到上下文用完为止。引擎默认是 8192（help 原文：
#     --default-max-tokens defaults to 8192 when omitted; 0 generates until the context runs out），
#     本脚本原先没写它，导致 xhigh 思考 + 8192 上限把输出额度吃光、正文一个 token 都轮不到
#     （实例：Hermes 会话 20261004_085731_ce07c0，引擎日志 req#17 output limit / output 8,192，
#      Hermes 侧 reasoning 49,953 字符而 content 长度 0 ⇒ 界面显示被截断的思考链）。
#   风险（已知悉）：单请求最长可生成到 262,144 上下文耗尽（prompt 32K 时最多约 230K token，
#     按 ~69 tok/s 约 55 分钟），且 --max-concurrency 1 期间独占实例；Hermes 侧
#     agent.gateway_timeout 1800 s 可能先掐断。若要兜底可加 --default-thinking-budget。
#   回退点：/data/ninfer/run.sh.bak-20261004-maxout-unbounded
#
# 2026-10-04 11:5x 内存回撤（目标：MemAvailable ≈1.2 GiB，用户「只留1.2G余量」的量级）：
#   strata 按 min-free 1280 停放后 RSS 53.3 GiB；随后 ninfer 重启又钉 2 GiB host KV + ~4 GiB media/shmem
#   ⇒ MemAvailable 掉到 111 MiB（无 OOM，但余量为零）。本次仅把 --host-kv-mib 2048 → 1024（−1 GiB 内存），
#   显存侧（--device-state-slots 8）不动；主机停放区变小后，超出的续段由磁盘层（--disk-kv-*）接管恢复。
#   回退点：/data/ninfer/run.sh.bak-20261004-hostkv1g
#
# 2026-10-04 11:2x 换档 dflash2 → MTP（用户逐字「ninfer生产换成mtp，依旧256K上下文。然后把显存吃满」）：
#   --spec dflash2 --draft-tokens 7 --lm-head-draft  →  --spec mtp --draft-tokens 7 --adaptive-mtp
#     * 模型同时带 MTP 与 DFlash2 两套草稿头（头部 JSON 已核），引擎规定两者互斥，只能选一个
#     * --adaptive-mtp：每轮在 3..7 之间按实测存活率/回合成本自适应选宽度（greedy 输出不变），
#       可避免「固定窗口在不可预测输出上更慢」；--lm-head-draft 属 dflash2 路数，换档时去掉
#   --device-state-slots 1 → 2（GPU 检查点槽 +1 = 把余下显存花在多留一份可复用状态上；
#     ctx 仍 262144，kv-capacity 不超池）
#   回退点：/data/ninfer/run.sh.bak-20261004-mtp
#
# 2026-10-04 10:0x 修正「不设上限」（同一 bug 两次复现，根因取证）：
#   症状：GPU0 100% 占用但只有 95 W、0% 显存带宽、一个主机线程烧满一个核，
#     llamacpp:tokens_predicted_total 与 draft_tokens_total 6 秒内零增长，
#     kv_cache_usage_ratio=1，requests_processing=1 ⇒ 引擎活锁，不是「模型在想」。
#   两次都出现在 thinking xhigh 且 max output = 2,147,483,647 的请求上；
#     同档 xhigh 但上限 256/512/8192 的请求全部正常收尾（08:49、09:01 的 req#17/#18）。
#   根因（引擎方手册 04-卡死与循环的防治.md §1 第 11 条 ENGINE_DEAD + L1 处方）：
#     max_tokens 远大于池 token 数 ⇒ 输出租约拿不到页 ⇒ 卡死；内测 32,000 崩 / 64 活，2/2 复现。
#     本机池 = 262,144；而 0 被引擎解释成 INT32_MAX(2,147,483,647) ⇒ 越池 8000 倍。
#   处置：--default-max-tokens 65536（旧默认 8192 的 8 倍，仍是「大而有限」）。
#     取舍：0=无上限在这台引擎上是坏的（引擎自带文档明令 max_tokens ≤ 池），
#       65536 已足够长思考（~69 tok/s 下约 16 分钟），且 prompt 到 19.6 万仍不越池。
#     若要再放宽，上限受 (262144 - 最大 prompt) 约束，别超 131072。
#   回退点：/data/ninfer/run.sh.bak-20261004-maxout64k
#
# 2026-10-04 10:2x 上限数值（用户逐字定案「改成--default-max-tokens 229376」）：
#   池 = 262144 ⇒ 本值只在 prompt ≤ 32768 时满足手册不等式 prompt + max_tokens ≤ 池；
#   prompt 超过这个数（本机已出现过 133,675 的 prompt）会再次越池 ⇒ 输出租约拿不到页 ⇒ 活锁。
#   若要「任意题面都安全」，上限需 ≤ 池 − 最大题面；本机 agent 会话常见 30K~130K 题面。
#   回退点：/data/ninfer/run.sh.bak-20261004-maxout-229376
#
# 2026-10-04 10:1x 定上限数值（用户逐字定案「--default-max-tokens 131072」）：
#   65536 → 131072。硬约束（手册 L1）：prompt + max_tokens ≤ 池 262,144
#     ⇒ 131072 只在 prompt ≤ 131072 时安全；prompt 超过这个数（本机已出现过 133,675）
#       就会再次越池卡死。若要彻底安全，另一条路是把上限降到 98304，或让 prompt 不越 131072。
#   回退点：/data/ninfer/run.sh.bak-20261004-maxout-131072
#
# 2026-10-03 22:0x 二次压边（用户定案「ninfer再往显存边缘压」）：
#   上下文 250880 → 262144（= 模型 max_position_embeddings 的原生上限，pages 4096/4096）
#   手段：--cuda-graph-allowance-mib 1792。该值「从 KV sizing 预算里扣」（引擎 --help 原文）；
#     默认按设备档算出的额度把 262144 挡在门外（requires 6585185280 > 6444220416 available，
#     差 134.4 MiB）；显式降到 256 MiB 后装得下，余量 93.9 MiB。
#   实测（测试实例，同模型同参数，GPU0）：
#     262144 起得来：CUDA graphs ready 2.1s、capacity KV 262,144 tokens rk4v4 explicit、
#       pages 4096/4096、runtime 5.91 GiB、free 255.7 MiB、GPU0 19,851/20,480 MiB；
#     数数字 800 出自停 decode 250.2 / 253.9 / 254.0 tok/s、dflash2 接受 89.6 / 91.0 / 91.0%
#       —— 与 250880 档（250.2 / 253.5 / 254.0、89.6 / 91.0%）逐位一致，降额度没换来解码变慢。
#   边界实测（engine 自己的 requires/available 报数）：
#     默认额度：254272 可、254336 不可（requires 6445265664 > 6444220416，差 1.0 MiB）；
#     allowance 256：294912 不可（requires 6937508864 > 6444220416，差 470.4 MiB）⇒ 显存上限其实
#       到 ~267328 token，但那已超过模型原生 262144（要 --rope-yarn 才谈得上质量），故取 262144 为止。
#   回退点：/data/ninfer/run.sh.bak-20261003-2155-ctx250880（250880 + 默认 cuda-graph 额度）
set -u
ROOT=/data/ninfer
LOADER=$ROOT/bundle/libs/lib64/ld-linux-x86-64.so.2
LIBPATH=$ROOT/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$ROOT/logs"
exec "$LOADER" --library-path "$LIBPATH" "$ROOT/bundle/bin/ninfer-serve" \
  "$ROOT/models/gsq_rco_iq3_s_dflash2_prop.ninfer" \
  --host 0.0.0.0 --port 18082 --model-id qwen3.8-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk4v4 \
  --cuda-graph-allowance-mib 1792 \
  --vision --vision-residency overlay --vision-max-merged 12288 --media-cache-mib 512 --media-live-mib 1024 \
  --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 \
  --device-state-slots 8 --host-state-slots 2 --host-kv-mib 512 --context-cache-policy rolling \
  --gdn-state-fp16 --spec mtp --draft-tokens 3 --adaptive-mtp \
  --chat-template "$ROOT/chat_template.jinja" --default-reasoning-effort max --default-max-tokens 131072 \
  "$@"

# 2026-10-04 14:5x 定案 A（用户逐字「两个引擎都改A」）：输出上限 229376 → 131072。
#   活锁根因 = prompt_tokens + max_tokens > KV池(262144)。229376 时安全题面上限仅 32768，
#   而带 44 个工具的 Hermes 会话题面 5 万起 ⇒ 必踩（14:28 那条 51,560 卡成 96W 空转）。
# 2026-10-04 15:0x 用户指令「ninfer的换成mtp3，吃满显存」：
#   draft-tokens 7 → 3（经典 MTP3；--adaptive-mtp 的上界收窄到 3 = 单档，CUDA graph 由 5 档减到 1 档
#     ⇒ 腾出运行时预算给 device-state-slots）。
#   显存：device-state-slots 29 → 先用 40 探上限（引擎会用 `requires A bytes, but only B available` 拒启，
#     按 B/A 反解真实可用槽数，再落到实测最大值）。
#   131072 ⇒ 安全题面上限抬到 131072（4×余量），131K 输出对 xhigh 思考仍充裕。
#   残余情形由 ninfer-watchdog.service（plan B）兜底自动重启。
# 2026-10-04 15:1x 显存上限实测（探 40 被拒后反解）：
#   拒启原文: requires 9,546,740,480 B, but only 8,709,144,576 B available（40 槽）
#   与历史 34 槽拒启联立 ⇒ 每槽 76,919,149 B + 固定基线 6,469,968,518 B(6.03 GiB)
#   ⇒ 29 槽需 8,700,624,839 ≤ 8,709,144,576 ✓（余 8.5 MB）；30 槽需 8,777,543,988 ✗
#   ⇒ **29 槽就是引擎硬上限**（19,893 MiB 用量 / 仅 212 MiB 空闲 ⇒ 显存已实质吃满）。
#   注：换 MTP3 并不改变这个天花板——基线（KV池+runtime）一样大。
