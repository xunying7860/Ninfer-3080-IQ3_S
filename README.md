# NInfer + Qwen3.8-27B IQ3_S —— RTX 3080 20GB 单卡 **Linux** 部署

同一个引擎、同一个 IQ3_S 制品，换到 **Ubuntu 22.04 + RTX 3080 20GB（单卡，sm_86）** 上**常驻**运行：
systemd 托管、固定用 GPU0、原生 glibc loader（不需要 Docker、不需要装 Python 环境），
另带一套**Web 监控面板**（`monitor/`）。

引擎源 **[iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all)**（默认分支 `master`；生产自编补丁件的基线 pin 在 `60f18c3`）。
部署脚本与 Windows 整合包 **[NInfer-Qwen3.8-27B-IQ3S-sm86-win64](https://github.com/lurenakillgore9-web/NInfer-Qwen3.8-27B-IQ3S-sm86-win64)**（Release 里 10 个 7z 分卷，17.3 GiB）同源，本仓库是它的 Linux 常驻版。
本仓库只放**把服务跑起来所必需的东西**：启动脚本、systemd 单元、看门狗、监控面板；
**不含**模型与引擎二进制（14 GiB 级）。

---

## 这套部署长什么样

| | |
|---|---|
| 引擎 | NInfer sm_86 Linux bundle（`bundle/bin/ninfer-serve`，1,249,869,248 B ≈ 1.16 GiB，md5 `2cbd88ab06318c81b695ec64c72b433d`；自带 loader + glibc，CUDA 用机器的 13.3）。生产跑自编补丁件，厂商原件留档 |
| 模型 | `Qwen3.8-27B-Coder390-EfficientThink-Q2LynnStyle-MTP.ninfer`（13,568,617,472 B，自带 MTP 与 DFlash2 两套草稿头 + 视觉塔）；上一件 `gsq_rco_iq3_s_dflash2_prop.ninfer`（15,017,456,128 B）仍在盘上，未删 |
| 机器 | Ubuntu 22.04.5 / kernel 5.15.0-119 / Xeon E5-2680 v4（28 线程）/ 62 GiB RAM |
| 显卡 | **单张 RTX 3080 20GB**（sm_86）；脚本里用 `CUDA_VISIBLE_DEVICES=0` 固定 GPU0 |
| 驱动 | 615.71.09 |
| 服务 | systemd `ninfer.service` → `/data/ninfer/run.sh`，端口 **18082** |
| 接口 | `http://<host>:18082/v1`（OpenAI 与 Anthropic 双协议），`--model-id qwen3.8-27b-coder390-q2lynn` |
| 聊天模板 | `qwen3.8-froggeric-v22.5` |
| 监控 | `monitor/` 监控面板 → `http://<host>:18083/` |

---

## 生产启动参数

`ninfer.service` 的 `ExecStart` 是 `/data/ninfer/run.sh`，实际命令行（去掉注释后）：

```bash
set -u
ROOT=/data/ninfer
LOADER=$ROOT/bundle/libs/lib64/ld-linux-x86-64.so.2
LIBPATH=$ROOT/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$ROOT/logs"
exec "$LOADER" --library-path "$LIBPATH" "$ROOT/bundle/bin/ninfer-serve" \
  "/data/nvme/ninfer-models/Qwen3.8-27B-Coder390-EfficientThink-Q2LynnStyle-MTP.ninfer" \
  --host 0.0.0.0 --port 18082 --model-id qwen3.8-27b-coder390-q2lynn \
  --max-context 317440 --kv-capacity 317440 --kv-dtype rk4v4 \
  --cuda-graph-allowance-mib 256 \
  --vision --vision-residency overlay --vision-max-merged 4096 --media-cache-mib 512 --media-live-mib 1024 \
  --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 \
  --device-state-slots 1 --host-state-slots 2 --host-kv-mib 512 --context-cache-policy rolling \
  --gdn-state-fp16 --spec mtp --draft-tokens 3 --adaptive-mtp \
  --chat-template "$ROOT/chat_template.jinja" --default-reasoning-effort max --default-max-tokens 131072 \
  "$@"
```

关键取舍都写在 `run.sh` 的注释里（按日期倒序），包括每次改档的**理由与回退点**：

- **dflash2 → MTP**（`--spec mtp --draft-tokens 3 --adaptive-mtp`）：模型同时带 MTP 与 DFlash2 两套草稿头，
  引擎规定两者互斥、只能选一个；`--adaptive-mtp` 每轮在 3..7 之间按实测存活率自适应选宽度，
  避免「固定窗口在不可预测输出上反而更慢」。
- **`--default-max-tokens 131072`**：引擎默认 8192（thinking xhigh 会把额度吃光、正文一个 token 都轮不到）；
  而 `0`（无上限）在这台引擎上是**坏的** —— 0 被解释成 `INT32_MAX`，远大于 KV 池，
  输出租约拿不到页 ⇒ 活锁（见「已知问题」）。约束不等式：`prompt + max_tokens ≤ 池(317440)`。
- **`--device-state-slots 1`**（引擎默认值 = `max-concurrency`）：它是**活跃会话之外**的额外检查点容量。
  曾经设 8，8 份检查点各占 ~210 页（每页 64 token），把 4096 页的分页 KV 池吃到 `allocated 3076`，
  再来一个带图请求要 1020 页 ⇒ `HTTP 429 server overloaded`（Hermes 侧显示
  `context cache exhausted: Paged KV pool cannot resize the page reservation`）。降到 1 后池里少占 ~1470 页。
  实测链式（每发延长上一发）命中率在 slots 1/2/4 下**逐项相同**（0/50.3/66.8/75.1/80.1/83.4/99.0）；
  只有 slots=0 才会退化成每发全量重算。
- **`--max-context`/`--kv-capacity 317440`**（310K，逐档实测的硬顶）：320K 差 173 MiB、315K 差 84 MiB 起不来。
  两个数要分清：**reservation 随 ctx 与槽位变**（每槽 ≈32 MiB），而
  **available runtime capacity 恒为 7,553,613,824 B**（slots 8/4/1 三次读数一字不差，任何旋钮都动不了）。
  现状：`pages 4,960/4,960 | runtime 7.02 GiB | free 253.7 MiB`（贴顶，无余量；要留余量退 300K=307200）。
- **`--host-kv-mib 512`**：主机停放区收小，显存侧留给会话。
- **`--vision-max-merged 4096`**（原 12288）：单图合并 token 上限收小，省显存与池占用。
- **`--max-concurrency 1`**：单槽独占，配合 `--context-cache-policy rolling` 保前缀命中率。
- **`--default-reasoning-effort max`**：思考档位最高，配合上面那个输出上限。

---

## 实测（2026-10-08，取自引擎自报日志与监控面板）

**加载与容量**（`engine ready` / `capacity` 行，引擎自报值）

```
weights 11.3 GiB | 加载 1m 52s（二次启动；冷读更久）
capacity  KV 262,144 tokens, rk4v4, explicit | pages 4,096/4,096 | runtime 8.05 GiB | free 1.79 GiB
context cache  1 active + 8 cached device states | host 2 states, 512.0 MiB KV
```

**吞吐**（单流）

| 指标 | 本机观测 | 口径 |
|---|---|---|
| decode | 60 – 96 tok/s | 引擎每 5 s `throughput` 行；高低由 MTP 接受率决定（实测 50–73%） |
| prefill | 1024 – 1,430 tok/s | 同上；82K 题面的单请求 `prefill 867 tok/s` |
| 端到端累计 | 81.5 tok/s | 面板统计行（37 请求 / 88,063 输出 token） |

**上下文缓存**：会话续写时前缀命中率常态 98–99.6%；本会话累计复用 token 占总提示 token 的约 77%。

**日志里的两种主力行**（运维与面板都基于它们）：

```
throughput | 5.0s | prefill 1.23k tok/s (6,144 tok) | decode 66.4 tok/s (332 tok) | running 1 (decode-ready 1) | batch 1.00 | host 7.4% (369 ms)
req#567 done | openai-chat | tool calls 1 | prompt 113,741 | output 989 | cache 112,870 (99.2%, private endpoint) | TTFT 1.7s | total 15.7s | queue 35.2 ms | prefill 724.0 tok/s | decode 70.6 tok/s | mtp accepted 617/1122 (55.0%)
```

---

## 运维

| 组件 | 位置 | 作用 |
|---|---|---|
| `deploy/ninfer.service` | 装到 `/etc/systemd/system/` | 引擎常驻、自启（`enabled`），`Restart=on-failure` / `RestartSec=15` / `LimitMEMLOCK=infinity` |
| `deploy/ninfer-watchdog.service` + `engine-watchdog.py` | 单元装到 `/etc/systemd/system/`，脚本放 `/data/ninfer/` | **活锁指纹**看门狗：`in_flight>0 且 util≥90% 且 功耗<130 W 且 计数不推进` 持续 ≥180 s ⇒ 自动 `systemctl restart ninfer`；另有 `/health` 连续 360 s 失联的兜底，与引擎空闲时才打一条最小请求的僵尸探针（25 s 超时，model-id 从 `/v1/models` 动态取） |
| `monitor/` | 本目录 | Web 监控面板（见其 README） |
| 日志 | `/data/ninfer/logs/ninfer-serve.log` | 引擎 stdout/stderr；每 5 s 一行 `throughput`，每请求一行 `req#N started/done` |

指纹判定只用三个互相独立的只读量（GPU 利用率、功耗、计数器是否推进），**不在引擎忙时打探针**，
以免污染测速统计；只有在指纹判不出、引擎示空闲时才补一条最小请求的僵尸探针（25 s 超时）。

---

## Web 监控面板（`monitor/`）

`http://<host>:18083/` —— **布局逐字节照抄 [Strata](https://github.com/Niko1221/Strata) 的 Monitor 视图**
（`components.css` / `app.css` / `sprite.svg` 三件套与原站 md5 一致），只把配色换成黑金（纯黑底 + 金）。
数据全部来自 NInfer 自己的只读来源：引擎日志增量 tail、`/metrics`、`/v1/models`、NVML（GPU0）、`/proc`。
**取不到的字段显示 `–`，不编数**；引擎 `/metrics` 拿不到时报 `Engine not reachable`，而不是假装 Idle。

用法与部署见 `monitor/README.md`；面板的独立开发仓库在作者本地，本目录是随部署发布的快照。

---

## 已知问题

1. **活锁（引擎已知）**：`max_tokens` 远大于 KV 池时，输出租约拿不到页。表现是
   GPU 100% 占用但功耗只有 ~95 W、显存带宽 0%、一个主机线程烧满一个核、计数器不推进，
   而 `/health` 仍返回 ok（systemd 的 `Restart=on-failure` 抓不到）。**看门狗按指纹自动重启。**
   约束：`prompt + max_tokens ≤ 池(317440)`。
2. **`cudaErrorIllegalAddress`**：`./src/core/device.cu:402 CUDA_CHECK(cudaStreamSynchronize(endpoint.stream))`，
   2026-10-07 22:25、2026-10-08 12:54、2026-10-08 19:27 各一次，**三次都发生在大题面的 prefill 进行中**
   （host 侧 46–57% CPU，prefill 处于峰值速率）。`Restart=on-failure` 自动拉起，重启到就绪约 2 分钟；
   期间面板显示 `Engine not reachable`。
3. 引擎**不提供**实时 prefill 进度百分比，也没有「专家缓存」类计数 ⇒ 面板对应位置留空，或用等价的真实量替代
   （例如用 `engine ready` 的权重体积画「模型占显存」）。
4. `ninfer.service` 的 `Description=` 已于 2026-10-12 随换档改正为
   `Q2LynnStyle Coder390 MTP repack / GPU0 / 317440 ctx rk4v4 / 端口 18082 / vision overlay 4096 / MTP draft3 adaptive / 设备检查点 1 / 无磁盘层`，与 `run.sh` 一致；改档时两处都要改。

---

## 引擎补丁（2026-10-10 起）

`scripts/` 与 `docs/` 记录我们对 **NInfer 引擎本体**的补丁与运维流程：**单会话直接重启也不再重算**
（实测 **1.2 s** vs 冷 28.0 s），另含 strata 会话文件自动落盘、监控面板 `tg` / `tg_3s` 口径修正。

详见 [`docs/引擎补丁与运维-重启不丢上下文.md`](docs/引擎补丁与运维-重启不丢上下文.md)。


> ⚠ 生产现跑**自编补丁件**（厂商原件已备份为 `…/bundle/bin/ninfer-serve.bak-*-diag`）；
> 以后更新引擎必须按 `docs/` 里的流程从 `iamwavecut/ninfer-all@60f18c3` 重编。

## 目录说明

只留部署与运维必需的东西，一共 115 个文件：

| 路径 | 说明 |
|---|---|
| `run.sh` | 生产启动脚本（含全部取舍注释：为什么选 MTP、为什么输出上限是 131072、每次改档的回退依据） |
| `chat_template.jinja` | 生产对话模板（`qwen3.8-froggeric-v22.5`），由 `run.sh` 指定加载 |
| `engine-watchdog.py` | 活锁看门狗（`--dry-run` / `--selftest` / `--once` 可人工跑） |
| `deploy/` | systemd 单元：`ninfer.service`（引擎）/ `ninfer-watchdog.service`（看门狗）/ `ninfer-monitor.service`（面板）/ `gpu-clocks.service`（开机锁频降压）；`profiles/` 存当前标定件；`archive-20261012-132024-pre-kvarn/` 是 2026-10-12 停生产前的回退点（`RESTORE.txt` 等） |
| `monitor/` | Web 监控面板：`server/`（Python 服务 + 自己的 unit）、`web/`（页面与样式）、`README.md` |
| `test/` | 门禁套件 `gate_test.py`（9 套件 + compare）、`slot-sidecar.py` 与候选实例控制脚本（`launch-cand.sh` / `run-cand.sh` / `restart-cand.sh` / `detach.sh` / `stop-save-hook.sh`）、`probe-master/` 只读取证快照 |
| `scripts/` | 运维与实验脚本（`deploy-to-x99.sh`、各档 `arm-*.sh` / `sweep-*.sh` / `verify-*.py`、崩溃复现 `crash_repro.py`、测速分析 `analyze-dmon.py` / `ab-test.py`、三针 `needle-test.py`） |
| `docs/` | 引擎补丁与运维流程（重启不丢上下文） |

## 许可与致谢

- 模型底座 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)：Apache-2.0
- 引擎 [Neroued/ninfer](https://github.com/Neroued/ninfer)：Apache-2.0
  —— 本 bundle 由作者在 Docker(noble) 里用同一份 sm_86 源构建，运行时自带 loader/glibc（见 `run.sh` 注释）
- 聊天模板 [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) v22.5
- 量化 GSQ-RCO：[ISTA-DASLab](https://huggingface.co/ISTA-DASLab)
- 面板布局参照 [Niko1221/Strata](https://github.com/Niko1221/Strata) 的 Monitor 视图（只换了配色）

本仓库只做**部署适配与运维工具**的整理与分发，不对上述项目主张任何权利；不分发模型与引擎二进制。
