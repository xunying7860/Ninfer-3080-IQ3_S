# ninfer-fusion-kvmem（下游衍生引擎 · KVMem 环能力）· x99 部署说明

> 落盘：`/data/ninfer-fusion-kvmem/` —— **SATA 盘**（`/data` 位于 `/` = `ubuntu--vg-ubuntu--lv` = `/dev/sda`；`/data/nvme` 是另一块 NVMe，本部署**不涉及**）。
> 部署日期：2026-10-08。生产 `ninfer`（18082）**未改动**，本目录是其"另一支引擎"的独立落地。

## 1. 这是什么 / 从哪来

| 项 | 值 |
|---|---|
| 源码 | https://github.com/1314521gjy/ninfer-fusion-kvmem `main @ 35aee2a62139bb1d88f7b701a47a1991f52cd131`（2026-10-08） |
| 上游 | NInfer 0.11.0-rtx3090（Apache-2.0），本仓是其下游衍生（含 `src/ops/kvmem/` 环、qw3 策略层等） |
| 为什么自己编 | Release 只发 Windows `ninfer-serve-{86,89,120a}.exe`；x99 是 Linux/glibc 2.35/sm_86 ⇒ **Linux 件必须自编** |
| 怎么编 | 本机 Docker（`nvidia/cuda:13.1.2-devel-ubuntu24.04` → `-runtime`，`CMAKE_CUDA_ARCHITECTURES=86`，目标 `ninfer`+`ninfer-serve`）→ 抽 `ldd` 闭包 → scp 到本机 |
| 件身份 | `bundle/bin/ninfer-serve` 886,842,896 B sha256 `8a3fe32e5fe1a6c3bdfd011d661f67389783c342bcf0cb50e2e76d695e2c5a8a`；`bundle/bin/ninfer` 884,076,472 B sha256 `7ec2f194a801f3ad0ede9c62a4dfb98ea3bbc569e778429cc7c9107832c6de00` |
| 与上游件差异取证 | 本件含 `NINFER_KV_WINDOW` / `NINFER_KV_RETRIEVE` / `NINFER_HOST_PAGEABLE` / `NINFER_KV_REUSE_HOSTBACKED` / `reuse host-backed` / `content scoring`；**上游生产件（`/data/ninfer/bundle`）这些字符串命中 0** |
| 运行库 | **排除 CUDA**：CUDA 用 x99 自带的 `/usr/local/cuda-13.3/lib64`；glibc/FFmpeg/curl 用 bundle 内的 |

## 2. 运行铁律

```bash
LOADER=/data/ninfer-fusion-kvmem/bundle/libs/lib64/ld-linux-x86-64.so.2
LIBPATH=/data/ninfer-fusion-kvmem/bundle/libs/lib/x86_64-linux-gnu:/usr/local/cuda-13.3/lib64:/usr/lib/x86_64-linux-gnu
"$LOADER" --library-path "$LIBPATH" /data/ninfer-fusion-kvmem/bundle/bin/ninfer-serve --help   # rc=0 即闭包可用
```

**绝不全局 `export LD_LIBRARY_PATH`**：宿主 `head/ls` 会加载 bundle 里的新 glibc ⇒ `symbol lookup error ... GLIBC_PRIVATE`（rc=127）。

## 3. 起服务（GPU0）

```bash
# 1) 让出 GPU0（必须先停 watchdog，否则它探活失败会把生产 ninfer 拉起来抢卡）
sudo systemctl stop ninfer-watchdog ninfer
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # 判据：GPU0 ~21 MiB

# 2) 起（默认档 = 调优后：rk4v4/256K/不开 KVMem/draft7；端口 18084）
cd /data/ninfer-fusion-kvmem && ./run.sh
#   回落到基线档：      DRAFT=3 ./run.sh
#   换 prefill 优先档：  DRAFT=7 CHUNK=2048 DSS=2 EXTRA="--prefill-cublas" ./run.sh

# 3) 判就绪：日志出现 listening on http://0.0.0.0:18084，且 GET /v1/models 返回 200（端口在听 ≠ 可服务）
# 4) 收工：Ctrl-C / kill，然后 sudo systemctl start ninfer ninfer-watchdog
```

`run.sh` 可用环境变量覆盖：`DRAFT` `CHUNK` `DSS` `HOSTKV` `EXTRA` `PORT` `GPU` `MODEL`。

## 4. 用户定案参数（固定，别自己改）

```
rk4v4 · --max-context 262144 --kv-capacity 262144 · 不开 KVMem（不设任何 NINFER_KV_*）
--temperature 1 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 0 --frequency-penalty 0
（repetition_penalty 引擎只接受默认 1.0 ⇒ "重复惩罚 1" = 不惩罚）
```

## 5. 调优结论（2026-10-08 实测，详见 `logs/` 与工作区报告）

| 臂 | 变更 | decode 均 tok/s | 32K prefill | 32K 复用 TTFT |
|---|---|---|---|---|
| base | `--draft-tokens 3`（生产档） | 139.0 | 1.37k | 243 ms |
| **draft7（取用）** | `--draft-tokens 7 --adaptive-mtp` | **151.5（+9.0%）** | 1.37k | **156 ms** |
| lk | +`--lookup-ngram 64` | 151.4（无效） | 1.37k | 246 ms |
| a8 | +`--mlp-a8-decode` | 151.4（无效） | 1.37k | 156 ms |
| pf3 | +`chunk 2048`+`--prefill-cublas`（DSS 2） | 148.0 | **1.42k（+3.6%）** | 165 ms |
| 拒启 | `draft 10` / `chunk 4096+cuBLAS` / `chunk 2048+cuBLAS(DSS 8)` | — | — | 超 runtime 上限 8,709,144,576 B |

质量旁证：32K 中段针（`BLUE-FALCON-7419`）**7 个臂全对**，追问 cache 99.9%。

## 6. KVMem 环怎么用（**本轮未启用、未验证**；要用时按此改）

五个环境变量**缺一不可**（没有 CLI 参数），只在 `docs/12-给接收方-agent-的操作手册.md` §5 的口径下成立：

```bash
export NINFER_KV_WINDOW=16384        # 常驻窗口 = 环的门
export NINFER_KV_RETRIEVE=8192       # 检索预算（缺它 = 长题面中段静默答错）
export NINFER_KV_RING=1
export NINFER_HOST_PAGEABLE=1
export NINFER_KV_REUSE_HOSTBACKED=1
# 池算术：池页数 = 窗口/64 + 预填块/64 + 8 ⇒ 16384/64 + 1024/64 + 8 = 280 页 = 17,920 token
# ⇒ 需 --kv-capacity 17920（< max-context）、--max-concurrency 1、argv 必须带 --max-shared-prefixes 0
```

⚠️ 本机主机内存被 strata 占掉 ~57 GiB（`MemAvailable` 仅数 GiB）⇒ 开环前先 `free -m`，`--host-kv-mib` 只能给 512 量级。

## 7. 目录

```
bundle/       引擎二进制 + ldd 闭包 + 自带加载器（+ SHA256.binaries.txt）
run.sh        启动脚本（默认 = 调优档）；run.sh.bak-20261008-pre-tune = 调优前备份
scripts/      engine-run.sh（参数化启动）· x99-arm2.sh（单臂跑测）· tp-probe.py（吞吐/复用/检索探针）
docs/         仓内 README/NOTICE/编译指南 + docs/ 12 篇副本
logs/         arm-<臂>.log（引擎全量日志）· orch-<臂>.log（探针输出 + 权威行回抄）
```
