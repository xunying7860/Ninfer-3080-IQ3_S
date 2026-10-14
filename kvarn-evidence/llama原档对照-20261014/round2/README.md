# round2 —— 把 llama.cpp KV **允许集合内**的真实对手补进对拍（2026-10-14）

本目录是 `oracle/` 的**副本 + 扩档**（原 `oracle/` 只读、一字未改）。目的：上一轮只跟
`q4_0/q4_1/q5_0/q5_1/q8_0/q4_K` 比过，而 `common/arg.cpp:341-357` 的 `kv_cache_types` 里
**`IQ4_NL` 在集合内却没测**、3 bit 附近也没有集合内档。本轮补齐后重出两张表。

**结论一句话**：`IQ4_NL` 加进来后 **K 侧 KVarN K4 仍然赢**（赢集合内最好的 4.5 bpw 档 `iq4_nl`
`+0.00198`，real_g128 / `+0.00239`，real_g64），方向不变、边际收窄；**V 侧 `llama iq4_nl` 不旋转就赢
KVarN K4 `+0.00132`**，更坐实"V 侧不值得"。完整报告见
`../llama原档对照-20261014.md` **§11**（§10 是主会话复核，§0–§9 旧内容未动）。

## 跑法（Windows 宿主 / git-bash；**不带 `--gpus`，不占 GPU**）

```bash
docker run --rm -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/llama原档对照-20261014/round2:/w" \
  -w /w nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run_llama2.sh"   # 约 22 秒
python xcheck_numpy.py out/dump.txt      # 宿主侧 numpy 独立重写对拍（需要 numpy）
python agg_llama.py > out/agg_llama.txt  # 重出两张表
```

## 新增的 3 个档（口径逐条来自 `ref-beellama/`，只读）

| 档 | 在 KV 集合里 | block | 每片字节 | 口径出处 |
|---|---|---|---|---|
| `iq4_nl` | ✅ `common/arg.cpp:348` | 32 / 18 B | 72 | 码本 `ggml-common.h:1203-1206`；量化 `ggml-quants.c:5522-5536`→`:5389-5498`；反量化 `:3148-3164`；布局 `ggml-common.h:530-536` |
| `q3_0` | ✅ `common/arg.cpp:353` | 32 / 14 B | 56 | 量化 `ggml-quants.c:376-415`；反量化 `:854-873`；布局 `ggml-common.h:291-298` |
| `q3_1` | ✅ `common/arg.cpp:354` | 32 / 16 B | 64 | 量化 `ggml-quants.c:417-456`；反量化 `:875-895`；布局 `ggml-common.h:300-312` |

`IQ4_R3` / `IQ4_R6`：`ref-beellama/src/` 全仓 grep **零命中** ⇒ 源码里不存在，未做。

## 校验（可复现）

- `out/xcheck_numpy.txt`：**9/9 PASS**，新增 3 档的**码流 hex 逐位相同** + 反量化 `max|Δdeq| = 0.000e+00`
- `out/selftest.txt`：**失败项 = 0**（含新增档字节账 18/14/16、码本幂等 9/9、256-dim head 旋转不变性 `Δcos 1.057e-10`）
- **回归**：168 份旧档输出 + `archive_check_slice.txt` 与旧 `oracle/out/` **`cmp` 逐字节相同（169/169）** ⇒ 加档没扰动任何旧数字

## 字节账口径（重要）

本目录所有表用的都是**上游紧凑 record 口径**（`llama-kvarn.cpp:673-687` + `:690-698`）：
`record_bytes = rows*cols*bits/8 + (2*rows+cols)*2`、`rows=cols=128`、`head_slices=2`
⇒ **K4 = V4 = 70 B/token·slice = 140 B/token·head，K4V4 合计 280**。

任务里给的 `K4 = 32·4+10 = 138 / V4 = 32·4+8 = 136` 与本机可查的三套引擎口径
（紧凑 record 140/140、生产 rk4v4 136/144、ninfer P1 页自足 148/144）**都对不上**；
把 P1 的 K/V scale leading（10 / 8，源码里单位是 **FP16 元素**）当**字节**算才得到 138/136
⇒ 判为**单位漏乘 2**（`engine-g128/src/ops/kv_cache/kvarn_layout.h:62-75`）。详见报告 §11.5。

## 文件

```
llama_oracle.cpp   oracle/llama_oracle.cpp + iq4_nl/q3_0/q3_1（枚举/字节/parse/llama_qdq/自检/字节账）
run_llama2.sh      204 次运行（9 档 × 6 数据 × 2 模式 + KVarN/消融 + 归档复刻）
xcheck_numpy.py    9 个 codec 的独立 numpy 重写（含 best_index_int8）
agg_llama.py       等字节表 + 等预算排序 + 等位宽表（3-bit 桶新增、4-bit 桶加 iq4_nl）
data/              real_g128.bin / real_g64.bin（与 oracle 相同，复制）
out/               selftest / xcheck_numpy / bytes / agg_llama / agg_detail.csv / 204 份原始输出 / run_llama2.log
```

## 边界（不蒙）

- 这是**重建保真 + 字节账**两层，**PPL 一条实测都没有**；本会话已实测"逐向量 cosine 不是 PPL 的有效代理量"
  ⇒ **本目录所有数字不能当 PPL 判据**。
- `q4_K` 仍是**假想档**（不在 `kv_cache_types`）。
- 等位宽表的 "3-bit" 桶混合了 3.375 / 3.5 / 4.0 bpw，不是等字节。
