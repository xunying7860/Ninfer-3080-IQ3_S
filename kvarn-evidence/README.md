# KVarN 移植取证（原始数据与对拍件）

每个目录都是**当时跑出来的东西**，不是整理过的结论；结论请看 `docs/`。
逐文件 md5 见本目录 `checksums.txt`（共 1,024 个文件 41,673,598 B）。

| 目录 | 它在证明什么 | 关键产物 |
|---|---|---|
| `oracle/` | 量化/反量化码流的**独立实现**对拍（不依赖引擎），含正交旋转不变性检验 | 9/9 PASS、Δcos 1.057e-10；`data/real_g64.bin`、`real_g128.bin` 是从生产模型 dump 的**真实 KV tile**，不是随机数 |
| `gate2-kvarn-quant/` | 量化 kernel 与 oracle 的**逐字节**对拍（GATE2 门禁） | `out/gate_def_full.txt` |
| `wht-gate/` | Hadamard/WHT 两条实现等价性的两道 gate | `gate.cu`、`out/` |
| `offsets-g128/` | P1 槽序与偏移表（几何定案的机器可读版） | `out/p1_slot_map_g128.csv`（712 KB） |
| `verify-g128/` | 写盘 tile 的校验与逐层对照 | `data/real_k4v4*.bin/.dump` |
| `decode-g128/` | 引擎级对拍（逐 token） | `out/per_token.txt` |
| `rk4v4消融取证-20261014/` | **生产 `rk4v4` vs int8 vs bf16** 的消融（113 份日志 + 83 份 json，逐请求） | PRE/CUR/B 三臂 `*-runq2.json` |
| `kvarn128验收取证-20261014/` | kvarn128 实机验收三臂（KV128 / INT8 / RK4V4） | `ev/*-runq3.json`、101 份日志 |
| `v侧探索-20261014/` | V 侧 codec 探索，**三语料合并口径** | `results.tsv`、`x99-results.tsv`、`results_final.tsv`、`round2/` |
| `llama原档对照-20261014/` | 与 llama.cpp 原生 KV 档（`iq4_nl`/`q3_0`/`q3_1` 等）的对照，round1 + round2 | `round2/out/agg_detail.csv`（300 组）、`xcheck_numpy.txt` |

## 判据（当时写死的，不是事后挑的）

- `Δ_int8 ≤ +0.10%` = 与真 int8 打平；`Δ_rk4v4 ≤ +0.10%` = 与生产打平。
- V 侧三语料收紧为：三语料均值 ≤ +0.10% **且** 任一语料劣于 +0.30% 者出局。
- 合并口径：`NLL` 为每 token 平均 CE，三语料 token 集互不重叠 ⇒ 合并 NLL = Σ(Nᵢ·NLLᵢ)/ΣNᵢ（共 11,392 scored token）。

## 几个反复引用的数字（原始出处都在本目录）

- PPL 锚点：`baseline_fp16 7.8917`；kvarn128 `5.848024456522766` / `6.609538479127182`
- Δ_rk4v4 `−0.0325% / −0.0166%`；Δ_int8 `+0.2914% / +0.2139%`
- 生产 `rk4v4` vs int8 `+0.3240% / +0.2305%`；vs bf16 `+0.3446% / +0.3361%`；int8 vs bf16 仅 `+0.0205% / +0.1054%`
- attention 逐向量 cos：kvarn K+V4bit `0.999137`、int8-K `0.998622`、**生产 `rk4v4` `0.997040`**、kvarn_k4v2 `0.949658`
- 显存：`rk4v4` ≈ **18.6 KiB/token**、int8 ≈ **35.0 KiB**（1.89×）
- 字节口径（单一真值 `src/ops/kv_cache/kvarn_layout.h`）：紧凑 record 280 B（140/140 分侧）、
  ninfer P1 = 148/144 = **292 B**、生产 `rk4v4` = 136/144 = **280 B**

## 已知的方法学边界（别把这些数字当绝对）

- 三语料 harness 在 **0.10% 判定档上分辨力不够**：连 280 B 的 `K4V4_g128` 自己合并口径也有 +0.0894%
  （引擎侧实测同一对是 −0.0325%）。要判 0.1% 级差异，语料还得再大一个量级。
- PPL 只测到困惑度层，**没跑** MATH500 / HumanEval / AIME；逐向量 cos 高不等于 PPL 好（本轮两者反向过一次）。
