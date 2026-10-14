# llama原档对照-20261014

**目的**：把 llama.cpp 原生 KV 量化档（`q4_0/q4_1/q5_0/q5_1/q8_0/q4_K`）与 beellama KVarN
（K4V4/K4V3/K4V2/K3V3/K5V5/K6V6/K8V8）**在同一份真实数据、同一 K/V 分工、同一度量口径**下对拍，
回答"同字节下到底谁更好、好在哪一环、值不值得替换"。

**结论先看** [`../llama原档对照-20261014.md`](../llama原档对照-20261014.md)（§0 一句话判定）。

---

## 目录

```
oracle/                       本任务的完整工作区（从 ../../oracle/ 复制而来）
  llama_oracle.cpp            ★新增：6 个 llama 档 + 4 个消融档 + 256-dim head 度量 + 字节账 + 8 项自检
  bee_oracle.cpp              原样保留（未改），KVarN 侧的参照实现
  run_llama.sh                ★新增：一条命令跑完 168 次
  xcheck_numpy.py             ★新增：numpy 独立重写 6 个 llama 档，与 C++ 码流逐位对拍
  check_archive.py            ★新增：KVarN 路径保真——与旧 oracle 的 _s0_ 行逐字段比对（16/16 全等）
  agg_llama.py                ★新增：解析 out/*.txt → 等字节表 / 等位宽表 / agg_detail.csv
  data/real_g128.bin          真实 tile（只读复用，已旋转态）
  data/real_g64.bin           真实 tile（只读复用）
  ref/                        上游源码只读参照（未改）
  out/                        所有原始输出（见下表）
```

## 一条命令复现

```bash
docker run --rm \
  -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/llama原档对照-20261014/oracle:/w" \
  -w /w nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run_llama.sh"
```

**注意：不带 `--gpus`，不占任何 GPU**（本任务是纯标量重建保真 + 字节账，不需要 GPU）。
宿主侧再跑两个（需要 numpy）：

```bash
python xcheck_numpy.py out/dump.txt     # 6 档独立交叉校验
python agg_llama.py > out/agg_llama.txt # 生成两张关键表
```

## 度量口径（重要）

- **字节单位**：每 token·head-slice（128 dim）。KVarN 用 `record` 口径，llama 用 `ggml_row_size(type, 128)`。
- **误差单位**：**一条 token 的 256-dim head 向量**（= head_dim，两片合起来）。
  这是必须的：跨片 Hadamard 会在 s0/s1 之间搬运能量，**单看 128-dim 片的 cos 在旋转前后不等**，
  只有 256-dim head 才是正交变换的不变量（selftest #6 实测 `max|Δcos_glob| = 1.1e-10`）。
- 所有档一律 `stage_f16 = 0`（不做 beellama 的 F16 stage），参照取 raw 域 f32，
  这样各档看到的是**完全相同的输入数据**，误差里只有 codec 本身。
- KVarN 档的码域是 rot（已旋转态），llama 档的码域是 raw（逆旋转还原）；两者在 256-dim head 口径下可直接横比。

## 模式

| `--mode` | 含义 | 码域 | 字节 |
|---|---|---|---|
| `kvarn` | KVarN 生产口径（Sinkhorn iters=16 + 旋转） | rot | `16×bits+6` |
| `rot-only` | 消融：**只旋转不平衡**（= KVarN `iters=0`） | rot | 同上 |
| `bal-only` | 消融：**只平衡不旋转**（Sinkhorn iters=16） | raw | 同上 |
| `neither` | 消融：**都不**（纯 RTN 逐通道 affine） | raw | 同上 |
| `llama` | llama.cpp 原生档（`--lk/--lv` 选类型） | raw | `ggml_row_size` |
| `llama-rot` | llama.cpp 原生档 + **先旋转** | rot | 同上 |

## out/ 产物

| 文件 | 内容 |
|---|---|
| `selftest.txt` | 8 项自检（WHT 自逆 / 旋转往返 / unrotate 往返 / 码本幂等 / block 字节 / 旋转不变性 …），失败项 0 |
| `bytes.txt` | 字节账（KVarN record 口径 vs `ggml_row_size`），含 D256 K+V 合计 |
| `xcheck_numpy.txt` | 6 档独立交叉校验：码流 hex 全等 + 反量化零差，6/6 PASS |
| `dump.txt` | C++ 侧测试向量 / 各档码流 hex / 反量化向量（xcheck 的输入） |
| `archive_check_slice.txt` | 归档口径复刻（`SLICE` 行，与 `../../oracle/out/real_g128_k4v4.txt` 逐位相同） |
| `agg_llama.txt` | ★ 两张关键表：等字节对照 + 等预算排序 + 等位宽对照 |
| `agg_detail.csv` | 300 组 (数据,模式,档,kv) 的汇总明细 |
| `<data>_<mode>_<spec>.txt` | 168 份原始输出（逐 tile 行 + 汇总行） |

## 数据矩阵

`real_g128` / `real_g64` / `synth_gauss_g128` / `synth_gauss_g64` / `synth_outlier_g128` / `synth_outlier_g64`
（合成数据 `--seed 20261013`，分布口径与 `bee_oracle.cpp:401-418` 相同）。

## 边界（别把它当万能）

见报告 §8「未确认 / 未做」：最要紧的三条是
(1) **PPL 层一条实测都没有**，本目录只交重建保真 + 字节账；
(2) `q4_K` **不是** llama.cpp 的 KV cache 档（它是权重侧 `QK_K=256` super-block），本目录当"假想档"测；
(3) 真实 tile 的 raw 是**逆旋转还原**的（误差 3.8e-06），不是直接读 `.pt`（本地无 torch）。
