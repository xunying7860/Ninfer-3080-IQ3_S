# KVarN 移植本体 · 基线与应用方法

## 这份代码从哪来（逐字照抄取件记录）

- 上游 = `https://github.com/iamwavecut/ninfer-all`，master HEAD `11327e96c69f289aa2010a3ff91079523bbe39f5`
- 取件 = codeload tar.gz 14,862,123 B，sha256 `c5e3a9d971aebfefdfab830090cabdaff47d1455a2078a61c849bd349cdc9fb0`
- 本地 git 化副本的基线 commit = `a5ba7ba`（该副本自带 G64 旧路线的工作改动，作为参考实现保留）
- 移植的 7 个 commit：`a5ba7ba` → `034347f` → `628a6d2` → `baf93d8` → `5ba2b33` → `8186a2e` → `1ddcccd` → `9392b8e`
  （最后一个带标签 `window-lazy-physical-20261014`）
- 移植体量：**33 个文件改动，+5,263 行 / −2,155 行**

## 怎么把它复原成一支可构建的引擎

```bash
git clone https://github.com/iamwavecut/ninfer-all
cd ninfer-all && git checkout 11327e96c69f289aa2010a3ff91079523bbe39f5
git am kvarn-port/patches/0001-*.patch kvarn-port/patches/0002-*.patch ... 0007-*.patch
```

只想要 kernel 而不想要 commit 历史：`git apply --3way kvarn-port/patches/*.patch`；
只想读 KVarN 那几个新 TU：本目录 `engine/` 下按引擎内相对路径原样放着
（`src/ops/kv_cache/kvarn_layout.h` 是几何与 scale leading 的**单一真值来源**，
`src/ops/kv_cache/append/kvarn128_{codec.cuh,launch.cu,launch.h}` 是 codec/launch 本体，
`build_check/kvarn128_gate.cu`、`kvarn_tile_harness.cu`、`kvarn128_unsealed_verify.cu` 是对拍 harness）。

## 构建口径（本机的实操约束，不是上游的说法）

- 构建在 Linux 容器里做（x99 本机无 nvcc、无 cmake），生成器 ninja；**并行必须 `-j8`**，`-j12` 会打死 Docker Desktop。
- 部署与实测都在 x99（Ubuntu 22.04 / glibc 2.35 / RTX 3080 sm86），用引擎自带 loader 起，见 `run.sh` 的写法。

## 诚实登记：这支引擎哪里没验完（照抄 `9392b8e` 的登记，别当已验证件用）

- 封口 / 未封口 tile 的对拍仍 FAIL（cos=0、`|inv_rot(stage)|=0`），且 `NINFER_KVARN128_WINDOW_FULL=1` 回退路同样 FAIL
  ⇒ 更像 dump/对拍脚本口径问题，**未查到底**。
- 门禁 `ctrl-rst`、`toolsctrl` vs `tools` 对照、短提示词三档（106/51/1）**未跑**。
- `src/serve/generation_service.cpp` 改动面大（±1,473 行），**未经完整门禁**。

## 这条线的结论（为什么冻结）

一句话：**移植做完了、能跑、数值对得上，但没有经济收益 ⇒ 冻结，不再投算力。**

| 口径 | 实测 |
|---|---|
| 质量 vs 生产 `rk4v4` | Δ −0.0325% / −0.0166%（打平） |
| 质量 vs 真 int8 | Δ +0.2914% / +0.2139%（略逊） |
| 每 token 字节 | P1 几何 **292 B**（K 侧 148 / V 侧 144）vs 生产 `rk4v4` **280 B** ⇒ **+4.29%**，反而更费 |
| 速度 | 32K TTFT **+39%**、prefill **−28%** |
| 窗口 | 改前封顶 65,536；「虚拟预留 + 惰性物理后备」改后 262,144 能起（4.25 GiB 倒亏 → ~272 MiB） |
| V 侧唯一站得住的候选 | `K4Vprom_k32` 252 B、三语料合并 Δ **+0.0837%** ⇒ 记技术资产，不投算力 |

完整论证与全部数字：`docs/KVarN-KV量化结论-20261014.md`（仓库根 `docs/`）、
`docs/kvarn-移植/KVarN-收单-20261014.md`；原始取证在仓库 `kvarn-evidence/`。
