# 06 · ninfer 侧接线清单

引擎源：https://github.com/iamwavecut/ninfer-all —— **pin 方向需更正**：`60f18c3`(0.11.0, 2026-09-26) 是 `11327e96`(0.12.0, 2026-10-12) 的**直系祖先、落后 599 commit**（compare 60f18c3...11327e96: ahead_by 599 / behind_by 0）；仓库默认分支是 `master`（无 `main`，codeload refs/heads/main 404），HEAD `e7b7e669…`。建议 pin `11327e96`。
部署：https://github.com/xunying7860/Ninfer-3080-IQ3_S

---

## 6.0 落地口径与已定路线（2026-10-13 用户已定）

**两套数字（别把 record 口径当 ninfer 的实际占用）：**

| 口径 | 闭式 | K4V2 | K4V3 | K3V3 | K4V4 |
|---|---|---:|---:|---:|---:|
| record 口径（上游 arena / 方案包字节账） | `32·(Kb+Vb)+24` | 216 | 248 | 216 | **280**（= 生产 rk4v4） |
| **ninfer P1 落地口径**（页自足，页 = 64 token） | `32·(Kb+Vb)+36` | 228 | 260 | 228 | **292** |

⇒ **ninfer 里主档 K4V4 是 292 B/token，不是 280**（对生产 280 **+4.29%**；327,680 ctx × 4 KV head ⇒ **365.0 MiB vs 350.0 MiB**）。P1 恒定比 record 多 12 B/token（维轴条目在 tile 两页各存一份，`P1几何与偏移表-20261013.md` §6）。

**路线：**

1. 先按 (A) **P1 页自足**把路径跑通（消费核不需要 sibling 页），主档 **K4V4**；`d256_profile.h` 四个 leading 取值 = `key=32·kb`、`value=32·vb`、`key_scale=10`、`value_scale=8`（K4V4 ⇒ **128 / 128 / 10 / 8**）。
2. **"与生产同字节（280）"只有走 blob 紧凑存储**（tile 一条 record 布局、不按 64-token 页自足）**才成立**，**属后续可选项**。
3. 紧接着做**全档位 36 组合**（K/V 独立 2/3/4/5/6/8-bit）：CLI `--kvarn-k-bits / --kvarn-v-bits`；plane leading **直接用 P1 CSV 表**（`offsets-g128/out/offsets_p1_g128.csv`，36 档 × 4 平面），**不重新推**。36 档 P1 值 = record 矩阵每格 + 12。

---

## 6.1 改动文件总表

| # | 文件 | 改动 | 风险 |
|---|---|---|---|
| 1 | `include/ninfer/types.h` | `enum class KvCacheStorage` **末尾**追加 `KvarnTile128` | 低（插中间会破坏既有编号） |
| 2 | `src/ops/kv_cache/d256_profile.h` | 加 case：code=`DType::U8`、`quant_group=128`、scale=FP16 | **高**：字段是 `key/value_leading_extent`、`scale/value_scale_leading_extent`（**无单数 `leading_extent`**）；switch **无 default，不加 case 直接 throw** |
| 3 | `src/ops/kv_cache/plane_types.h` | **五个** `KvPlaneStorage` 特化 + `assert_kv_planes`（constexpr 模板函数） | 中 |
| 4 | **新增** `src/ops/kv_cache/kvarn_layout.h` | `make_layout` / `make_record_layout` / `packed_bytes` constexpr | 低（单一真值来源） |
| 5 | **新增** `src/ops/kv_cache/kvarn_quantize.cuh` | `kvarn_quantize_tile` / `_stage` / `_lowshmem` | 高（核心） |
| 6 | **新增** `src/ops/kv_cache/kvarn_wht.cuh` | WHT 128（**不补 cross-slice**）或直接复用 ninfer 的 `normalized_hadamard_d256_inplace` | 中（**已实测等价**：除上游常量 0.57 ulp 外逐位相同，3,145,728 元素 max\|Δ\|=0） |
| 7 | **新增** `src/ops/kv_cache/append/kvarn_launch.cu` | 三路分派 A/B/C | 高 |
| 8 | `src/core/paged_kv_cache.cpp` | 1 tile = 2 pages，成对分配 | 中 |
| 9 | **`src/ops/kernel/paged_kv_window.cuh` 在 `60f18c3` 不存在**（旧基线独有） | 128 token 共享一条 record 头；须**重新定位**窗口内核（`src/ops/kernel/` 无同名文件）或改 `src/ops/kv_cache/append/*` | 高 |
| 10 | `src/ops/kernel/paged_kv_address.cuh` | A/C 用 `table_rows`，不用 `valid_columns` | **高**（当前卡点） |
| 11 | `apps/cli/options.cpp` | `--kvarn-k-bits` / `--kvarn-v-bits` | 低 |
| 12 | `src/serve/serve_options.cpp` | 同上 | 低 |
| 13 | `src/serve/operational_log.cpp` | dtype 名称表同步 | 低但**易漏**（你踩过 unknown） |
| 14 | **`src/calibration/route_catalog.cpp` 在 `60f18c3` 不存在**；落点 = `src/ops/common/device_route.{h,cpp}` + `src/runtime/engine/device_profile.{h,cpp}` + `src/runtime/engine/device_profiles.json` | 加 `kvarn.*` 路由键 | 低（digest 变→旧 profile 失效） |
| 15 | `src/ops/kv_cache/CMakeLists.txt` 或 `sources.cmake` | **显式列举**新增 .cu | 低但**易漏**（无 GLOB） |
| 16 | **`src/core/paged_kv_storage.h`**（表**漏列**） | `paged_kv_storage_layout()` 加 KVarN 的 `{storage, head_dim, {U8,…}, {U8,…}}` 分支 | 中（不加直接 throw） |

---

## 6.2 关键接线点详解

### `KVAR_N` 常量集中定义

```cpp
// src/ops/kv_cache/kvarn_layout.h
namespace ninfer::kvarn {
constexpr uint32_t kGroup      = 128;   // KVAR_N_GROUP
constexpr uint32_t kDim        = 128;   // KVAR_N_DIM (record_dim)
constexpr uint32_t kTileValues = kDim * kDim;
constexpr uint32_t kReduce     = 16;

constexpr size_t kSharedFloats   = kTileValues + 8 * kDim + 2 + kReduce;  // 17,426 → 68.1 KB
constexpr size_t kLowShmemFloats = 6 * kDim + 2 + kReduce;                // 786 → 3.1 KB

constexpr bool valid_bits(int b) { return b==2||b==3||b==4||b==5||b==6||b==8; }

constexpr size_t packed_bytes(size_t n_values, int bits) {
    return (n_values * size_t(bits) + 7) / 8;
}
}  // namespace
```

### 动态 smem opt-in（必须）

```cpp
cudaFuncSetAttribute(kvarn_store_kernel<4,2>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     ninfer::kvarn::kSharedFloats * sizeof(float));  // 69,704
```

sm_86 的 `smpbo` ≈ 99 KB → 69,704 装得下。**不加这行会启动失败（请求超过静态上限）。**

实测（GATE2 §5）：required 69,704 B、opt-in 上限 101,376 B、静态 smem 0、寄存器 30/线程 ⇒ **占用率 1 block/SM**（smem/SM 102,400 B，**不能并发 2 block**；128 线程/SM，上限 1536）。

### `d256_profile.h` 的 `leading_extent` 问题

现有结构假设"按 D 展开"（per-token 四平面）。tile record 下（注：字段名是 `key_leading_extent`/`value_leading_extent`/`scale_leading_extent`/`value_scale_leading_extent`，无单数 `leading_extent`；结构体确无 `tile_bytes`/`tokens_per_tile`）：

- 一个 tile = 128 token × 2 slice × K/V，**不是按 D 连续展开**
- `leading_extent` 语义（现结构是 `key/value_leading_extent` 四个字段）要么改成 tile_bytes，要么扩展结构

**建议**：扩展 `D256KVCacheProfile` 增加 `tile_bytes` 与 `tokens_per_tile` 字段，而不是硬塞。硬塞会与现有八档的格式统一性断裂（见 `08` 风险）。

### `paged_kv_address.cuh` 的 Masked 感知（当前卡点）

```cpp
// 实测更正：paged_kv_address.cuh 两基线逐字节相同，其中没有这个「== nullptr → 拒绝」分支
// valid_tokens() 只在 if constexpr (Masked) 里读 valid_columns[0]；那条拒绝是旧移植自己写进 append/launch.cu 的
if (Masked && metadata.valid_columns == nullptr) → 拒绝
if (!Masked) → 用 metadata.table_rows 定位，不检查 valid_columns
```

并且 append 的三路分派里，**B 路径（组中间写 stage）完全不碰 metadata**。

---

## 6.3 部署侧（Ninfer-3080-IQ3_S）

### run.sh 改动

```bash
# 2026-10-13 已定路线（见 §6.0）：主档 K4V4 —— ninfer P1 落地口径 292 B/token/KV head（比生产 280 多 4.29%）
#   容量：327680 ctx, 4 head: 365.0 MiB（P1）vs 350.0 MiB（生产）；"与生产同字节（280）"只有走 blob 紧凑存储才成立，属后续可选项
# 2026-10-xx KV 换档 rk4v4 → 对照臂 kvarn-k4v2（G128，record 口径 216 B/token/KV head；P1 口径 228）
#   动机：对照臂省 22.9% KV 显存（327680 ctx, 4 head: 270.0 MiB vs 350.0）
#   质量：<填实测 PPL / 三针>
#   tile=128 ⇒ 1 tile = 2 pages；staging 2 slots（回滚保险）
#   回退点：/data/ninfer/run.sh.bak-202610xx-kvarn
```

```diff
-  --max-context 327680 --kv-capacity 327680 --kv-dtype rk4v4 \
+  --max-context 327680 --kv-capacity 327680 --kv-dtype kvarn --kvarn-k-bits 4 --kvarn-v-bits 4 \
+  # 对照臂（已判死，仅对撞用）：--kvarn-k-bits 4 --kvarn-v-bits 2
```

**同时同步线上已改未入库的参数**：`--device-state-slots 1`。

### 三项必须回归的部署件

| 项 | 原因 |
|---|---|
| **自研补丁重打** | 重编会覆盖 `bundle/bin/ninfer-serve`（重启不重算 1.2s / strata 落盘 / tg 口径） |
| **看门狗阈值** | tile flush 是每 128 步一次的周期突发，改变功耗/利用率曲线；现有判据 `util≥90% 且 功耗<130W 且 计数不推进 ≥180s` 需复核 |
| **监控面板解析** | 日志行 `rk4v4` → `kvarn` 字面量变化，若正则硬编码会**静默显示 `–`** |

### 构建

沿用现有流程（Win Docker 编 sm_86，自带 glibc 打包到 `bundle/libs/`）：

```bash
git checkout 11327e96   # 60f18c3 是旧树（落后 599 commit），先与主控确认 pin
# 先重打自研补丁，再叠 kvarn 改动
cmake -S . -B build-sm86 -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DNINFER_BUILD_APPS=ON -DNINFER_MULTICALL=ON \
  -DNINFER_PTXAS_VERBOSE=ON \      # 必须：确认 smem ≤99KB、无 spill
  -DBUILD_TESTING=OFF
cmake --build build-sm86 --parallel 4
cp bundle/bin/ninfer-serve bundle/bin/ninfer-serve.bak-kvarn-pre
```

---

## 6.4 顺序依赖

**先修会话持久化，再换 KVarN。** 换 kvarn 会让磁盘层 tag 变、旧页全 miss，两个变量混在一起无法归因。

---

## 6.5 本阶段 Gate

- [ ] `--kv-dtype kvarn --kvarn-k-bits 4 --kvarn-v-bits 4` 能启动（对照臂再各跑一次 `--kvarn-v-bits 2`）
- [ ] `capacity | … kvarn …` 不报 unknown
- [ ] `--kv-capacity auto` 算出的容量与 **P1 落地口径**对得上（K4V4 292 B/token ⇒ 365.0 MiB / 4 head；`02` §2.4 是 record 口径，差 12 B/token）
- [ ] ptxas 无 local spill，smem ≤ 99 KB（实测 required 69,704 B / 上限 101,376 B，**占用率 1 block/SM**、128 线程/SM）
- [ ] 三个自研补丁功能复现
