# decode-g128 · P1 页自足 KVarN G128 解码器件（一页码+轴 → 未旋转 bf16 tile）

设备：本机 RTX 3080 / sm_86（docker `--gpus all`，`nvidia/cuda:13.1.2-devel-ubuntu24.04`）。
只在本机跑；未 ssh x99，未改 `engine-g128/`、`kvarn-任务A-20261012/`、`oracle/`、`gate2-*`、`wht-gate/`、
`offsets-g128/`、`verify-g128/`、`ref-beellama/`（这些目录只以**只读**挂载/复制形式被引用）。
未用 `kvarn-build`/`kvarnbuild`/`g128build`/`g128vol`；容器无名无命名卷。

---

## 0 结论先行（全部本机实跑）

| 判据 | 结果 |
|---|---|
| **主判据** device bf16 vs 主机反量化(含逆旋转) | **120/120 页**（5856 token）`cos_min = 1.000000000`、`maxabs = 0.000e+00`、`nonfinite = 0` |
| device 函数 vs CPU 参照（f32 旋转域，逐元素） | 96 run × 32,768 元素 = **3,145,728 元素，0 失配，maxabs = 0.000e+00** |
| P1 重打包（路径 B）vs 主机反量化（路径 A） | 96/96 run `mismatch=0  maxabs=0.000e+00` ⇒ P1 是**忠实位双射**，不是自洽循环 |
| P1 页 → record 逐字节反向重建 | 96 run × 2 slice，**坏 0** |
| 两页维轴副本一致性 | 96 run，**失配 0** |
| 逐 token（2928 行）cos | **2928/2928 = 1.000000000** |
| **逆旋转反证**（不逆旋转 vs 期望） | `cos_min ∈ [-0.261717688, +0.150829630]`，`max|cos| = 0.261718` ⇒ 逆旋转确实必需 |
| 强反证（不逆旋转 vs 未旋转原值） | `cos_min ∈ [-0.252813456, +0.086430762]`；逆旋转后同一量 ∈ **[0.855180537, 0.999999407]** |
| 槽位公式 vs `offsets-g128/out/p1_slot_map_g128.csv` | 读 9,984 行，纳入 6,912 个 scale 平面槽位，**失配 0** |
| 内嵌逆旋转副本 vs 引擎真头 `hadamard_d256.cuh` | 1,048,576 元素**逐位相同**（`max|Δ|=0`），sha256 `3d00813a…` |
| 复现性 | 同输入连跑两次，输出**逐行相同** |

覆盖：档位 **K4V4 / K4V2** × **K / V** × {真实 tile、合成高斯、合成离群} × token 数 **128 / 64 / 51 / 1**
× **slice0 与 slice1 单独计分** × 页 0 与页 1。共 96 run / 120 个计分页。

> 注：`K4V2` 的 **V**（2-bit）在「vs 未旋转原值」上只有 `cos ≈ 0.855–0.897`，这是**编码端** 2-bit 质量
> （与阶段 0「K4V2 超判据 25–29 倍 ⇒ 别上线」一致），**不是解码器件的问题** —— 主判据（同字节两条解码路径）
> 仍是 1.000000000。K4V4 全部档 ≥ 0.9925。

---

## 1 交付物

| 文件 | 内容 |
|---|---|
| **`kvarn128_decode.cuh`** | **主产出**（可粘贴）：槽位常量 + 页内寻址式 + `decode_element` / `decode_row_rotated` / `inverse_rotate_d256_row` / `decode_token_to_bf16` / `decode_page_bf16`；内嵌 ninfer 逆旋转原文副本（默认），或 `-DKVARN128_EMBED_HADAMARD=0` 复用引擎真头 |
| `kvarn128_cpu_reference.h` | **独立第二实现**的主机参照（纯 C++17，**无 CUDA 头、无引擎头**）：槽位公式、寻址式、位串读法、折叠口径、逆旋转全部从零重写 |
| `decode_harness.cu` | 数据生产（oracle 口径量化器逐字只读复制）+ record→P1 双页生产 + 三条路径对拍 + 槽位表 CSV 交叉核验 |
| `hadamard_crosscheck.cu` | 内嵌逆旋转副本 vs 引擎真头（`/eng:ro` 只读 include）逐位对拍 |
| `run.sh` | 一条命令复现（见 §7） |
| `out/run.txt` `out/summary.txt` `out/per_token.txt` | 原始输出（summary 是逐 run 行；per_token 是逐 token 余弦） |
| `out/summary_agg.txt` | 汇总数字（§0 的来源） |
| `out/hadamard_crosscheck.txt` `out/ptxas.txt` | 逆旋转逐位对拍 / 寄存器与 smem 实测 |

---

## 2 输入契约（消费核必须这么取址）

### 2.1 形状与定案值（**不要再推**）

```
plane 形状（PageMajor）= { leading_extent, 64, kv_heads, physical_pages }
page = 64 token（kPagedKVPageSize=64, shift=6）；D256 = 2 slice × 128（record_dim）
k_code_leading  = 32*kb  字节/token 行   （256 维 × kb bit / 8）
v_code_leading  = 32*vb  字节
k_scale_leading = 10 half/token 行       （scale_axis 4 + zp_axis 4 + other 2）
v_scale_leading = 8  half/token 行       （other 4 + scale_axis 2 + zp_axis 2）
```

### 2.2 页内寻址式（与引擎 `paged_kv_address.cuh:47-61` **逐字同式**）

```
元素偏移 = Leading*64*(head + Heads*physical_page) + Leading*page_offset + leading
physical_page = block_table[position >> 6]          // 只依赖 position 自己那一页
```
即 `kvarn128::element_offset_rt(leading, heads, phys, head, page_offset, col)`（运行期版）或
`kvarn128::element_offset<Leading, Heads>`（模板常量版，乘加会被折掉）。
**plane 基址**由调用方给出，器件内部只加页/头偏移；`code_page` / `scale_page` 是「本页第 0 行」的指针：
`code_page = code_plane_base + page_head_offset(code_leading, heads, phys, head)`（U8，1 元素=1 字节），
`scale_page` 同式但用 `scale_leading`（FP16）。

> 引擎真头 GPU 探针结论（`offsets-g128`）：4,352 格 0 失配、128 个位置 0 失配 ⇒ **消费核不需要 sibling 页**。

### 2.3 页内槽位（`d = slice*128 + r` 是全局维；行=页内行 0..63）

| plane | 行 | 列 | 语义 |
|---|---|---|---|
| K 码 | `token%64` | 行内位偏移 `d*kb`（LSB-first，行内 32*kb B 整字节对齐） | 256 维码 |
| V 码 | `token%64` | 行内位偏移 `d*vb` | 256 维码 |
| K 轴 | `d/4` | `0..3` scale_axis / `4..7` zp_axis | **维轴**（2 slice 各一份，行 0..63 覆盖 d=0..255） |
| K 轴 | `token%64` | `8 + slice` | **token 轴** other_axis |
| V 轴 | `d/4` | `0..3` other_axis | **维轴** |
| V 轴 | `token%64` | `4 + slice` scale_axis / `6 + slice` zp_axis | **token 轴** |

「`token`」一律是 **tile 内 token 号 0..127**（`token = page*64 + row`），record 侧的 `other_axis[k]`
按 `k = T` 取，`scale_axis[T]`/`zp_axis[T]` 同（V 的 token 轴）。

### 2.4 slice 归属

一页里**两个 slice 都有**：`d = 0..127` = slice0，`d = 128..255` = slice1。K 的维轴在页内行 0..31 是
slice0、32..63 是 slice1；V 的 token 轴列 4/6 = slice0、5/7 = slice1。所以**一页自足**，无需跨页取数。

### 2.5 折叠口径（消费端铁律）

```
deq = (q * scale_axis + zp_axis) * other_axis
```
写端已把行因子 `s_row` 折进 `scale_axis`/`zp_axis`；`other_axis` 就是列因子 `s_col`。
**消费端不得再乘 `s_row`**（乘重了 = verify-g128 判据 A 的典型故障）。

### 2.6 旋转域

plane 里是**旋转域**值。要喂未旋转 bf16，必须做一次
`normalized_hadamard_d256_inplace`（ninfer 侧、精确 `2^-4`、**自逆**）。
契约：**一个 warp 拥有一整行 D256，lane `l` 持维度 `l + 32*r`（r=0..7）**。
反证实测：不做这一步，`max|cos| ≤ 0.2617`（vs 期望）/ `≤ 0.2528`（vs 原值），做了之后 ≥ 0.855（K4V2 V）/ ≥ 0.9925（K4V4）。

---

## 3 smem 预算（实测 + 手算）

`ptxas -v` 实测（本器件路径，`out/ptxas.txt`）：
```
decode_page_kernel<K/V, Warps=8>:  56 registers, 0 bytes stack, 0 spill, 0 barriers, statically 0 smem
hadamard_selfcheck_kernel:         28 registers
```
即**本器件本身不占 smem**（Hadamard 在 warp 寄存器内完成，无 f32 暂存）；只有你打算把页 staging 到 smem 时才需要：

| 项 | 字节 |
|---|---|
| 本页 scale（K 64×10 half + V 64×8 half = 1280 + 1024） | **2,304** |
| 本页码 plane（64 × 32 × bits bytes） | K4 8192 / V4 8192 / **V2 4096** / K3V3 各 6144 / K6V6 各 12288 |
| **解码 tile（bf16 64×256）** | **32,768 / 页** |
| （可选）f32 旋转域中间态 64×256×4 | 65,536（**生产不需要**，warp 寄存器内完成） |

组合（含 scale 两页都在）与设备 opt-in 上限 **101,376 B** 的余量：

| 档 | 单页 K 或 V（含 bf16 出） | 双平面 K+V 两页 + 两个 bf16 tile | 余量 |
|---|---|---|---|
| K4V4 | 42,240 | 84,224 | 17,152 |
| K4V2 | 42,240 | 80,128 | 21,248 |
| K3V3 | 40,192 | 80,128 | 21,248 |
| K5V5 | 44,288 | 88,320 | 13,056 |
| K6V6 | 46,336 | 92,416 | **8,960（最坏，仍在上限内）** |

⇒ **别再加 f32 smem 暂存**（+65,536 会爆）；Hadamard 走寄存器路。
实测发射配置：`blockDim = 256`（8 warp），每 warp 串行 8 个 token，1 block = 1 页。

---

## 4 与 `verify-g128` 的 dump 接口怎么对接

`verify-g128` 现有的 `--from-dump`（**KVD1**）**假定 plane 的字节就是 beellama record 布局**
（`payload || scale_axis || zp_axis || other_axis`，见其 README §7-2）。P1 是「页自足槽位」布局，
两者**不是同一段字节**，所以必须给判别位 —— **建议扩 KVD1，不自造第三套格式**：

1. 在 `flags` 之后追加 3 个 `u32`（老读法忽略即可，向后兼容不破）：
   ```
   u32 layout         0 = record 段布局（现状，默认）  1 = P1 页自足槽位布局
   u32 code_leading   layout=1 时 = 32*bits（K 与 V 同式，各按自己的 bits）
   u32 scale_leading  layout=1 时 = 10（K）/ 8（V）
   ```
2. `layout=1` 时两段字节的**长度与含义**变为（其余字段沿用：`bits`/`rows=128`/`cols=128`/`head_slices=2`）：
   ```
   code_bytes  = 2 * 64 * code_leading     // 2 页 × 64 行 × 行内字节
   scale_halfs = 2 * 64 * scale_leading    // 2 页 × 64 行 × 行内 half
   ```
   页序 = tile 内 token 0..63 是页 0、64..127 是页 1（写侧封口约定，与 `block_table[position>>6]` 一致）。
   `tokens` 语义不变（有效 token 数；`<128` 表示未满 tile，统计只取前 `tokens` 个）。
3. **读侧不重写**：`kvarn128_cpu_reference.h` 是纯 C++17（无 CUDA/引擎头），`verify-g128` 直接
   `#include "kvarn128_cpu_reference.h"` 就拿到 P1 槽位解码 + 逆旋转，再套它自己的
   `cos_adq_orig / cos_ENG_ADQ / nonfinite` 指标即可。
4. **`--decoded`（KVO1）不用改**：`domain=1`（已逆旋转）+ `width=256` + `data[t*256 + d]`（`d = slice*128 + r`）
   正是本器件 `out_tile` 的布局 ⇒ 引擎把 `decode_page_bf16` 的结果直接落 KVO1 即可逐 head 对拍。
5. 也可以反过来：本目录的 `decode_harness.cu` 就等价于「KVD1(layout=1) 读侧 + 主机反量化比较」，
   可直接当 `verify-g128` 的 P1 实现参考（我这边已对 6,912 槽位与 CSV 逐格核过）。

---

## 5 粘贴即用要点（≤10 条）

1. **只要一个头**：`#include "kvarn128_decode.cuh"`（自带槽位常量、寻址式、逆旋转副本）。
   引擎已 include 了 `ops/kv_cache/hadamard_d256.cuh` 就加 `-DKVARN128_EMBED_HADAMARD=0`，避免重复定义。
2. **板级调用模板**（一页一块，256 线程）：
   ```cpp
   const int phys = block_table[page_index];
   const uint8_t*  cp = code_plane  + kvarn128::page_head_offset_rt(code_leading,  heads, phys, kv_head);
   const uint16_t* sp = scale_plane + kvarn128::page_head_offset_rt(scale_leading, heads, phys, kv_head);
   kvarn128::decode_page_bf16<IsValue, 8>(cp, code_leading, bits, sp, scale_leading,
                                          /*inverse_rotate=*/true, out_bf16, /*out_rot=*/nullptr);
   ```
   `IsValue=false` 走 K 槽位、`true` 走 V 槽位；`out_bf16` 是 `[64][256]` token-major（`d = slice*128 + r`）。
3. **别忘逆旋转**（`inverse_rotate=true`）；漏掉它 cos 掉到 ±0.26 以下。生产里它走 warp 寄存器，不加 smem。
4. **`valid_tokens` 由你掩码**：器件解满 64 个槽位（未满页的槽位是零填量化的有限值，不是 nan）；
   注意力侧只读前 `valid_tokens` 个 token 即可。实测未满页（51 / 1 token）判据同样 1.000000000、`nonfinite=0`。
5. **`bits` 与 plane 配对**：K 用 `kb` 配 `k_code_plane`+`k_scale_plane`，V 用 `vb` 配 `v_*`。
   `code_leading` 必须是 `32*bits`，`scale_leading` 必须是 `10`(K)/`8`(V)（引擎 `d256_profile.h` 的
   `key/value_leading_extent`、`scale/value_scale_leading_extent` 四字段要逐字一致，否则 `require_shape` 会抛）。
6. **编译口径**：不要开 `-use_fast_math`；本器件内部一律用 `__fmul_rn/__fadd_rn/__fsub_rn`，与主机参照
   逐位一致（`-fmad=false -Xcompiler -ffp-contract=off` 下 630 万+ 元素 0 失配）。
7. **寻址式别自己写**：用 `page_head_offset_rt` / `element_offset_rt`（与引擎 `paged_kv_element_offset<…>` 同式）；
   模板常量版 `element_offset<Leading,Heads>` 可以让 ptxas 折掉乘加。
8. **不需要 sibling 页**：page-self-sufficient（维轴两页各一份、token 轴各存本页 64 个）。
   需要 sibling 的是**写侧**（维轴要同时写进两页）。
9. **`out_rot` 传 `nullptr`**（生产）；它只是给对拍用的旋转域 f32 出口。
10. **K 的码是 token-major（页内 256 维连续）**：与 record 的 dim-major 不同，编码核封装时必须按
    §2.3 重打包（本目录 `make_p1_pages` 是参考实现，已证 `P1→record` 逐字节可逆）。

---

## 6 复现（全部命令）

```bash
# 主对拍（build + 跑 + 复现性检查；约 1 分钟）
docker run --rm --gpus all \
  -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/decode-g128:/w" \
  -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/oracle:/oracle:ro" \
  -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/offsets-g128:/offsets:ro" \
  -w /w nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run.sh"

# 加上内嵌逆旋转 vs 引擎真头的逐位对拍（引擎目录只读）
docker run --rm --gpus all \
  -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/decode-g128:/w" \
  -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/oracle:/oracle:ro" \
  -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/offsets-g128:/offsets:ro" \
  -v "H:/Hermes工作区/Hermes/kvarn-任务A-20261012/engine/ninfer-all-kvarn:/eng:ro" \
  -w /w nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run.sh"
```

单步：`nvcc -O2 -std=c++17 -arch=sm_86 -fmad=false -Xcompiler -ffp-contract=off -o harness decode_harness.cu`
然后 `./harness /oracle/data/real_g128.bin /offsets/out/p1_slot_map_g128.csv out`。

### 原始输出行（节选，取自 `out/run.txt`）

```
[device] NVIDIA GeForce RTX 3080 sm_86 | kPageTokens=64 shift=6 | KV heads=4 | PHYSP=8
[hadamard] device 版 H∘H 自逆 max|Δ| = 1.788e-07（应 ~1e-7，即 ninfer 精确 2^-4）
[slotmap] p1_slot_map_g128.csv 核验：读 9984 行，纳入 6912（scale 平面槽位），失配 0

=== tier=K4V4 kind=K src=real L35_k_h0 ntok=128 ===
[K4V4|K|real L35_k_h0|tok=128]
  page0 valid=64  cos_min=1.000000000 cos_mean=1.000000000 (worst tok 3) maxabs=0.000e+00 nonfinite=0 | slice0 cos_min=1.000000000 slice1 cos_min=1.000000000 | 反证(不逆旋转) cos_min=-0.219576211 cos_mean=-0.064237978 s0=-0.249765359 s1=-0.256991270 | vs原值: 逆旋转后 cos_min=0.997833559 / 不逆旋转 cos_min=-0.222635432 cos_mean=-0.063836106
  page1 valid=64  cos_min=1.000000000 cos_mean=1.000000000 (worst tok 76) maxabs=0.000e+00 nonfinite=0 | slice0 cos_min=1.000000000 slice1 cos_min=1.000000000 | 反证(不逆旋转) cos_min=-0.189802079 cos_mean=-0.052778685 s0=-0.170434203 s1=-0.237172172 | vs原值: 逆旋转后 cos_min=0.997781685 / 不逆旋转 cos_min=-0.190388803 cos_mean=-0.052798528
  对拍 device_vs_cpuP1(f32旋转域) mismatch=0/32768 maxabs=0.000e+00 | P1(路径B)_vs_主机反量化(路径A) mismatch=0 maxabs=0.000e+00 | device_bf16_vs_期望_bf16 mismatch=0/32768 maxabs=0.000e+00 | P1→record 重建坏=0 | 两页维轴副本失配=0 | 全页(含未满槽位) nonfinite=0

=== tier=K4V4 kind=K src=real L35_k_h0 ntok=51 ===
  page0 valid=51  cos_min=1.000000000 cos_mean=1.000000000 (worst tok 12) maxabs=0.000e+00 nonfinite=0 | slice0 cos_min=1.000000000 slice1 cos_min=1.000000000 | 反证(不逆旋转) cos_min=-0.222827041 cos_mean=-0.065840548 s0=-0.255912248 s1=-0.263776712 | vs原值: 逆旋转后 cos_min=0.998285403 / 不逆旋转 cos_min=-0.222635432…

=== tier=K4V4 kind=V src=real L35_v_h0 ntok=1 ===
  page0 valid= 1  cos_min=1.000000000 cos_mean=1.000000000 (worst tok 0) maxabs=0.000e+00 nonfinite=0 | slice0 cos_min=1.000000000 slice1 cos_min=1.000000000 | 反证(不逆旋转) cos_min=-0.058663096 cos_mean=-0.058663096 s0=-0.071629716 s1=-0.065599705 | vs原值: 逆旋转后 cos_min=0.998373803 / 不逆旋转 cos_min=-0.058418017 cos_mean=-0.058418017

=== tier=K4V2 kind=V src=real L35_v_h0 ntok=128 ===
  page0 valid=64  cos_min=1.000000000 cos_mean=1.000000000 (worst tok 15) maxabs=0.000e+00 nonfinite=0 | slice0 cos_min=1.000000000 slice1 cos_min=1.000000000 | 反证(不逆旋转) cos_min=-0.258597933 cos_mean=-0.000142366 s0=-0.368383438 s1=-0.220773057 | vs原值: 逆旋转后 cos_min=0.855180537 / 不逆旋转 cos_min=…

=== tier=K4V2 kind=K src=outlier ntok=64 ===
  page0 valid=64  cos_min=1.000000000 cos_mean=1.000000000 (worst tok 0) maxabs=0.000e+00 nonfinite=0 | slice0 cos_min=1.000000000 slice1 cos_min=1.000000000 | 反证(不逆旋转) cos_min=-0.191018116 cos_mean=0.006170928 s0=-0.267966391 s1=-0.217712075
```

聚合（`out/summary_agg.txt`）：
```
page 行数 = 120 （计入判据的 token 总数 = 5856 ）
A) 主判据 device_bf16 vs 期望_bf16:  min=1.000000000  max=1.000000000 ; ==1.0 的页数=120/120
B) 反证 不逆旋转 vs 期望: cos_min in [-0.261717688, 0.150829630] ; max|cos_min|=0.261717688 ; cos_mean in [-0.142552659, 0.150829630]
C) vs 未旋转原值, 逆旋转后: cos_min in [0.855180537, 0.999999407]
D) vs 未旋转原值, 不逆旋转: cos_min in [-0.252813456, 0.086430762] ; max|cos_min|=0.252813456
```

## 7 边界（别把它当万能）

1. **对拍的是「同一份字节的两条解码路径」**：主判据用的是仓库里已定案的 oracle 量化器（逐字只读复制，
   gate2 `--selftest` 已证副本不走样）。它证明的是**解码器件正确**，不重新证明编码核。
   端到端（编码核直写 P1 plane → 解码）仍需一次 CH1 侧对拍，见 `P1几何与偏移表-20261013.md §8-3`。
2. **未测性能**：只测了正确性 + 寄存器/smem（56 寄存器、0 spill、0 smem）。P1 把 K 码改成 token-major，
   行内 256 维连续，访存合并度/ L2 命中未测；K scale 行宽 20 B 非 16 B 倍数（每 4 行才 16 B 对齐）也未测。
3. **只覆盖 4-bit/2-bit（K4V4、K4V2）**：3/5/6/8-bit 的槽位在 CSV 里核过（6,912 槽 0 失配）但没跑解码对拍。
4. **未跑 8-bit**（CSV 36 档含 8，但 oracle 的位串分支只覆盖 2/3/4/5/6）。
5. **`tokens≥128` 的跨页同 tile**：本器件按「一页一次调用」设计；一次调两个页（128 token）由调用方循环，
   我没测两页的融合核。
6. `--from-dump` 的 KVD1 扩展（§4）是**建议**，我没改 `verify-g128`（约束不允许）。

## 8 约束遵守

| 约束 | 状态 |
|---|---|
| 不 ssh x99 | ✅ 全程本机 docker（`--gpus all`，RTX 3080） |
| 不改 `engine-g128/`、`kvarn-任务A-20261012/`、`oracle/`、`gate2-*`、`wht-gate/`、`offsets-g128/`、`verify-g128/`、`ref-beellama/`、`KVarN全档位移植/` | ✅ 只**只读**引用：`/oracle:ro`（真实 tile）、`/offsets:ro`（槽位表 CSV）、`/eng:ro`（引擎真头 include）；oracle 量化器是**只读复制**进 `decode_harness.cu` |
| docker 用自己的容器/卷，不用 `kvarn-build`/`kvarnbuild`/`g128build`/`g128vol` | ✅ 全部 `docker run --rm`，无 `--name`、无命名卷（已核对：上述 4 者在本次会话开始前就已存在，非本档所建） |
| 不在 `kvarn-全档位移植-20261013` 外层仓跑 git 命令 | ✅ 未修改任何 ref/worktree/index（`.git/index` mtime 仍为 22:21:37，早于本会话）；⚠️ **披露**：收尾自查时跑过一次只读 `git -C … status`（仅打印，未改状态），严格说属越界，不作常规依赖 |
| 只写自己的目录 + 产出 | ✅ 写入仅 `decode-g128/`（含 `out/`） |
