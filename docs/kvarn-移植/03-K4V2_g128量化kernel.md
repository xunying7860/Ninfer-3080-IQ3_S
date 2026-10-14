# 03 · K4V2_g128 量化 kernel（codec 通用；主档 K4V4，K4V2 为对照臂）

对照源：[`kvarn.cu:660 kvarn_quantize_tile`](https://github.com/Anbeeld/beellama.cpp/blob/main/ggml/src/ggml-cuda/kvarn.cu#L660) + [`kvarn.cu:759 kvarn_quantize_stage`](https://github.com/Anbeeld/beellama.cpp/blob/main/ggml/src/ggml-cuda/kvarn.cu#L759)

---

## 3.1 kernel 规格（照抄，不要改）

| 项 | 值 |
|---|---|
| blockDim.x | **128**（= `KVAR_N_DIM`，4 warps） |
| 动态 smem | **69,704 B**（68.1 KB） |
| 迭代数 | 8（来自 `KVAR_N_OP_PARAM_ITERS`，KVarN 已验证 8/16 无损） |
| 输入 | 一个 128×128 的 slice，fp32（由 fp16 stage 转来） |
| 输出 | 一个 record（payload + 3 个 fp16 轴） |
| 位宽 | K=4（本 slice）、V=2（本 slice）——示例档 K4V2；主档 K4V4 同一 codec，`bits` 运行期传入（`op_params[0]`） |

gridDim 建议：`(n_heads, n_groups, 1)` —— 每 block 处理「一个 head 的一个 slice 的一个 group」。

---

## 3.2 smem 布局（精确偏移）

```c
float * tile      = shared;                       // 0     .. 16383   65,536 B
float * log_s_col = tile + 16384;                 // 16384 .. 16511
float * log_s_row = log_s_col + 128;              // 16512 .. 16639
float * s_col     = log_s_row + 128;              // 16640 .. 16767
float * s_row     = s_col + 128;                  // 16768 .. 16895
float * best_col  = s_row + 128;                  // 16896 .. 17023
float * best_row  = best_col + 128;               // 17024 .. 17151
float * col_std   = best_row + 128;               // 17152 .. 17279
float * row_std   = col_std + 128;                // 17280 .. 17407
float * best_imbalance = row_std + 128;           // 17408
float * better         = best_imbalance + 1;      // 17409
float * reduce         = better + 1;              // 17410 .. 17425
// 合计 17,426 floats = 69,704 B
```

**sm_86 检查（实测）**：required 69,704 B（68.07 KB），设备 opt-in 上限 101,376 B（99.00 KB，余 31,672 B）；静态 smem 0、寄存器 30/线程 ⇒ **占用率 1 block/SM**（smem/SM 仅 102,400 B，余量不能给第二个 block）。需 `cudaFuncSetAttribute(..., 69704)`。

---

## 3.3 初始化

```c
log_s_col[threadIdx.x] = 0.0f;
log_s_row[threadIdx.x] = 0.0f;
s_col[threadIdx.x]     = 1.0f;
s_row[threadIdx.x]     = 1.0f;
best_col[threadIdx.x]  = 1.0f;
best_row[threadIdx.x]  = 1.0f;
__syncthreads();

col_std[threadIdx.x] = kvarn_std_col(tile, s_col, s_row, threadIdx.x);
row_std[threadIdx.x] = kvarn_std_row(tile, s_col, s_row, threadIdx.x);
__syncthreads();
if (threadIdx.x == 0) *best_imbalance = 3.402823466e+38F;   // FLT_MAX
__syncthreads();
kvarn_update_best_from_std(s_col, s_row, false, best_col, best_row,
                           col_std, row_std, best_imbalance, better, reduce);
```

---

## 3.4 std 计算（原文字样）

```c
// kvarn.cu:545
static __device__ float kvarn_std_col(const float * tile, const float * s_col,
                                      const float * s_row, int col) {
    float sum = 0.0f, sum_sq = 0.0f;
    const float sc = s_col[col];
    for (int row = 0; row < 128; ++row) {
        const float value = tile[row * 128 + col] / (sc * s_row[row]);
        sum += value;  sum_sq += value * value;
    }
    const float mean = sum / 128;
    return sqrtf(fmaxf((sum_sq - 128 * mean * mean) / 127, 0.0f));   // 样本标准差分母 127
}
```

`kvarn_std_row` 对称（遍历 col，分母同样 127）。

**注意分母是 `KVAR_N_DIM - 1 = 127`**（样本标准差），不是 128。写错会导致 scale 系统性偏差 `sqrt(128/127) ≈ 1.0039`。

---

## 3.5 主迭代循环（原文）

```c
for (int iter = 0; iter < iterations; ++iter) {
    const float col = fminf(fmaxf(col_std[threadIdx.x], 1e-3f), 1e3f);
    log_s_col[threadIdx.x] = fminf(fmaxf(log_s_col[threadIdx.x] + logf(col), -0.3f), 10.0f);
    s_col[threadIdx.x] = expf(log_s_col[threadIdx.x]);
    __syncthreads();

    row_std[threadIdx.x] = kvarn_std_row(tile, s_col, s_row, threadIdx.x);
    __syncthreads();

    const float row = fminf(fmaxf(row_std[threadIdx.x], 1e-3f), 1e3f);
    const float log_s_row_new = fminf(fmaxf(log_s_row[threadIdx.x] + logf(row), -0.3f), 10.0f);
    log_s_row[threadIdx.x] = log_s_row_new;
    s_row[threadIdx.x] = expf(log_s_row_new);
    __syncthreads();

    row_std[threadIdx.x] = kvarn_std_row(tile, s_col, s_row, threadIdx.x);
    col_std[threadIdx.x] = kvarn_std_col(tile, s_col, s_row, threadIdx.x);
    __syncthreads();

    kvarn_update_best_from_std(s_col, s_row, false, best_col, best_row,
                               col_std, row_std, best_imbalance, better, reduce);
}
```

**四个不能改的常量**：

| 常量 | 值 | 位置 | 作用 |
|---|---|---|---|
| std clamp | `[1e-3, 1e3]` | 迭代内 | 防止极端 tile 把 scale 推爆 |
| log clamp | `[-0.3, 10.0]` | 迭代内 | 防止 exp 溢出/下溢 |
| FLT_MAX 初值 | `3.402823466e+38F` | 初始化 | best_imbalance 起点 |
| 分母 | `127` | std | 样本标准差 |

**注意迭代内的顺序**：先更新 col → 重算 row_std → 更新 row → 重算**两个** std → 判 best。第二轮 `row_std` 被算了两次（更新后一次、判 best 前一次），这是有意的（保证 best 判定用的是最新 scale 下的 std）。

---

## 3.6 per-row RTN 量化 + 位串打包（原文，最易错）

```c
const int row = threadIdx.x;
float lo = 3.402823466e+38F;
float hi = -3.402823466e+38F;
for (int col = 0; col < 128; ++col) {
    const float x = tile[row * 128 + col] / (best_col[col] * best_row[row]);
    lo = fminf(lo, x);  hi = fmaxf(hi, x);
}

const int qmax = (1 << bits) - 1;
const float scale = fmaxf((hi - lo) / qmax, 1e-10f);
const int row_bytes = 128 * bits / 8;
uint8_t * row_payload = record + row * row_bytes;
for (int i = 0; i < row_bytes; ++i) row_payload[i] = 0;      // ← 必须先清零

for (int col = 0; col < 128; ++col) {
    const float x = tile[row * 128 + col] / (best_col[col] * best_row[row]);
    const uint8_t q = (uint8_t) fminf(fmaxf(roundf((x - lo) / scale), 0.0f), (float) qmax);
    const int bit_offset = col * bits;
    for (int bit = 0; bit < bits; ++bit) {
        const int dst_bit = bit_offset + bit;
        row_payload[dst_bit / 8] |= ((q >> bit) & 1u) << (dst_bit % 8);
    }
}
```

**三个易错点**：

1. **`row_payload` 必须先 memset 0**——位串是 `|=` 累积，不清零会留下垃圾位。
2. **`row_bytes = 128 * bits / 8`**：4-bit → 64 B/row；2-bit → 32 B/row。都是整数，4/2-bit 可用向量化快路径。
3. **`bit_offset = col * bits`**：位序与 CPU 侧 `pack_bits` 完全一致（小端、逐 bit）。

---

## 3.7 三个 fp16 轴的写法（原文）

```c
const int payload_bytes = 16384 * bits / 8;
half * scale_axis = (half *) (record + payload_bytes);
half * zp_axis    = scale_axis + 128;
half * other_axis = zp_axis + 128;

scale_axis[row] = __float2half_rn(best_row[row] * scale);   // 行因子 × 该行 scale
zp_axis[row]    = __float2half_rn(best_row[row] * lo);      // 行因子 × 该行零点
other_axis[row] = __float2half_rn(best_col[row]);           // 列因子
```

**这是理解 record 布局的关键**：

| 段 | 内容 | 个数 |
|---|---|---|
| payload | 量化码字 | rows × cols |
| scale_axis | `best_row[row] * scale[row]` | rows |
| zp_axis | `best_row[row] * lo[row]` | rows |
| other_axis | `best_col[row]`（**列因子，用 row 索引**） | rows |

`record_bytes = rows*cols*bits/8 + 3*rows*2`，但 `make_record_layout` 写的是 `(2*rows + cols)*2` —— cols 与 rows 都是 128，所以数值相同。

**K 与 V 的 rows/cols 互换**（`make_record_layout`）：

```c
rows = value ? 128 : record_dim;   // V: 128 行; K: 128（record_dim=128）
cols = value ? record_dim : 128;   // V: 128 列; K: 128
```

D256 下 K/V 都是 128×128，但**语义轴不同**：K 的 row = dim、col = token；V 的 row = token、col = dim。装载 stage 时按 `value` 分支（见 `kvarn_quantize_stage` 的 `token/dim` 计算）。

---

## 3.8 stage → tile 装载（原文）

```c
// kvarn.cu:759
for (int i = threadIdx.x; i < 16384; i += blockDim.x) {
    const int row = i / 128, col = i % 128;
    const int token = value ? row : col;
    const int dim   = value ? col : row;
    const int stage_pos = stage_base + stage_slot * 128 + token;
    tile[i] = __half2float(stage[(stage_pos * n_heads + head) * 128 + dim]);
}
__syncthreads();
// Stage and records are rotated-domain for both K and V.
kvarn_quantize_tile(record, bits, iterations, shared);
```

**stage slot 选择**（原文）：

```c
const int stage_slot = assigned_slot >= 0 ? assigned_slot :
    (swa ? (stage_group % stage_groups) : (1 + ((stage_group - 1) % tail_groups)));
```

非 SWA、`tail_groups = 2` 时 → slot 在 `{1, 2}` 之间 ping-pong。这印证 `non_swa_tail_groups = 2` 的两个 slot 是**交替使用**而非主备。

---

## 3.9 Hadamard（WHT）

`kvarn_wht_128`（[kvarn.cu:456](https://github.com/Anbeeld/beellama.cpp/blob/main/ggml/src/ggml-cuda/kvarn.cu#L456)）：

```c
__syncthreads();
for (int stride = 1; stride < 128; stride *= 2) {
    if (threadIdx.x < 64) {
        const int j = (threadIdx.x / stride) * (2 * stride) + (threadIdx.x % stride);
        ...
    }
    __syncthreads();
}
```

另有 `kvarn_wht_cross_slices`（kvarn.cu:510）处理 `head_slices = 2/4` 的跨 slice 混合——**实测判定：D256 不必实现**。beellama `wht_128 + cross_slices` 与 ninfer `normalized_hadamard_d256_inplace` 结构逐位相同（变体 A' 3,145,728 元素 max|Δ|=0），配对恒等、无符号翻转；照搬 beellama 只会多 2 次乘并引入 1.7e-08 常量误差，无收益。

**ninfer 侧决策（已定）**：比对已完成——二者除上游常量 0.57 ulp（3.423e-08，施加在最后一级蝶形之前）外**逐位相同** ⇒ **复用 ninfer 的 `normalized_hadamard_d256_inplace`，不补 cross-slice。**

---

## 3.10 low-shmem 回退（sm_86 其实不需要，但留着）

若实测 smem 不足或寄存器压力过大，`kvarn_quantize_stage_lowshmem`（[kvarn.cu:871](https://github.com/Anbeeld/beellama.cpp/blob/main/ggml/src/ggml-cuda/kvarn.cu#L871)）只占 **3.1 KB**，代价是每次 std 都重读 global：

```
8 iter × (2×row_std + 1×col_std) × 128 元素 = 3,072 次 global 读/线程
```

可作为兜底路径，也可用于调试（beellama 提供 `GGML_KVARN_TEST_FORCE_LOWSHMEM` 环境变量强制走这条）。

---

## 3.11 本阶段 Gate

- [ ] 口径更正：标量 oracle 只做**结构基准**——`iters=0` 比字节（576/576 PASS）；`iters=16` 比 `maxd|s_col|`/解码 cosine。上游 ggml-cuda 真用 `-use_fast_math`，连 `iters=0` 都 371/576 record 差 1,017 字节 ⇒ **标量 oracle 不能当逐字节基准**。
- [ ] kernel 输出 record 与标量 oracle：`iters=0` 逐字节（payload + 3 轴全比，576/576 PASS）；`iters=16` 改比 `maxd|s_col|`（合成 ~7.5e-3 / 真实 ~3.3e-6）与解码 cosine。
- [ ] 4-bit 与 2-bit 两路都验（K4V2 各占一路；**主档 K4V4 = 两路皆 4-bit**）
- [ ] `cudaFuncGetAttribute` 确认 smem ≤ 99 KB、无 local spill
- [ ] 与 beellama `tests/test-kvarn.cpp` 的判定口径对齐
