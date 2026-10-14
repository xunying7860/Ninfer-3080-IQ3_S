# 参考 · beellama.cpp 逐行核对与 layout 抠取报告

生成时间：2026-10-13 19:37 (+0800)
核对人：子代理（PC）
用途：作为 KVarN 全档位移植的**唯一源码依据**。所有摘录均为真实源码原文（已核对行号），凡未找到的均明确标注。

---

## 0. 拉取方式与版本

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/Anbeeld/beellama.cpp |
| **commit hash** | **`58a1629274d8d74ea5a6acf34f30fb5946c2422b`** |
| commit 时间 | 2026-09-30T11:00:05Z |
| commit 摘要 | `Preserve CUDA build cache after Windows taskkill races` |
| 拉取时间 | 2026-10-13 19:37:35 +0800 |
| 拉取方式 | **codeload tarball**（不是 git clone） |
| 落地目录 | `H:\Hermes工作区\Hermes\kvarn-全档位移植-20261013\ref-beellama\src\`（解包目录原为 `beellama.cpp-main/`，已重命名为 `src/`） |

**为什么用 tarball**：本机 git 走 Windows schannel，`git clone https://github.com/...` 直接失败：

```
fatal: unable to access 'https://github.com/Anbeeld/beellama.cpp.git/':
schannel: next InitializeSecurityContext failed: CRYPT_E_NO_REVOCATION_CHECK (0x80092012)
```

`-c http.schannelCheckRevoke=false` 无效（git 仍报同一错误）；curl 8.18.0（同为 schannel 后端）也失败。可用通道是 **`curl --ssl-no-revoke`**，据此：

- commit hash 取自 `https://api.github.com/repos/Anbeeld/beellama.cpp/commits/main`
- 源码包取自 `https://codeload.github.com/Anbeeld/beellama.cpp/tar.gz/refs/heads/main`（38,059,749 B）

⚠️ **注意**：tarball 落地后**没有 `.git` 目录**，本目录内跑不了任何 git 命令（也正好符合"不在本仓跑 git"的约束）。若后续要 diff 版本，需重新下载或改用 `GIT_SSL_NO_VERIFY`/自建 CA 的 git。

**已实际读取并摘录的源文件**（相对 `ref-beellama\src\`）：

```
src/llama-kvarn.h
src/llama-kvarn.cpp
src/llama-kv-cache-kvarn.h
src/llama-kv-cache-kvarn.cpp
ggml/src/ggml-cuda/kvarn.cu
ggml/src/ggml-cuda/fattn-mma-kvarn-load.cuh
ggml/src/ggml-cuda/fattn-mma-kvarn.cuh
docs/beellama-features.md
tests/test-kvarn.cpp
```

---

## 1. doc 01 逐条核对结果

约定：✅ = 与真实源码一致；❌ = 对不上，给真实行号。

### 1.1 布局与几何（doc 01 §1）

| doc 01 条目 | doc 01 行号 | 真实行号 | 结论 |
|---|---|---|---|
| `KVAR_N_GROUP = 128` | `llama-kvarn.h:13` | **`llama-kvarn.h:10`** | ❌ |
| `llama_kvarn_tile_layout` 结构 | `llama-kvarn.h:71` | **`llama-kvarn.h:78`** | ❌ |
| `llama_kvarn_record_layout` 结构 | `llama-kvarn.h:57` | **`llama-kvarn.h:66`** | ❌ |
| `make_layout` | `llama-kvarn.cpp:634` | `llama-kvarn.cpp:634` | ✅ |
| `make_record_layout` | `llama-kvarn.cpp:673` | `llama-kvarn.cpp:673` | ✅ |
| `geometry_for` | `llama-kvarn.cpp:698` | **`llama-kvarn.cpp:690`**（`case 256` 在 **694**） | ❌ |
| `packed_bytes` | `llama-kvarn.cpp:709` | `llama-kvarn.cpp:709` | ✅ |
| `pack_bits` | `llama-kvarn.cpp:715` | `llama-kvarn.cpp:715` | ✅ |
| `unpack_bits_value` | `llama-kvarn.cpp:733` | **`llama-kvarn.cpp:735`** | ❌ |
| `hadamard_64` | `llama-kvarn.cpp:~750` | `llama-kvarn.cpp:750` | ✅ |
| （doc 01 漏列）`hadamard_128` | — | `llama-kvarn.cpp:770` | 补 |

另外：doc 01 §1 里 `llama-kvarn.h:57` 那个位置实际是 `llama_kvarn_type_desc`（51 行），`llama-kvarn.h:71` 实际是 `llama_kvarn_record_layout` 的成员。doc 01 把三个结构的行号整体上移/错位了。

**doc 01 的 `make_layout`"原文"块不是逐字原文**，是压缩改写版：真实源码带 4 条 `assert`、每段一行一语句、结尾 `return layout;`。语义等价但**不能当逐字依据**。真实原文见 §3.1。

**doc 01 的 `geometry_for`"原文"块是逐字原文**（真实行 692–695），仅缺 `default: geometry = {}; return false;` 两行。

### 1.2 CUDA 量化核心（doc 01 §2）

| doc 01 条目 | doc 01 行号 | 真实行号 | 结论 |
|---|---|---|---|
| `KVAR_N_DIM = 128` | `kvarn.cu:12` | `kvarn.cu:12` | ✅ |
| `KVAR_N_SHARED_FLOATS` | `kvarn.cu:16` | `kvarn.cu:16` | ✅ |
| `KVAR_N_LOWSHMEM_FLOATS` | `kvarn.cu:18` | `kvarn.cu:18` | ✅ |
| op_params 索引常量 | `kvarn.cu:20–30` | 真实 **21–35**（0..10 全在） | ❌（范围） |
| `ggml_cuda_kvarn_valid_bits` | `kvarn.cu:447` | **`kvarn.cu:444`** | ❌ |
| `required_shared_bytes` | `kvarn.cu:448` | `kvarn.cu:448` | ✅ |
| `low_shared_bytes` | `kvarn.cu:452` | `kvarn.cu:452` | ✅ |
| `kvarn_wht_128` | `kvarn.cu:456` | `kvarn.cu:456` | ✅ |
| `kvarn_std_col` | `kvarn.cu:545` | `kvarn.cu:545` | ✅ |
| `kvarn_std_row` | `kvarn.cu:562` | `kvarn.cu:562` | ✅ |
| `kvarn_reduce_std_ranges` | `kvarn.cu:593` | `kvarn.cu:593` | ✅ |
| `kvarn_update_best_from_std` | `kvarn.cu:624` | `kvarn.cu:624` | ✅ |
| `kvarn_quantize_tile` | `kvarn.cu:660` | `kvarn.cu:660` | ✅ |
| `kvarn_quantize_stage` | `kvarn.cu:759` | `kvarn.cu:759` | ✅ |
| `kvarn_std_col_lowshmem` | `kvarn.cu:813` | `kvarn.cu:813` | ✅ |
| `kvarn_quantize_stage_lowshmem` | `kvarn.cu:871` | `kvarn.cu:871` | ✅ |
| `kvarn_store_kernel_hishmem` | `kvarn.cu:993` | `kvarn.cu:993` | ✅ |
| `kvarn_store_kernel_headwide` | `kvarn.cu:1059` | `kvarn.cu:1059` | ✅ |
| `kvarn_store_kernel_lowshmem` | `kvarn.cu:1145` | `kvarn.cu:1145` | ✅ |
| `ggml_cuda_op_kvarn_store` | `kvarn.cu:2207` | `kvarn.cu:2207` | ✅ |
| `kvarn_materialize_live_kernel` | `kvarn.cu:2780` | `kvarn.cu:2780` | ✅ |
| `ggml_cuda_op_kvarn_materialize` | `kvarn.cu:3088` | `kvarn.cu:3088` | ✅ |
| `GGML_KVARN_TEST_FORCE_LOWSHMEM` | `kvarn.cu:~2260` | **`kvarn.cu:2248`** | ❌（近似值偏大 12） |
| （doc 01 漏列）`kvarn_std_row_lowshmem` | — | `kvarn.cu:842` | 补 |
| （doc 01 漏列）`KVAR_N_STAGE_CHUNK = 4` | — | `kvarn.cu:20` | 补 |
| （doc 01 漏列）`KVAR_N_REDUCE_FLOATS = 4*4` | — | `kvarn.cu:15` | 补 |

### 1.3 op_params 编码表（doc 01 §2 表）

doc 01 表基本正确，但 idx 1/2 的标注有串位。真实定义（`kvarn.cu:21–35`）：

```
21: KVAR_N_OP_PARAM_BITS          = 0
22: KVAR_N_OP_PARAM_ITERS         = 1
23: KVAR_N_OP_PARAM_MAT_VALUE     = 1     // ← 与 ITERS 同槽（mat 用），doc 01 串到 idx2 了
24: KVAR_N_OP_PARAM_STORE_VALUE   = 2
25: KVAR_N_OP_PARAM_MAT_STREAM_START = 2  // ← 与 STORE_VALUE 同槽（mat 用）
26: KVAR_N_OP_PARAM_MAT_N_STREAM  = 3
27: KVAR_N_OP_PARAM_TOKENS_PER_STREAM = 3 // ← 与 MAT_N_STREAM 同槽
28: KVAR_N_OP_PARAM_MAT_EMIT_ROTATED = 4
29: KVAR_N_OP_PARAM_STORE_SWA     = 4
30: KVAR_N_OP_PARAM_HEAD_SLICES   = 5
31: KVAR_N_OP_PARAM_MAT_SWA       = 6
32: KVAR_N_OP_PARAM_STAGE_GROUPS  = 7
33: KVAR_N_OP_PARAM_TAIL_GROUPS   = 8
34: KVAR_N_OP_PARAM_EAGER_RECORDS = 9
35: KVAR_N_OP_PARAM_READ_INDIRECT = 10
```

即 **mat 有一条 MAT_VALUE=1，store 有一条 STORE_VALUE=2**，doc 01 写"idx 2 = STORE_VALUE / MAT_VALUE"把 MAT_VALUE 写成 2 是错的（MAT_VALUE 是 1）。doc 01 表里 idx 6（MAT_SWA）也漏了。

### 1.4 tail 与 workspace（doc 01 §3）

| doc 01 条目 | doc 01 行号 | 真实行号 | 结论 |
|---|---|---|---|
| `llama_kvarn_tail_policy_for` | `llama-kv-cache-kvarn.h:21–45` | **函数在 29–49**；结构体 `llama_kvarn_tail_policy` 在 21–27 | ❌ |
| `llama_kvarn_workspace_groups`（=1） | `llama-kv-cache-kvarn.h:47` | **53–56**（`return 1;` 在 55） | ❌ |
| `llama_kvarn_non_swa_tail_groups`（=2） | `llama-kv-cache-kvarn.h:51` | **58–82**（`return 2;` 在 81） | ❌ |

- `intrinsic = min(group_size, effective_window)` → 真实 **37** 行 ✅（原文一致）
- `effective_tokens = max(intrinsic, explicit_tokens)` → 真实 **41** 行 ✅（原文一致）
- doc 01 引的 "Two slots keep the reopened group's own F16 source resident …" 注释 → 真实 **64–71** 行 ✅（逐字基本一致，doc 01 省略了中间两句）
- doc 01 引的 "Perplexity is unchanged between depth 1 and 2 (7.1898 either way, Qwen3.6-27B kvarn6 at 4K)." → 真实 **78** 行 ✅ **逐字一致**
- doc 01 引的 `llama_kvarn_can_remove_range` 承诺句 → 真实 **64–65** 行 ✅

### 1.5 WHT（doc 01 §4）

| doc 01 条目 | 真实 | 结论 |
|---|---|---|
| `ggml_cuda_op_kvarn_wht` 声明 `kvarn-wht.cuh` | 文件存在 | ✅ |
| `ggml_cuda_op_kvarn_wht` 实现 `kvarn-wht.cu` | 文件存在 | ✅ |
| `kvarn_wht_128`（device 内）`kvarn.cu:456` | `kvarn.cu:456` | ✅ |
| `kvarn_wht_cross_slices` `kvarn.cu:510` | `kvarn.cu:510` | ✅ |
| （doc 01 漏列）`kvarn_wht_64_lane` / `kvarn_wht_128_lane` | `kvarn.cu:474` / `492` | 补 |

### 1.6 文档与档位阶梯（doc 01 §5）

doc 01 引的 `beellama-features.md` 阶梯原文 ✅ **逐字一致**（真实 `docs/beellama-features.md:26–30`）：

> The general benchmark ladder starts with `kvarn5 / kvarn4` and a 1024-token tail as its balanced default, with `kvarn6 / kvarn5` as the more conservative body and `kvarn4 / kvarn3` as the smallest recommended tier.

（doc 01 漏了紧随其后的限定句："These presets come from Qwen 3.6 27B measurements, not a universal model guarantee."）

### 1.7 测试文件（doc 01 §6）

doc 01 列的 5 个测试文件**全部存在**（`tests/test-kvarn.cpp`、`test-kvarn-d64-parallel-static.py`、`test-kvarn-rollback-static.py`、`test-kvarn-long-context-checkpoint.py`、`test-kvarn-mtp-sharing-static.py`）。另本 commit 还多出 `test-kvarn-eager-workspace-static.py`、`test-kvarn-fit-fail-closed-static.py`、`test-kvarn-hip-*.py`、`test-kvarn-port-safety-static.py`、`test-qwen4exp-qsa-kvarn-static.py`、`test-dflash-kvarn-*.py`。

### 1.8 ⚠️ 最重要的一条：doc 00/02 的字节账口径是错的（差 6 B/token）

doc 02 §2.1 自己发现了 768 B 的差并留了"必须查清"的行动项。**查清了**：

- `llama_kvarn_make_layout(head_dim, group, kb, vb)`（doc 00/01/02 全部字节表的来源）**在引擎里根本没有调用者**——全树 grep 只在 `src/llama-kvarn.cpp:634`（定义）、`src/llama-kvarn.h:137`（声明）和 **`tests/test-kvarn.cpp:680/919/966`（且都用 `head_dim=128`）**出现。
- 真正决定分配的是 `src/llama-kv-cache-kvarn.cpp:303`：

```cpp
size_t kvarn_record_bytes(int record_dim, int bits, bool value) {
    return llama_kvarn_make_record_layout(record_dim, bits, value).record_bytes;
}
```

用于 `src/llama-kv-cache-kvarn.cpp:1482–1483`，配 `:1499–1500` 的张量尺寸：

```cpp
const size_t k_record_size = kvarn_record_bytes(k_record_dim, this->params.key_bits, false);
const size_t v_record_size = kvarn_record_bytes(v_record_dim, this->params.value_bits, true);
...
auto * k_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, k_record_size, n_head_k_sliced, n_record_groups);
auto * v_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, v_record_size, n_head_v_sliced, n_record_groups);
```

其中 `n_head_k_sliced = n_head_kv * k_slices`（`:1497`）、`k_record_dim = k_geometry.record_dim`（`:1480`）。D256 ⇒ `record_dim = 128`、`slices = 2`（`geometry_for` case 256）。

**所以 D256 的真实占用是「每 slice 一个 128×128 record」，两套口径：**

| 口径 | 公式（B / 128-token group / KV head） | K4V2 值 | 每 token |
|---|---|---|---|
| `make_layout`（head_dim=256，**只被测试用**） | `4096*(kb+vb) + 2304` | 26,880 | `32*(kb+vb)+18` = **210** |
| **真实分配（record × 2 slice）** | `4096*(kb+vb) + 3072` | **27,648** | `32*(kb+vb)+24` = **216** |

差 **768 B/group = 6 B/token**，逐档恒定。doc 00/02 的 36 组合表（146/178/210/…/530）**每格都要 +6**：

| K\V | 2 | 3 | 4 | 5 | 6 | 8 |
|---:|---:|---:|---:|---:|---:|---:|
| **2** | 152 | 184 | **216** | 248 | 280 | 344 |
| **3** | 184 | 216 | 248 | 280 | 312 | 376 |
| **4** | **216** | 248 | 280 | 312 | 344 | 408 |
| **5** | 248 | 280 | 312 | 344 | 376 | 440 |
| **6** | 280 | 312 | 344 | 376 | 408 | 472 |
| **8** | 344 | 376 | 408 | 440 | 472 | 536 |

（用 36 组合全量对拍：doc 00/02 表 = `make_layout` 口径 **逐个吻合**；真实 record 口径 = doc 表 **逐个 +6**。）

容量换算修正（327,680 ctx，n_kv_heads=4）：K4V2 = **270.0 MiB**（doc 写 262.5），对生产 rk4v4(280) 省 **80.0 MiB**（−22.9%，doc 写 −25%）。n_kv_heads=2 ⇒ 135.0 MiB。

**移植结论**：容量预算以 `make_record_layout`（每 slice）为唯一真值；`make_layout` 不要用（它是 D128 及以下的等价体，D256 起会低估）。

---

## 2. 三条 doc 00 前置认知的复核

### ① smem 够用、tile 常驻从不修改 —— ✅ 成立

`kvarn.cu:15–18`：

```cpp
static constexpr int KVAR_N_REDUCE_FLOATS = 4 * 4;
static constexpr int KVAR_N_SHARED_FLOATS = KVAR_N_TILE_VALUES + 8 * KVAR_N_DIM + 2 + KVAR_N_REDUCE_FLOATS;
static constexpr int KVAR_N_SHARED_BYTES = KVAR_N_SHARED_FLOATS * sizeof(float);
static constexpr int KVAR_N_LOWSHMEM_FLOATS = 6 * KVAR_N_DIM + 2 + KVAR_N_REDUCE_FLOATS;
```

= `16384 + 1024 + 2 + 16 = 17,426` floats = **69,704 B = 68.07 KB** ✅ 与 doc 的 17426/68.1KB 一致。
lowshmem = `6*128 + 2 + 16 = 786` floats = **3,144 B = 3.07 KB** ✅。

tile 常驻、每迭代从原始 tile 重算：`kvarn.cu:545–560`（`kvarn_std_col`）与 `:562–577`（`kvarn_std_row`）每轮都除一次 `tile[...]/(sc*s_row[row])`；`kvarn_quantize_tile` 里 `tile` 只在 `kvarn_quantize_stage`（`:779–786`）写入一次，之后全程只读。doc 00 引的 `kvarn.cu:554` ✅ 行号正确、原文逐字正确。

**但 doc 00/01 漏了一个关键分支**：D256 在目标卡（sm_86，smpbo≈99KB ≥ 68.07KB）走的是 **headwide** 内核，不是 hishmem（`kvarn.cu:2536`，详见 §5 细节 2）。

### ② 量化是 per-row RTN —— ✅ 成立（并给出更精确的归属）

`kvarn.cu:660–757` 真实逻辑：

- `lo/hi` 是**每个 row 在自己 128 列上的 min/max**（`:723–730`），量化域是 `x = tile[row*128+col] / (best_col[col] * best_row[row])`（`:727`）
- `scale = max((hi-lo)/qmax, 1e-10)`（`:733`）
- 三段轴：`scale_axis[row] = best_row[row]*scale`（`:753`）、`zp_axis[row] = best_row[row]*lo`（`:754`）、`other_axis[row] = best_col[row]`（`:755`）

doc 00 的对应片段 ✅ 正确（只是它把 `other_axis[row] = best_col[row]` 注释成"列因子存在这里"，实际写入下标是 `row`、读的是 `best_col[row]`，即整列因子数组，语义对）。

注意 `scale`/`zp` 只由 `best_row[row]` 吸收，而 `best_col[col]` 落到 `other_axis` 并被**解码端按列乘回**（`fattn-mma-kvarn-load.cuh:223–224`）：`(q*scale_axis[row] + zp_axis[row]) * other_axis[col]`。✅

### ③ 线程固定 128（4 warps）—— ✅ 成立

- `kvarn_quantize_tile` 全程 `threadIdx.x` 即 row 索引（`:678–687`、`:723`），即 blockDim=128
- 启动配置证据：`kvarn.cu:2514/2550/2650/2735/2758` 全部 `<<<..., KVAR_N_DIM, ...>>>` = 128 线程
- reduce：`kvarn_warp_min/max` 用 `__shfl_down_sync` 32 路（`:579–591`），`kvarn_reduce_std_ranges` 用 `reduce[warp*4 + k]` 四 warp 汇总（`:593–622`），`reduce` 数组正好 `4*4=16` floats ✅

---

## 3. layout / 打包位序 / 迭代 —— 关键原文摘录

### 3.1 CPU 侧 `make_layout`（`src/llama-kvarn.cpp:634–671`，逐字）

```cpp
llama_kvarn_tile_layout llama_kvarn_make_layout(int head_dim, int group, int key_bits, int value_bits) {
    assert(head_dim > 0);
    assert(group > 0);
    assert(llama_kvarn_valid_bits(key_bits));
    assert(llama_kvarn_valid_bits(value_bits));

    llama_kvarn_tile_layout layout = {};
    size_t off = 0;

    layout.k_payload_off = off;
    layout.k_payload_bytes = llama_kvarn_packed_bytes(head_dim * group, key_bits);
    off += layout.k_payload_bytes;

    layout.k_s_col_off = off;
    off += size_t(head_dim) * sizeof(uint16_t);

    layout.k_zp_off = off;
    off += size_t(head_dim) * sizeof(uint16_t);

    layout.k_s_row_off = off;
    off += size_t(group) * sizeof(uint16_t);

    layout.v_payload_off = off;
    layout.v_payload_bytes = llama_kvarn_packed_bytes(group * head_dim, value_bits);
    off += layout.v_payload_bytes;

    layout.v_s_col_off = off;
    off += size_t(head_dim) * sizeof(uint16_t);

    layout.v_s_row_off = off;
    off += size_t(group) * sizeof(uint16_t);

    layout.v_zp_off = off;
    off += size_t(group) * sizeof(uint16_t);

    layout.tile_bytes = llama_kvarn_align_up(off, 8);
    return layout;
}
```

### 3.2 CPU 侧 `make_record_layout`（kernel 真实口径，`src/llama-kvarn.cpp:673–688`，逐字）

```cpp
llama_kvarn_record_layout llama_kvarn_make_record_layout(int record_dim, int bits, bool value) {
    assert(record_dim == 64 || record_dim == 128);
    assert(llama_kvarn_valid_bits(bits));

    llama_kvarn_record_layout layout = {};
    layout.token_group = KVAR_N_GROUP;
    layout.record_dim = uint32_t(record_dim);
    layout.rows = value ? KVAR_N_GROUP : uint32_t(record_dim);
    layout.cols = value ? uint32_t(record_dim) : KVAR_N_GROUP;
    layout.payload_bytes = llama_kvarn_packed_bytes(int(layout.rows * layout.cols), bits);
    layout.scale_off = layout.payload_bytes;
    layout.zp_off = layout.scale_off + size_t(layout.rows) * sizeof(uint16_t);
    layout.other_off = layout.zp_off + size_t(layout.rows) * sizeof(uint16_t);
    layout.record_bytes = layout.other_off + size_t(layout.cols) * sizeof(uint16_t);
    return layout;
}
```

### 3.3 `geometry_for`（`src/llama-kvarn.cpp:690–698`，逐字）

```cpp
bool llama_kvarn_geometry_for(int head_dim, llama_kvarn_geometry & geometry) {
    switch (head_dim) {
        case 64:  geometry = { KVAR_N_GROUP,  64,  64, 1 }; return true;
        case 128: geometry = { KVAR_N_GROUP, 128, 128, 1 }; return true;
        case 256: geometry = { KVAR_N_GROUP, 128, 256, 2 }; return true;
        case 512: geometry = { KVAR_N_GROUP, 128, 512, 4 }; return true;
        default:  geometry = {}; return false;
    }
}
```
（字段序：`{token_group, record_dim, head_dim, head_slices}` —— 见 `llama-kvarn.h:59–64`。**D256 ⇒ record_dim=128, head_slices=2** ✅ 与 doc 01 一致。）

### 3.4 打包位序（CPU 侧，`src/llama-kvarn.cpp:709–748`，逐字）

```cpp
size_t llama_kvarn_packed_bytes(int n_values, int bits) {
    assert(n_values >= 0);
    assert(llama_kvarn_valid_bits(bits));
    return (size_t(n_values) * size_t(bits) + 7) / 8;
}

void llama_kvarn_pack_bits(const uint8_t * values, int n_values, int bits, uint8_t * dst) {
    ...
    const uint8_t mask = uint8_t((1u << bits) - 1u);
    for (int i = 0; i < n_values; ++i) {
        const uint8_t value = values[i] & mask;
        const size_t bit_offset = size_t(i) * size_t(bits);

        for (int bit = 0; bit < bits; ++bit) {
            const size_t dst_bit = bit_offset + size_t(bit);
            dst[dst_bit / 8] |= uint8_t(((value >> bit) & 1u) << (dst_bit % 8));
        }
    }
}

uint8_t llama_kvarn_unpack_bits_value(const uint8_t * src, int index, int bits) {
    ...
    uint8_t value = 0;
    const size_t bit_offset = size_t(index) * size_t(bits);
    for (int bit = 0; bit < bits; ++bit) {
        const size_t src_bit = bit_offset + size_t(bit);
        value |= uint8_t(((src[src_bit / 8] >> (src_bit % 8)) & 1u) << bit);
    }
    return value;
}
```

**位序结论**：全局**连续比特流**，元素 `i` 占 `[i*bits, i*bits+bits)` 位；**位内 LSB-first**（`value>>bit` 对应 `dst_bit%8 = bit`）；也就是"一个元素的低位在低地址位"。**没有字节内高位在前**。

### 3.5 CUDA 侧行内打包（逐字，`kvarn.cu:732–755`）

```cpp
    const int qmax = (1 << bits) - 1;
    const float scale = fmaxf((hi - lo) / qmax, 1e-10f);
    const int row_bytes = KVAR_N_DIM * bits / 8;
    uint8_t * row_payload = record + row * row_bytes;
    for (int i = 0; i < row_bytes; ++i) {
        row_payload[i] = 0;
    }
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        const float x = tile[row * KVAR_N_DIM + col] / (best_col[col] * best_row[row]);
        const uint8_t q = (uint8_t) fminf(fmaxf(roundf((x - lo) / scale), 0.0f), (float) qmax);
        const int bit_offset = col * bits;
        for (int bit = 0; bit < bits; ++bit) {
            const int dst_bit = bit_offset + bit;
            row_payload[dst_bit / 8] |= ((q >> bit) & 1u) << (dst_bit % 8);
        }
    }

    const int payload_bytes = KVAR_N_TILE_VALUES * bits / 8;
    half * scale_axis = (half *) (record + payload_bytes);
    half * zp_axis = scale_axis + KVAR_N_DIM;
    half * other_axis = zp_axis + KVAR_N_DIM;
    scale_axis[row] = __float2half_rn(best_row[row] * scale);
    zp_axis[row] = __float2half_rn(best_row[row] * lo);
    other_axis[row] = __float2half_rn(best_col[row]);
```

**record 字节序（CUDA = CPU record_layout，D256 下 `rows == cols == 128`）**：

```
[0, payload_bytes)                量化码流（row-major，每行 row_bytes = 128*bits/8 独立字节对齐）
[payload_bytes, +128*2)           scale_axis[128]  fp16  ← best_row[row]*scale
[+128*2,      +128*2)             zp_axis[128]     fp16  ← best_row[row]*lo
[+128*2,      +128*2)             other_axis[128]  fp16  ← best_col[col]
record_bytes = payload_bytes + 768
```

**重要**：`128*bits/8 = 16*bits` 对所有合法位宽都是整数，**所以 3/5/6-bit 每行仍是整字节边界，不会跨行溢出**。这是非字节对齐位宽能按行独立打包的前提。

### 3.6 解码端位序（读侧真值，`fattn-mma-kvarn-load.cuh:5–50`，逐字）

```cpp
static __device__ __forceinline__ uint8_t ggml_cuda_fattn_kvarn_unpack_record(
        const uint8_t * record, const int index, const int bits) {
    if (bits == 8) {
        return record[index];
    }
    if (bits == 4) {
        const uint8_t packed = record[index >> 1];
        return (packed >> ((index & 1) << 2)) & 0x0fu;
    }
    if (bits == 2) {
        const uint8_t packed = record[index >> 2];
        return (packed >> ((index & 3) << 1)) & 0x03u;
    }
    const int bit_offset = index * bits;
    const int byte_offset = bit_offset >> 3;
    const int bit_in_byte = bit_offset & 7;
    const uint16_t packed = (uint16_t) record[byte_offset] | ((uint16_t) record[byte_offset + 1] << 8);
    return (packed >> bit_in_byte) & ((1u << bits) - 1u);
}
```

4-bit：`value[2i] = 低半字节`、`value[2i+1] = 高半字节`。2-bit：`value[4i+k] = (byte >> 2k) & 3`。**LSB-first 与写侧严格互逆** ✅。3/5/6-bit 走 16-bit 窗口 `record[bo] | record[bo+1]<<8` 再右移 `bit_in_byte`，即**跨字节小端读**。

### 3.7 量化主循环（per-row RTN + best 选择，`kvarn.cu:723–757` 已在 §3.5 上半给出，此处补 best 侧）

`kvarn.cu:698–721`（迭代体）：

```cpp
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

        kvarn_update_best_from_std(
                s_col, s_row, false,
                best_col, best_row, col_std, row_std,
                best_imbalance, better, reduce);
    }
```

**默认迭代数 = 16**（`src/llama-kvarn.cpp:198`、`common/common.h:358`：`.sinkhorn_iters = 16`）。

### 3.8 smem 偏移表（`kvarn.cu:665–676`，逐字）

```cpp
    float * tile = shared;
    float * log_s_col = tile + KVAR_N_TILE_VALUES;
    float * log_s_row = log_s_col + KVAR_N_DIM;
    float * s_col = log_s_row + KVAR_N_DIM;
    float * s_row = s_col + KVAR_N_DIM;
    float * best_col = s_row + KVAR_N_DIM;
    float * best_row = best_col + KVAR_N_DIM;
    float * col_std = best_row + KVAR_N_DIM;
    float * row_std = col_std + KVAR_N_DIM;
    float * best_imbalance = row_std + KVAR_N_DIM;
    float * better = best_imbalance + 1;
    float * reduce = better + 1;
```
共 `16384 + 8*128 + 2 + 16` ✅。

### 3.9 tile ← stage 的载入与 K/V 转置（`kvarn.cu:779–786`，逐字）

```cpp
    for (int i = threadIdx.x; i < KVAR_N_TILE_VALUES; i += blockDim.x) {
        const int row = i / KVAR_N_DIM;
        const int col = i % KVAR_N_DIM;
        const int token = value ? row : col;
        const int dim = value ? col : row;
        const int stage_pos = stage_base + stage_slot * KVAR_N_DIM + token;
        tile[i] = __half2float(stage[(stage_pos * n_heads + head) * KVAR_N_DIM + dim]);
    }
```

**⇒ K（value=false）tile[row=dim][col=token]（转置）；V（value=true）tile[row=token][col=dim]。** 读侧 `fattn-mma-kvarn-load.cuh:219–220` 的 `row = value?pos:dim; col = value?dim:pos` 与之严格对应。

---

## 4. doc 01 未覆盖、但移植必须知道的原文

### 4.1 D256 的旋转 = 128-WHT + 跨 slice 蝶形（不是朴素 256-WHT 调用）

`kvarn.cu:1113–1121`（headwide store kernel 逐字）：

```cpp
        float values[4] = {};
        for (int slice = 0; slice < head_slices; ++slice) {
            const int head = head0 + slice;
            shared[dim] = current[((int64_t) token * n_heads + head) * KVAR_N_DIM + dim];
            kvarn_wht_128(shared);
            values[slice] = shared[dim];
        }

        kvarn_wht_cross_slices(values, head_slices);
```

`kvarn.cu:510–543`（逐字）：

```cpp
static __device__ void kvarn_wht_cross_slices(float values[4], int head_slices) {
    if (head_slices == 1) {
        return;
    }
    for (int stride = 1; stride < head_slices; stride <<= 1) {
        for (int base = 0; base < 4; base += 2 * stride) {
            if (base + stride >= head_slices) { continue; }
            for (int i = 0; i < stride; ++i) {
                const int a_idx = base + i;
                const int b_idx = base + stride + i;
                if (b_idx >= head_slices) { continue; }
                const float a = values[a_idx];
                const float b = values[b_idx];
                values[a_idx] = a + b;
                values[b_idx] = a - b;
            }
        }
    }
    const float scale = head_slices == 2 ? 0.7071067811865475f : 0.5f;
    for (int slice = 0; slice < 4; ++slice) {
        if (slice < head_slices) { values[slice] *= scale; }
    }
}
```

**数值等价结论**：`wht_128`（归一化 1/√128）作用在每个 128-slice，再对 slice 做 2 点蝶形 ×1/√2 ⇒ 复合 = **归一化 256 点 FWHT**（Sylvester 序，最后一段 stride=128 恰好配对 i 与 i+128）。即 beellama 的 D256 旋转与 ninfer 的 `normalized_hadamard_d256_inplace` **在数学上应当等价**，但**必须用标量 oracle 逐元素验**（slice0↔index[0..127]、slice1↔index[128..255] 的配对必须一致），不能假设。

`kvarn_wht_128` 本体（`kvarn.cu:456–472`，逐字）：

```cpp
static __device__ void kvarn_wht_128(float * values) {
    __syncthreads();
    for (int stride = 1; stride < KVAR_N_DIM; stride *= 2) {
        if (threadIdx.x < 64) {
            const int j = (threadIdx.x / stride) * (2 * stride) + (threadIdx.x % stride);
            const float a = values[j];
            const float b = values[j + stride];
            values[j] = a + b;
            values[j + stride] = a - b;
        }
        __syncthreads();
    }
    if (threadIdx.x < KVAR_N_DIM) {
        values[threadIdx.x] *= 0.08838834764831845f;
    }
    __syncthreads();
}
```

### 4.2 store 内核选择（`kvarn.cu:2536`，逐字）

```cpp
    if (head_slices > 1 && smpbo >= KVAR_N_SHARED_BYTES) {
        g_kvarn_store_headwide_monolithic.fetch_add(1, std::memory_order_relaxed);
        ...
        kvarn_store_kernel_headwide<<<n_heads / head_slices, KVAR_N_DIM, KVAR_N_SHARED_BYTES, stream>>>(...)
```

D256 + sm_86（smpbo 99KB ≥ 68.07KB）⇒ **走 headwide**（每 block 一个 256-head，2 slice 一起处理，做跨 slice 蝶形）。headwide 的 grid 是 `n_heads/head_slices`，也就是 blockIdx.x 指向"逻辑 head"，slice 在 block 内循环。

### 4.3 staging / tail 的槽位算法（`kvarn.cu:1043–1044`，逐字）

```cpp
        const int stage_slot = assigned_slot >= 0 ? assigned_slot :
            (swa ? (group % stage_groups) : (group == 0 ? 0 : 1 + ((group - 1) % tail_groups)));
```
group 0 永远占 slot 0（永久 sink），其余 group 在 `1 + (group-1)%tail_groups` 里循环；非 SWA 的 `stage_groups = tail_groups + 1`（见 `llama-kv-cache-kvarn.h:311–321` 注释）。

---

## 5. 移植时最容易踩错的 3 个细节（附出处）

1. **字节账用错口径 ⇒ 每档少算 6 B/token（分配不足）**
   doc 00/01/02 的 36 组合表来源 `llama_kvarn_make_layout`（head_dim=256）在引擎里**没有调用者**，真实分配是 `llama_kvarn_make_record_layout` × head_slices。D256 真实 = `32*(kb+vb)+24` B/token/KV head（K4V2 = **216**，不是 210）。出处：`src/llama-kv-cache-kvarn.cpp:303`、`:1480–1483`、`:1497–1500`；`src/llama-kvarn.cpp:634`（make_layout 定义）vs `:673`（record 定义）；grep 全树 make_layout 调用者只有 `tests/test-kvarn.cpp:680/919/966`。按 210 分配会短 768 B/group/head。

2. **D256 的旋转不能只做「128 点 WHT」，也不能想当然套 256 点 FWHT 的下标配对**
   目标卡走 headwide 路径，是 `wht_128 ×2 slice + kvarn_wht_cross_slices`（slice0=(a+b)/√2，slice1=(a−b)/√2），并且**K 要转置、V 不转置**（`token = value?row:col`）。出处：`kvarn.cu:1059`（headwide 定义）、`:1113–1121`（WHT + cross_slices）、`:510–543`（cross_slices 本体）、`:2536`（D256 选 headwide 的条件 `head_slices>1 && smpbo>=KVAR_N_SHARED_BYTES`）、`:779–786`（K/V 转置语义）；读侧对应 `fattn-mma-kvarn-load.cuh:219–220`。混了任一处，解码值全错但形状不乱。

3. **位打包必须「连续比特流 + LSB-first + 每行整字节对齐」，不能按字节内高位在前**
   写侧 `bit_offset = col*bits`、`((q>>bit)&1) << (dst_bit%8)`；读侧 4-bit 是 `record[index>>1] >> ((index&1)<<2)`、2-bit 是 `>> ((index&3)<<1)`；3/5/6-bit 用 16-bit 小端窗口。出处：`kvarn.cu:734–747`（写）、`src/llama-kvarn.cpp:715–748`（CPU 写/读）、`fattn-mma-kvarn-load.cuh:5–50`（读侧真值）。**另外**：`row_bytes = 128*bits/8 = 16*bits` 必须整除（3/5/6-bit 恰好成立），移植时若改成非 128 的 group 或非 16*bits 的行宽，3/5/6-bit 会跨行串位。

---

## 6. 没找到 / 未覆盖

- **没找到** `GGML_KVARN_TEST_FORCE_LOWSHMEM` 之外的强制路径开关；`kvarn.cu:2248` 是它唯一出现处。
- doc 01 §2 表里"idx 6"未列出（doc 01 只列到 idx 5 就跳 7），真实有 `MAT_SWA = 6`；`MAT_VALUE=1` 的真实值 doc 01 未列。
- 本报告**未**逐行核对 `ggml-common.h` / `ggml-quants.c/h`——因为 KVarN 在本 commit 里是**自有 record 格式（GGML_TYPE_I8 张量 + kvarn 自定义 op）**，不走 ggml 量化类型表：记录张量在 `src/llama-kv-cache-kvarn.cpp:1499–1500` 以 `GGML_TYPE_I8` 分配，量化/反量化的唯一真值就是 §3.5/§3.6 那两处。若主控仍要 `ggml-common.h` 里的 layout/类型定义，请重新指派（本次未读）。
- `test-kvarn.cpp:680/919/966` 里 `make_layout(128,128,...)` 的用法**只覆盖 D128**，因此测试通过不能证明 D256 的 make_layout 口径可用（它本来就是死代码）。

---

**报告路径**：`H:\Hermes工作区\Hermes\kvarn-全档位移植-20261013\参考-beellama核对.md`
**只在 `ref-beellama\` 与本文件内改动**：✅ 未触碰 `kvarn-任务A-20261012`，未跑 docker，未 ssh，未在本仓跑 git。
