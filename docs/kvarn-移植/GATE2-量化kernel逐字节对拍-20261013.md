# GATE 2 · KVarN G128 CUDA 量化/存储 kernel 与标量 oracle 的 record 逐字节对拍（2026-10-13）

生成时间：2026-10-13 +0800
设备：本机 `NVIDIA GeForce RTX 3080`（sm_86，CC 8.6）——**本机实跑，非 x99**
产物目录：`H:\Hermes工作区\Hermes\kvarn-全档位移植-20261013\gate2-kvarn-quant\`
用途：判定「把 beellama KVarN G128 的量化/存储 kernel 搬到 CUDA 后，产出的 record 是否与阶段 1 的标量
oracle 逐字节一致」，并给出实测 required smem 与设备上限。

---

## 结论先行

**分开说，因为两个结论方向相反：**

1. **结构性（布局/打包/量化/record 三轴）逐字节一致 —— PASS，而且是很硬的 PASS。**
   - `iters=0`（Sinkhorn 不迭代，只剩 tile→record 的布局与算术）：**576/576 record、0 差异字节**。
     覆盖 2/3/4/5/6-bit × {合成高斯、离群、真实数据} 全部 18 个 dump。
   - 真实数据（`tiles/real_g128.bin`，就是 beellama D256 口径旋好的 K/V 投影切片）在 **iters=16、
     flags=默认**下：K4V4 **32/32 record、0 差异字节**；全 6 档合计 **7/192 record 有差异、共 8 字节**。
   - 存储路径的「旋转 + F16 stage」段：**0 / 131072 字节差异**。
2. **严格「任意配置下逐字节相等」—— FAIL。** 原因不是移植错，是**宿主 libm 与 CUDA 设备数学的 ulp 级差被
   Sinkhorn 迭代放大**，且**上游 ggml-cuda 真用 `-use_fast_math`**（下述第 5 节），使得标量 oracle
   在物理上不可能作为 GPU 构建的逐字节基准。

**三个关键数字（全量 iters=16、K4V4/K4V2/2/3/5/6-bit × 三数据集 = 576 record）：**

| 量 | 值 |
|---|---|
| `flags=默认`（精确数学）差异 | 193/576 record 有差异，**48,747** 字节（其中真实数据只占 **8** 字节） |
| `flags=-use_fast_math`（上游真 flag）差异 | 266/576 record 有差异，**74,583** 字节 |
| 1-ulp 输入扰动（设备自身敏感性，同一 iters） | `g_it16` 差异 **4,600** 字节 vs 实测 host-vs-device **4,532** 字节（同量级 → 差异已被解释） |

**required smem 与设备上限（实测，非引用值）：**

| 项 | 值 |
|---|---|
| required（= `KVAR_N_SHARED_FLOATS=17426` floats） | **69,704 B = 68.07 KB** |
| 设备 opt-in 上限（`cudaDevAttrMaxSharedMemoryPerBlockOptin`） | **101,376 B = 99.00 KB** |
| 余量 | **31,672 B** |
| 默认上限 / smem-per-SM / 每 block 预留 | 49,152 B / 102,400 B / 1,024 B |
| 编译后 kernel 属性 | 静态 smem 0 B、寄存器 30/线程、`maxDynamicSharedSizeBytes=69704`、occupancy **1 block/SM**（128 线程 + 69,704 B 动态 smem） |

---

## 0. 可复现命令行（原文）

```bash
# 全量复现（一条命令：编译 + 生成 oracle dump + 全部对拍）
docker run --rm --gpus all -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/gate2-kvarn-quant:/w" -w /w \
  nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run.sh"

# 最小可复现（只跑「iters=0 全档位结构性对拍」，20 秒级）
docker run --rm --gpus all -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/gate2-kvarn-quant:/w" -w /w \
  nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc '
    g++ -O2 -std=c++17 -w -o dump dump_oracle.cpp &&
    nvcc -O2 -std=c++17 -w -arch=sm_86 -o gate_def kvarn_quant_gate.cu &&
    for kv in "4 4" "4 2" "2 2" "3 3" "5 5" "6 6"; do set -- $kv
      ./dump dumprec data/z0_real_k$1v$2.bin --tiles tiles/real_g128.bin --bits-k $1 --bits-v $2 --iters 0 >/dev/null; done &&
    ./gate_def data/z0_real_*.bin'
```

> `--gpus all` 是相对任务书示例命令的**唯一**增补：任务书给的示例命令（gate 1 的纯数值对拍）不需要设备，
> 而 gate 2 的「实测 smem / 设备上限 / 占用率」必须在真设备上跑。不加 `--gpus all` 时容器会报
> `WARNING: The NVIDIA Driver was not detected`。除此之外未用任何命名容器或命名卷（`docker run --rm`，
> 无 `--name`、无 `-v 卷名`），未触碰 `kvarn-build` 容器 / `kvarnbuild` 卷。

工具链（`out/build.txt`）：

```
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Build cuda_13.1.r13.1/compiler.37061995_0        # nvcc -arch=sm_86
```

---

## 1. 被对拍的两侧（都抄原文，不凭记忆）

### 1.1 CUDA 侧（`kvarn_quant_gate.cu`）

从 `ref-beellama/src/ggml/src/ggml-cuda/kvarn.cu`（commit `58a1629274d8d74ea5a6acf34f30fb5946c2422b`）
逐行抄入，**唯一改动是 `kvarn_quantize_tile` 末尾追加 3 行 `[GATE2]` 诊断导出（默认 `nullptr`，对 record 无影响）**：

| 本文件符号 | 上游行号 |
|---|---|
| `KVAR_N_DIM/_TILE_VALUES/_REDUCE_FLOATS/_SHARED_FLOATS/_SHARED_BYTES` | `kvarn.cu:12,14,15,16,17` |
| `kvarn_wht_128` | `kvarn.cu:456-472` |
| `kvarn_warp_min` / `kvarn_warp_max` | `kvarn.cu:579-591` |
| `kvarn_reduce_std_ranges` | `kvarn.cu:593-622` |
| `kvarn_update_best_from_std` | `kvarn.cu:624-658` |
| `kvarn_std_col` / `kvarn_std_row` | `kvarn.cu:545-577` |
| `kvarn_quantize_tile` | `kvarn.cu:660-757` |
| `kvarn_wht_cross_slices` | `kvarn.cu:510-543` |
| `kvarn_store_rotate_stage_kernel`（存储路径的旋转+F16 stage 段） | `kvarn.cu:1113-1131`（附件文件） |

任务书点名的 5 个性质逐条对上：

| 要求 | 落点 | 证据 |
|---|---|---|
| sm_86 | `-arch=sm_86` | `out/build.txt`；设备 sm_86 |
| 128 线程 | `<<<1, KVAR_N_DIM, ...>>>` | `kvarn.cu:678` 起 `threadIdx.x` 即 row；`kvarn.cu:723` `const int row = threadIdx.x;` |
| opt-in smem | `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, 69704)` | 69,704 B > 49,152 B 默认上限，必须 opt-in |
| tile 常驻不原地改 | `kvarn_std_col/row` 每轮除 `(sc*s_row[row])`；`kvarn_quantize_tile` 只在**打包**时写 `record` | `kvarn.cu:553-559`、`kvarn.cu:726-741`：`tile[]` 全程只读 |
| std 分母 127 | `(sum_sq - KVAR_N_DIM*mean*mean) / (KVAR_N_DIM - 1)` | `kvarn.cu:559`、`kvarn.cu:576` |
| 打包先 memset | `for (i<row_bytes) row_payload[i] = 0;` | `kvarn.cu:736-738` |

### 1.2 oracle 侧（`dump_oracle.cpp`）

`oracle/bee_oracle.cpp` 的**逐字复制**，SHA-256 校验一致：

```
2c1cb2470a21e0a5965a3d9bcb431b62d6bbd14c101944cfd07257a565fda3a2  oracle/bee_oracle.cpp
2c1cb2470a21e0a5965a3d9bcb431b62d6bbd14c101944cfd07257a565fda3a2  gate2-kvarn-quant/dump_oracle.cpp   （复制时刻）
```

唯一追加 = 子命令 `dumprec`（文件内已标注 `[GATE2 追加段]`）：把 `bee_quant_tile`（GPU 口径，
`kvarn.cu:660` 的宿主转写）的**输入 tile（过 `h16` 后）**、**record 全文**、**best_col/best_row/
best_iter/best_imbalance** 一起落盘，供 CUDA 侧读同一份输入跑 kernel 再比字节。
`oracle/` 目录**只读**，未写入任何文件。

---

## 2. 全量逐字节对拍结果（iters=16）

命令：`./gate_def -v <18 个 dump>` / `./gate_ff -v <18 个 dump>`（`out/gate_def_full.txt` / `out/gate_ff_full.txt`）

### 2.1 分数据集汇总

| 数据集 | records | `flags=默认` 有差异 record / 字节 | `flags=-use_fast_math` 有差异 record / 字节 |
|---|---:|---:|---:|
| 合成高斯 gauss | 192 | 111 / **27,412** | 143 / **42,207** |
| 合成离群 outlier | 192 | 75 / **21,327** | 110 / **32,362** |
| **真实数据 real** | 192 | **7 / 8** | 13 / 14 |
| 合计 | 576 | 193 / **48,747** | 266 / **74,583** |

### 2.2 真实数据逐档（`out/gate_def_full.txt`，最该看的一列）

| dump | records | 有差异 record | 差异字节 | `maxd\|s_col\|` | `maxd\|s_row\|` |
|---|---:|---:|---:|---:|---:|
| `real_k4v4` | 32 | **0** | **0** | 3.338e-06 | 1.192e-06 |
| `real_k4v2` | 32 | 2 | 2 | 3.338e-06 | 1.192e-06 |
| `real_k2v2` | 32 | 2 | 2 | 3.338e-06 | 1.192e-06 |
| `real_k3v3` | 32 | **0** | **0** | 3.338e-06 | 1.192e-06 |
| `real_k5v5` | 32 | 1 | 1 | 3.338e-06 | 1.192e-06 |
| `real_k6v6` | 32 | 2 | 3 | 3.338e-06 | 1.192e-06 |

### 2.3 合成数据逐档（`flags=默认`，K\V 位宽对）

| dump | rec B | 有差异 record | 差异字节 | `maxd\|s_col\|` | `maxd\|s_row\|` |
|---|---:|---:|---:|---:|---:|
| `gauss_k4v4` | 8960 | 18 | 4,532 | 7.486e-03 | 7.071e-03 |
| `gauss_k4v2` | 8960/4864 | 18 | 4,594 | 7.486e-03 | 7.071e-03 |
| `gauss_k2v2` | 4864 | 18 | 4,636 | 7.486e-03 | 7.071e-03 |
| `gauss_k3v3` | 6912 | 19 | 4,609 | 7.486e-03 | 7.071e-03 |
| `gauss_k5v5` | 11008 | 18 | 4,522 | 7.486e-03 | 7.071e-03 |
| `gauss_k6v6` | 13056 | 20 | 4,519 | 7.486e-03 | 7.071e-03 |
| `outlier_k4v4` | 8960 | 12 | 3,554 | 7.486e-02 | 3.554e-03 |
| `outlier_k4v2` | 8960/4864 | 12 | 3,553 | 7.486e-02 | 3.554e-03 |
| `outlier_k2v2` | 4864 | 12 | 3,569 | 7.486e-02 | 3.554e-03 |
| `outlier_k3v3` | 6912 | 12 | 3,561 | 7.486e-02 | 3.554e-03 |
| `outlier_k5v5` | 11008 | 15 | 3,554 | 7.486e-02 | 3.554e-03 |
| `outlier_k6v6` | 13056 | 12 | 3,536 | 7.486e-02 | 3.554e-03 |

> 记法：`8960/4864` = K 记录 8960 B（4-bit）+ V 记录 4864 B（2-bit），同一 dump 里两种位宽各 16 条。
> 记录长度与 record 口径闭式 `2048·bits + 768` 逐一相符（4-bit→8960、2-bit→4864、3-bit→6912、
> 5-bit→11008、6-bit→13056、8-bit→17152）。

---

## 3. 差异归因（这是本档最有价值的部分）

### 3.1 迭代数扫描 → 差异从第 2 个迭代开始，随迭代单调放大

K4V4，`flags=默认`（`out/diag_def.txt`），列 = 「有差异 record / 差异字节」：

| iters | gauss | outlier | real |
|---:|---:|---:|---:|
| 0 | **0 / 0** | **0 / 0** | **0 / 0** |
| 1 | **0 / 0** | 1 / 1 | **0 / 0** |
| 2 | 3 / 3 | 2 / 2 | **0 / 0** |
| 4 | 2 / 2 | 2 / 2 | **0 / 0** |
| 8 | 15 / **3,276** | 4 / **1,160** | **0 / 0** |
| 16 | 18 / **4,532** | 12 / **3,554** | **0 / 0** |

对应的 `maxd|s_col|` 同步增长：iters=1 时 1.19e-07（≈1 ulp）→ iters=8 时 2.35e-03 → iters=16 时 7.49e-03。
**即：Sinkhorn 的 `log/exp` 更新把 ulp 级差放大了约 4 个数量级。**

### 3.2 1-ulp 输入扰动探针 → 证明这是算法自身病态，不是移植错

把**同一份输入**的每个元素朝 +∞ 挪 **1 ulp**（`nextafterf`），只跑设备 kernel（宿主不参与），
看 record 变多少（`out/diag_def.txt` 的 `[sens ]`）：

| dump | host-vs-device 差异字节 | 1-ulp 输入扰动差异字节 |
|---|---:|---:|
| `g_it0` | 0 | 76 |
| `g_it1` | 0 | 3 |
| `g_it2` | 3 | 1 |
| `g_it4` | 2 | 2 |
| `g_it8` | 3,276 | 5,929 |
| **`g_it16`** | **4,532** | **4,600** |
| `o_it16` | 3,554 | 4,037 |
| `r_it16` | **0** | 14 |

**读法**：在 iters=16 上，一个 1-ulp 的输入扰动就能让 record 变 4,600 字节；而宿主↔设备的实际差是 4,532 字节。
**两者同量级 ⇒ 实测差异完全可以用「宿主 libm 与 CUDA 设备数学的 ulp 级差 + 迭代放大」解释，
不需要假设移植有结构 bug。** 反证方向同样成立：iters=0 时输入扰动只影响 76 字节而 host-vs-device 是 **0** 字节
——说明布局/量化算术两条路径**逐位相同**。

### 3.3 `-fmad=false` 不能救（`out/diag_nofma.txt`）

把它单独关掉 FMA 收缩（`-fmad=false`，其余同默认）：576 record 里 62 有差异、14,025 字节——与默认的
193/48,747 **同量级**。⇒ 主因不是 FMA，而是 `expf/logf`（CUDA 与 glibc 各差 ≤1~2 ulp）。关掉 FMA 反而
把 iters=0 的稳定性换来的是「迭代起点不同」，量级不变。

### 3.4 ⚠️ 否定发现：上游真的用 `-use_fast_math`

```
ref-beellama/src/ggml/src/ggml-cuda/CMakeLists.txt:216
    set(CUDA_FLAGS -use_fast_math -extended-lambda)
```

把 gate 换 `-use_fast_math` 编译（= 上游生产构建的数学），**连 iters=0 都不逐字节一致**：

| 配置 | 结果 |
|---|---|
| `iters=0`、`flags=默认`、全 6 档 × 3 数据集 | **576/576 record、0 差异字节 → PASS** |
| `iters=0`、`flags=-use_fast_math`、同 18 个 dump | **371/576 record 有差异、1,017 字节 → FAIL** |

（`out/gate_iters0_fullgrid.txt`；对应 `g_it0` 单档：30/32 record、82 字节，`maxd|deq|=4.744e-01`。）

**⇒ 结论：阶段 1 的标量 oracle 在物理上无法充当「上游 GPU 生产构建」的逐字节基准。**
`-use_fast_math` 会把 `div`/`sqrtf`/`expf`/`logf` 换成 approx 版本（≈2.4e-7 相对误差）并开 FTZ，
标量 oracle 用的是精确宿主 libm 与精确除法——两者在量化取整边界上必然偶发翻位（实测每档 ~2.7 字节/record）。

---

## 4. 存储路径（旋转 + F16 stage）—— PASS

`kvarn_store_gate.cu` 抄 `kvarn.cu:1113-1131` 的旋转 + stage 写（去掉 record 封口调度），
与宿主逐行转写（`wht_128` = 纯 add/sub + 一次 mul；`cross_slices` = 纯 add/sub + 一次 mul）对拍：

```
================ GATE2 附件：存储路径（旋转 + F16 stage）================
[device] NVIDIA GeForce RTX 3080 sm_86 ; smem/block opt-in = 101376 B
[sel   ] kvarn.cu:2536 选 headwide 的条件：head_slices(2) > 1 && smpbo(101376) >= KVAR_N_SHARED_BYTES(69704)
         => 走 headwide（每 block 一个 D256 head，block 内跨 2 slice）
[stage ] tokens=128 heads=4 dim=128 slices=2 ; stage 元素=65536
[stage ] CUDA 旋转+F16 stage vs 宿主转写：差异字节 = 0 / 131072  首差异@-1
判定：PASS（存储路径旋转 + F16 stage 逐字节一致）
```

（原始输出 `out/store.txt`。）这条同时实测确认了 **sm_86 上 D256 走的是 headwide 路径**
（`kvarn.cu:2536` 的条件在本机成立），不是 hishmem，也不是 lowshmem。

---

## 5. smem 与设备上限（实测原始输出）

```
================ GATE2: KVarN G128 CUDA 量化/存储 kernel ================
[device] NVIDIA GeForce RTX 3080  sm_86  (compute capability 8.6)
[device] smem/block 默认上限      cudaDevAttrMaxSharedMemoryPerBlock      = 49152 B
[device] smem/block opt-in 上限   ...PerBlockOptin                      = 101376 B  <== 设备上限
[device] smem/SM                  ...PerMultiprocessor                  = 102400 B
[device] 每 block 预留 smem        ...ReservedSharedMemoryPerBlock       = 1024 B
[need  ] KVAR_N_SHARED_FLOATS = 16384 + 8*128 + 2 + 16 = 17426 floats
[need  ] KVAR_N_SHARED_BYTES                          = 69704 B (68.07 KB)  <== 实测所需动态 smem
[need  ] 每 block 线程数 = 128（KVAR_N_DIM；kvarn.cu:678 起 threadIdx.x 即 row）
[attr  ] kernel 静态 smem=0 B, 寄存器=30/线程, maxDynamicSharedSizeBytes=69704 B
[attr  ] 占用率：1 blocks/SM（128 线程 + 69704 B 动态 smem/block）=> 128 线程/SM (上限 1536)
```

**要点**：
- `69,704 B` 与上游 `KVAR_N_SHARED_BYTES`（`kvarn.cu:16-17`）**逐位相同**，sm_86 上是**一次一块/SM**：
  每个 SM 102,400 B 只够 1 个 69,704 B 的 block；想 2 blocks/SM 需要 ≤ 51,200 B（含 1,024 B 预留），
  69,704 B 差得远。⇒ 目标卡的 D256 KVarN 是**占用率受限**的 kernel（这也解释了 `kvarn.cu:18-19` 那条
  `KVAR_N_LOWSHMEM_BYTES = 3144 B` 的存在意义：lowshmem 变体才是为「多 block/SM」准备的）。
- 余量 31,672 B 看着多，但对比 **每 SM 只有 102,400 B**，所以余量不可用于第二个 block。
- `kvarn.cu:2536` 用 `smpbo >= KVAR_N_SHARED_BYTES` 选 headwide，本机 `101376 >= 69704` 成立。

---

## 6. 给 ninfer 移植的建议（可执行）

1. **布局/打包照搬，零风险**：`iters=0` 下 576/576 record 逐字节一致，已覆盖 2/3/4/5/6-bit。
   这意味着 `row_bytes = 128*bits/8`、LSB-first 连续比特流、每行独立字节对齐、`record = payload +
   128·scale(fp16) + 128·zp(fp16) + 128·other(fp16)`、打包先 memset —— 这些**可以照抄**，不用再验。
2. **决定要不要「与 beellama 逐位一致」，这是策略问题**：
   - 若要 **record 逐位可互换**（同一份 KV cache 目录被两个引擎读；或做 A/B 质量对撞要排除 codec 变量）：
     必须 (a) 用 `-use_fast_math`（与上游一致），且 (b) 同迭代数、同 FMA 设置、同 `expf/logf` 实现
     —— 换句话说，**必须在设备上跑**，宿主标量实现永远对不齐。
   - 若只是**质量等价**（ninfer 自己产自己读）：用精确数学（默认 flag）反而更稳，与上游的差别在
     `maxd|s_col| ~ 7e-3`（合成）/ `3e-6`（真实）；真实数据 K4V4 甚至 **0 字节差**。代价是
     合成/离群这类分布上解码值最大绝对差可达 ~0.33（相对稀疏，出现在个别被翻位的码上）。
3. **oracle 只能当「结构性基准」，别当「逐字节基准」**：它对 `iters≥2` 的合成数据没有仲裁力
   （实测 1-ulp 扰动就能改 4,600 字节）。建议把验收口径改成
   —— 结构性：`iters=0` 逐字节（本档已 PASS）；
   —— 数值性：`iters=16` 比 `maxd|s_col|`/`maxd|s_row|` + 反量化域 `cosine`，而不是比字节。
4. **风险提示（应进 ninfer 风险清单）**：Sinkhorn 的 `best_iter` 选择跑在一条**病态轨迹**上。
   真实数据 `best_iter=0`（稳定，因此逐字节可复现）；合成/离群数据 `best_iter≈13~16`（放大 1e4 倍）。
   ⇒ 换 GPU/换 CUDA 版本/换编译 flag 都可能改变 record。**不要把 KVarN 的 record 当作跨构建可复现的
   制品**去做缓存校验或 checksum。

---

## 7. 未确认（不编）

1. **未在 x99 上跑**（任务书禁 ssh，且模块硬约束「只在 PC 上」）。x99 若是 Ada/Blackwell，
   `smpbo` 与占用率结论可能不同；本档所有设备数字只对 **RTX 3080 / sm_86** 有效。
2. **未与 ninfer 的 `normalized_hadamard_d256_inplace` 对拍** —— 那是 gate 1 的范围。本档的输入 tile
   直接取自 oracle 的 `tiles/real_g128.bin`（python 侧旋好），所以本条记录的结论**不依赖**旋转等价性。
3. **`-use_fast_math` 下 `iters=0` 的 1,017 字节差异未逐字节定位**（只统计了总数与首个差异位置）。
   经验判断是 `div.approx`/`sqrt.approx` 造成的取整边界翻位，但没有逐条列。
4. **未测 `KVAR_N_LOWSHMEM` 路径与多 stream 并发**（本档只验单 block 的数值正确性）。
5. **未测 8-bit**（任务书未点名；`packed_bytes` 对 8-bit 是 `record=17152`，本档的 bits 分支已按
   `bits∈{2,3,4,5,6}` 覆盖，8-bit 只看过 oracle 的字节账）。

---

## 8. 约束遵守

| 约束 | 状态 |
|---|---|
| 只在 PC 上，不 ssh x99 | ✅ 全程本机 RTX 3080 |
| 自己的容器，不用 `kvarn-build` / 卷 `kvarnbuild` | ✅ `docker run --rm`（无 `--name`、无命名卷） |
| 不改别人的目录、不改 `kvarn-任务A-20261012\` | ✅ `oracle/` 只读（只 `cp` 出来），`kvarn-任务A-20261012\` 未触碰 |
| 不在 `kvarn-全档位移植-20261013` 仓里跑 git | ✅ 未跑任何 git 命令 |
| 所有断言给文件:行号或原始输出 | ✅ 见各节引用与 `out/*.txt` |

---

## 9. 产物清单（`gate2-kvarn-quant/`）

```
dump_oracle.cpp          bee_oracle.cpp 逐字复制 + 追加 dumprec（sha256 已核）
kvarn_quant_gate.cu      量化 kernel（逐行抄 kvarn.cu）+ 对拍 harness + smem 报告
kvarn_store_gate.cu      存储路径（旋转+F16 stage 段，kvarn.cu:1113-1131）+ 宿主转写对拍
run.sh                   全量复现（编译 + 生成 dump + 全部对拍）
diag.sh                  差异归因（3 个 flag 变体 + iters 扫描）
tiles/real_g128.bin      oracle 产物只读复制（2,098,116 B）
data/*.bin               各配置的 oracle dump（**中间产物，已清空以保持仓库精简**；
                         由 `run.sh` 在 1 分钟内全量重生成，本目录只留空 data/）
out/build.txt            工具链与编译
out/dumps.txt            18 个 dump 的生成日志
out/gate_def_full.txt    全量对拍（flags=默认，逐 record）
out/gate_ff_full.txt     全量对拍（flags=-use_fast_math，逐 record）
out/gate_iters0_fullgrid.txt  iters=0 全档位结构性对拍（PASS）
out/diag_def.txt         iters 扫描 + 1-ulp 探针（flags=默认）
out/diag_nofma.txt       iters 扫描（-fmad=false）
out/diag_ff.txt          iters 扫描 + 1-ulp 探针（-use_fast_math）
out/store.txt            存储路径对拍
```
