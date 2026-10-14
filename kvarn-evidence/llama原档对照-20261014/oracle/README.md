# 阶段 1 · beellama KVarN 标量 oracle（bee_oracle）

> **本目录是 `../../oracle/` 的副本（2026-10-14 复制）**，用于 llama.cpp 原生 KV 档对拍任务。
> 本文件描述的是**原来的** bee_oracle 工具链，内容未改，作为 KVarN 侧口径的参照。
> 新增内容见同目录 `llama_oracle.cpp` / `run_llama.sh` / `xcheck_numpy.py` / `agg_llama.py`，
> 以及上一级 `../README.md` 与 `../llama原档对照-20261014.md`。

纯 C++17 标量实现，复现 beellama.cpp 的 **KVarN G128 量化 + record 存储口径**，作为后续 CUDA kernel 的唯一数值基准。
不碰引擎、不碰 GPU、无需 nvcc 也能跑（nvcc 只用于 fp16 转换器的交叉校验）。

---

## 1. 编译与运行（全部命令）

宿主侧先生成真实 tile（需要 torch；**只需跑一次**，产物已入库）：

```bash
python make_tiles.py --group 128      # -> data/real_g128.bin  (32 tiles)
python make_tiles.py --group 64       # -> data/real_g64.bin   (64 tiles)
```

编译与全部实验（容器里，一条命令跑完）：

```bash
docker run --rm -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/oracle:/w" -w /w \
  nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run.sh"
```

单步命令（minimal）：

```bash
docker run --rm -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/oracle:/w" -w /w \
  nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc \
  "g++ -O2 -std=c++17 -o bee bee_oracle.cpp && ./bee table"
```

子命令：

| 命令 | 作用 |
|---|---|
| `./bee table` | 36 组合三口径字节账逐格对照 + 每格 tile 码流长度与三段偏移 |
| `./bee packtest` | `bee_pack_bits`/`bee_unpack_bits_value` 对 bits∈{2,3,4,5,6,8} 往返 + 每行打包等价性 |
| `./bee sinktest data/real_g128.bin` | Sinkhorn 的 GPU 口径 vs CPU 口径逐迭代对照 + 中途 dump |
| `./bee quant --tiles data/real_g128.bin --bits-k 4 --bits-v 2 --group 128 --iters 16` | 真实数据量化→反量化误差 |
| `./bee quant --gauss --group 128 --bits-k 4 --bits-v 4` | 合成高斯 |
| `./bee quant --outlier --group 64 --bits-k 4 --bits-v 4` | 合成（带离群通道） |
| `BEE_DEBUG=1 ./bee sinktest ...` | 打印每个迭代的 imbalance 中间量 |

`quant` 全选项：`--bits-k/--bits-v/--group/--iters/--cpu-order/--no-stage-f16/--tiles/--gauss/--outlier/--seed`。

---

## 2. 固定下来的可复现条件

| 项 | 值 |
|---|---|
| 随机种子 | `20261013`（`--seed`，合成数据） |
| 合成分布 | 高斯 N(0,1)；`--outlier` = 第 0/7/63 行 ×20 + 每元素乘 `1+0.25·|t(3)|`（模拟离群通道/重尾） |
| tile 尺寸 | K: 128(dim) × G(token)；V: G(token) × 128(dim)；G ∈ {64,128} |
| 迭代数 | 16（beellama 默认，`llama-kvarn.cpp:198 sinkhorn_iters = 16`）；另测 8 |
| stage fp16 | 默认开（beellama stage 是 F16，`kvarn.cu:785 __half2float(stage[...])`） |
| 真实数据 | `layer35/63_{k,v}_proj.pt`（256 token × 4 head × 256 dim），按 beellama D256 口径旋转后切片 |
| 旋转 | 逐片 WHT128(×0.08838834764831845) + 跨 2 片 Hadamard(×0.7071067811865475)（`kvarn.cu:1105-1130`） |

真实 tile 的 sha256 写在 `data/*.bin` 生成日志与 `manifest.txt` 里；`make_tiles.py` 每次运行都会重算并打印。

---

## 3. 代码对应关系（逐行对照上游，见 `ref/`）

| 本文件函数 | 上游 |
|---|---|
| `f32_to_f16_rn` | `__float2half_rn` 的标量等价（`f16_check.cu` 逐位对拍 6,000,021 个值，0 不一致） |
| `bee_layout` | `llama-kvarn.cpp:634 llama_kvarn_make_layout` |
| `make_record_layout` | `llama-kvarn.cpp:673 llama_kvarn_make_record_layout` |
| `bee_pack_bits` / `bee_unpack_bits_value` | `llama-kvarn.cpp:715 / :735` |
| `bee_quant_tile`（默认） | `kvarn.cu:660 kvarn_quantize_tile`（GPU，生产路径），依赖 `kvarn.cu:545/562/593/624` |
| `bee_quant_tile --cpu-order` | `llama-kvarn.cpp:824 variance_normalize` + `:892 quantize_tile` |
| `bee_dequant_tile` | `llama-kvarn.cpp:999/1019 dequantize_k/v_tile` 与 `kvarn.cu` materialize 的读法 |
| `make_record_layout` 的 `rows/cols` | `kvarn.cu:2233 record_rows/record_cols`（同一套公式，arena 侧断言在 `kvarn.cu:2236`） |

`ref/` 里是抓下来的上游源码原文（`--ssl-no-revoke` 从 raw.githubusercontent.com 取），sha256 见 `manifest.txt`。

---

## 4. 目录

```
bee_oracle.cpp     标量 oracle（全部逻辑）
f16_check.cu       用 nvcc 的 __float2half_rn 校验标量 fp16 转换器
make_tiles.py      torch 侧：真实 K/V 按 beellama 口径旋转 + 切片 -> data/*.bin
aggregate.py       解析 out/*.txt 汇总报告数字
run.sh             一条命令跑完全部实验
ref/               上游源码原文（只读参照）
data/              real_g128.bin / real_g64.bin（真实 tile）
out/               各实验的原始输出（报告引用的是这些）
```

---

## 5. 已知边界（别把它当万能）

1. **WHT 数值等价性未验**：本 oracle 的旋转是照 `kvarn.cu` 的写法实现的，但没有跟 ninfer 的
   `normalized_hadamard_d256_inplace` 对拍（doc 03 §3.9 的开放项，属阶段 2）。codec 本身与旋转无关。
2. **G64 是"同一 codec 推广到 group=64"**，不是 ninfer 旧 G64 codec 的复刻——beellama 的
   `token_group` 是编译期常量 `KVAR_N_GROUP = 128`，`make_record_layout` 里根本没有 group 参数。
3. **未验 3/5/6-bit 的位序与 GPU 一致**：`bee_pack_bits` 逐行照抄 CPU 侧，且已证明与
   `kvarn.cu:734` 的每行打包等价；但没有真实 GPU 上跑 3/5/6-bit 的上游数据可比。
4. **误差口径**：本 oracle 报的是"原向量 vs 反量化向量"的 cosine，**不是**旧档案 §22/§23/§29 那种
   "host 全链 vs kernel 平面解码"的 codec-对-codec cosine，两者不可直接比。
