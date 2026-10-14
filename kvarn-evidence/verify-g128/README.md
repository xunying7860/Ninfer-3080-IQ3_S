# verify-g128 —— KVarN G128 引擎级对拍工具

把「喂给 attention 的那份值」从未被对拍过的教训（`实机取证-20261013.md` §10/§11）变成一条可复现的判据：

> **`record 解码` == `主机反量化`**（含列因子 / 行 scale 折叠口径、含逆旋转），逐向量 cosine ≥ 0.999。

纯 host 计算，**不需要 GPU、不需要 nvcc、不需要 python**，只用 g++。

---

## 1. 一条命令跑完

```bash
docker run --rm -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/verify-g128:/w" -w /w \
  nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run.sh"
```

独立容器/独立卷，**不碰 `kvarn-build`/`kvarnbuild`**，不写别人的 gate 目录与 oracle 目录
（oracle 只以「只读复制」形式 import，见 §6）。全部原始输出落在 `out/`。

单步（比如 CH1 只想验自己 dump 出来的一份 plane）：

```bash
./verify --from-dump <你的dump> [--decoded <引擎解码输出.kvo>] [--verbose]
```

---

## 2. 三个入口

| 入口 | 输入 | 用途 |
|---|---|---|
| `--oracle <dump.bin>` | GATE2 的 dumprec（magic `GVK2` v2） | 真实/合成 tile 的 record，逐 head 对拍 |
| `--from-dump <file>` | **KVD1**（本工具定义的引擎 plane dump 最小格式，§3） | CH1 从引擎 plane dump 原始字节直接喂进来 |
| `--decoded <file.kvo>` | **KVO1**（§3.5） | 引擎自己解出来的值 —— **主判据的真身** |

辅助入口：`--make-dump`（GVK2→KVD1）、`--gen-dump`（内部造真 dump）、`--gen-fake`（逐字段手写最小 dump + 正确/投毒两份引擎解码）、`--selftest`、`--sweep`。

---

## 3. dump 最小格式（CH1 只要照着写这么点字段）

**全部小端（x86 原生）；无对齐/padding；字符串 UTF-8 不带终止符。**

### 3.1 KVD1 —— 引擎 plane dump（`code` + `scale` 两段，页数/token 数）

```
偏移   类型        字段          说明
  0    u32 magic   'KVD1'        = 0x3144564B
  4    u32 version 1
  8    u32 n_records
--- 每条 record 重复 ---
 12    u32 name_len  name_len
       u8  name[name_len]
       i32 kind         0 = K（row=dim, col=token）  1 = V（row=token, col=dim）
       i32 bits         1..8（G128 实测 2/3/4/5/6/8）
       i32 rows         一条 record 的行数（G128 record_dim=128 ⇒ 恒 128）
       i32 cols         一条 record 的列数（token_group=128 ⇒ 恒 128）
       i32 tokens       有效 token 数（未满 tile < rows；满 tile = 128）
       i32 pages        覆盖页数（每页 64 token ⇒ ceil(tokens/64)，最小 1）
       i32 head_slices  2（D256 = 2×128；配对用）
       u32 flags        bit0 = 本 record 带「量化前原值」
       u32 code_bytes   = rows*cols*bits/8
       u8  code[code_bytes]       ← 一段「record 字节」（码平面，低位先）
       u32 scale_halfs  = 2*rows + cols
       u16 scale[scale_halfs]     ← 一段「scale 字节」：
                                     [0, rows)      = scale_axis[r]（fp16，已折进行因子 best_row）
                                     [rows, 2rows)  = zp_axis[r]   （fp16，已折进行因子 best_row）
                                     [2rows, +cols) = other_axis[c]（fp16，列因子 best_col）
  [flags&1 时] u32 orig_floats = rows*cols
               f32 orig[orig_floats]   量化前原值（旋转域），行主序
```

**两条 record 相邻 = 同一个 head 的 slice 0 / slice 1**（`head_slices=2`）；`--decoded` 的 head 也按同一顺序配对。

**最小例子（实测产物 `data/fake_min.dump`，298,224 B = 12 + 4×74,553）**：

```
od -A d -t x1 -N 12 data/fake_min.dump
0000000 4b 56 44 31 01 00 00 00 04 00 00 00
        \_KVD1__/  ver=1   n_records=4

第一条 record（偏移 12 起）：
  12: 09 00 00 00                  name_len = 9
  16: 66 61 6b 65 5f 4b 5f 73 30   "fake_K_s0"
  25: 00 00 00 00                  kind   = 0 (K)
  29: 04 00 00 00                  bits   = 4
  33: 80 00 00 00                  rows   = 128
  37: 80 00 00 00                  cols   = 128
  41: 80 00 00 00                  tokens = 128
  45: 02 00 00 00                  pages  = 2
  49: 02 00 00 00                  head_slices = 2
  53: 01 00 00 00                  flags  = 1（带原值）
  57: 00 20 00 00                  code_bytes = 8192
  61: <8192 B 码>  <768 B = 384 × fp16 scale>  <4 B>  <65536 B = 16384 × f32 原值>
```

### 3.2 CH1 侧写 dump 的最短路径

引擎里只要能把 plane 的两段字节指针拿出来，就是：

```c
// 每个 (页组/head) 一条 record；k_plane/v_plane 是 U8，k_scale/v_scale 是 FP16
static void write_kvd1(FILE* f, const char* name, int kind, int bits, int rows, int cols,
                       int tokens, int pages, const uint8_t* code, const uint16_t* scale) {
  uint32_t nl = strlen(name), cb = rows*cols*bits/8, sh = 2*rows + cols;
  fwrite(&nl,4,1,f); fwrite(name,1,nl,f);
  int32_t a[7] = {kind,bits,rows,cols,tokens,pages,2}; fwrite(a,4,7,f);
  uint32_t flags = 0; fwrite(&flags,4,1,f);                       // 无原值时 0
  fwrite(&cb,4,1,f); fwrite(code,1,cb,f);
  fwrite(&sh,4,1,f); fwrite(scale,2,sh,f);
  uint32_t of = 0; fwrite(&of,4,1,f);
}
// 文件头：'KVD1' + ver=1 + n_records
```

**只喂 plane 字节（flags=0）时**：工具用「主机反量化」和「引擎口径解码」两条独立路径互校
（判据 A 退化为工具自洽见证）；要判引擎读路径，再加 `--decoded`。

### 3.3 KVO1 —— 引擎解码输出（`--decoded`）

```
  0    u32 magic 'KVO1' = 0x314F564B
  4    u32 version 1
  8    u32 n_heads
--- 每个 head ---
       u32 name_len; u8 name[]
       i32 kind    0=K 1=V
       i32 tokens
       i32 width   256
       i32 domain  0 = 旋转域（attention 直接吃的形态）
                   1 = 已做逆旋转（materialize/bf16 mirror 形态）
       u32 n_floats = tokens*width
       f32 data[tokens*width]   data[t*256 + d]，t=token，d=0..255（d=s*128+dl）
```

`domain` 决定工具把「主机反量化」也逆旋转过去再比 —— 这样两种形态都能对。

---

## 4. 判据（无形容词）

逐 head（一条 head = 2 条 slice record = 一个 token 的 D256 向量）计算：

| 量 | 定义 | 作用 |
|---|---|---|
| `cos_adq_orig` | `主机反量化` vs **量化前原值**（旋转域） | **诊断 C**：编码端质量 |
| `cos_eng_orig` | `record 解码` vs 量化前原值 | 读路径端到端 |
| `cos_ENG_ADQ` | **`record 解码` vs `主机反量化`** | **判据 A**（同一份字节、两条独立解码路径） |
| `cos(引擎解码输出, 主机反量化)` | 有 `--decoded` 时 | **判据 A 的真身** |
| `maxabs` / `maxrel` / `nonfinite` | 最大绝对误差 / 该向量 max\|ref\| 归一的相对误差 / nan+inf 计数 | 健全性 |

### 算过

> **判据 A**：有 `--decoded` 时 —— 每个 head 的 `cos(引擎解码输出, 主机反量化) ≥ 0.999`；
> 无 `--decoded` 时 —— 两条独立解码路径逐元素 `maxabs == 0`（工具自洽见证）。
> **判据 B**：`nonfinite == 0`。
> **A ∧ B ⇒ PASS**。

### 不过 ⇒ 病灶在哪

| 现象 | 结论 |
|---|---|
| `cos(引擎解码输出, 主机反量化) < 0.999`，而 `cos_adq_orig` 正常 | **解码端**：引擎读路径错 —— 列因子/行 scale 折叠乘重了或漏了、位序/半字节槽位寻址错、`domain` 与逆旋转不匹配 |
| `cos_adq_orig < 0.99`（`bits ≥ 4`） | **编码端**：取水前就已偏 —— 量化 kernel、旋转（`normalized_hadamard_d256`）或 F16 stage |
| 两条都对但 `--decoded` 的 `cos(引擎解码输出, 原值)` 崩 | 引擎解出来的值没喂到 attention，或喂之前又被改了一道 |
| `nonfinite > 0` | 读到了未初始化/padding 页或错 plane（`scale_leading_extent` 给错） |

`bits = 2` 的 `cos_adq_orig ≈ 0.90` **不是故障**：2-bit 量化噪声本身如此（阶段 0 已定 K4V2 不上线），工具把它单列为诊断、不计入判据。

**两个 entry 的真伪由工具自己证**：`--gen-fake` 逐字段手写最小 dump（码/scale/原值/两组引擎解码都由字节直接拼出），
正确组必须 `PASS`、投毒组（K 的 token 7 整体取反）必须 `FAIL 且指向解码端`；`run.sh` §6a 两步都跑。
另有 `tools/make_fake_dump.py` 作为**独立第二实现**写同一格式，同一 reader 读出来必须给出同样的 1.0。

---

## 5. 覆盖矩阵与实测结果

`--sweep`（合成，含未旋转原值，可跑逆旋转臂）：**K4V4 / K4V2 × {128, 127, 51, 1} token × {gauss, outlier} × {K, V} = 32 格**。
`--oracle`（真实 tile = layer35/63 k/v_proj，按 beellama D256 口径旋转后切片）：K4V4 16 head + K4V2 16 head。

「未满 tile」口径：单条 record 恒有 128 个 token 槽位，N 之后的槽位置 0 再量化，统计只取前 N 个 token
—— 与引擎 B/C 路（尾块先写 stage、封口时整 128 槽一起编码）同口径。

关键实测（原始行见 `out/sweep.txt`、`out/oracle_real_k4v4.txt`）：

- `--sweep`：32 格全部 `cos_ENG_ADQ = 1.000000000`，`maxabs = 0.000e+00`，`nonfinite = 0`。
- 逆旋转臂（gauss_K tok=128）：`cos(逆旋转(解码), 未旋转原值) = 0.995129324`；
  反证 `cos(不逆旋转(解码), 未旋转原值) = -0.008403976` ⇒ 逆旋转确实必需且施加正确。
- 真实 tile K4V4：`min cos_adq_orig = 0.993589292`（v 通道最差，与 oracle 的 `cos_glob ≈ 0.9938` 一致，
  差异来自本工具按 token 的 D256 取向量、oracle 按整 record 取）。
- **`--selftest`：`iters=0` 与 `iters=16` 均 32/32 record、0 差异字节** ⇒ import 的量化器副本没走样。
- 手写最小 dump：正确组 3 路 cosine `1.000000000`、`maxabs=0`、PASS；投毒组 `min cos = 0.984375000`、FAIL。

---

## 6. 与 oracle 的关系（只读 import）

`kvarn_g128_verify.cpp` 的 §0–§5 是 `gate2-kvarn-quant/dump_oracle.cpp` 的**逐字复制**（该文件本身是
`oracle/bee_oracle.cpp` 的只读复制 + `dumprec` 追加段）：`f32_to_f16_rn` / `f16_to_f32` / `h16` /
`bee_packed_bytes` / `bee_pack_bits` / `bee_unpack_bits_value` / `std_of` / `sinkhorn_gpu` /
`QuantMeta` / `bee_quant_tile` / `bee_dequant_tile` / `fill_gauss`。
`run.sh` 的 `--selftest` 每次都会把这些副本再跑一遍、与 dump 逐字节比，证明副本没有走样。

本工具**新增**（不来自 oracle）：`norm_hadamard_256`（= ninfer `normalized_hadamard_d256_inplace` 的标量等价，
自逆，两道 gate 结论 §gate1 已定案照搬 ninfer 侧）、`engine_elem`（行内 merge 字节寻址）、D256 逐 token 组装、
指标/三种 reader/KVD1·KVO1 读写/sweep。

`import/dump_oracle.cpp` 是只读副本（未改动），只为产 GATE2 dumprec 用。

---

## 7. 边界（别把工具当万能）

1. **没有 `--decoded` 时判据 A 是自洽见证，不是引擎检查**：`record 解码`（本工具按引擎口径实现）与
   `主机反量化`读的是同一份字节、数学等价，必然 1.0。要判**引擎的实现**，必须让引擎把它解码出来的值 dump 成 KVO1。
2. **`--from-dump` 假定 plane 的字节就是 beellama record 布局**（`payload || scale_axis || zp_axis || other_axis`）。
   引擎若用 P1 的「页自足 + 槽位」布局（接线细案 §2.3：每页 64 token、维轴按页存两份），
   **dump 前需要按 record 语义重排**，或把槽位口径也写进 README 后再扩展本工具的 reader。
3. 逆旋转用 `norm_hadamard_256`（整数 FWHT + 精确 `2^-4`）。若引擎实际用的是 beellama 的
   `0.08838834764831845f × 0.7071067811865475f`（= 0.062499997860716139，与 1/16 差 0.57 ulp），
   cosine 仍会到 0.999+，但逐字节不会相等 —— **不要拿这条对拍当逐字节判据**。
4. 只在 host 上做数值对拍；不跑引擎、不派 GPU、不 ssh。
