# offsets-g128 —— KVarN G128「P1 页自足」leading / 页内偏移表（可机检）

本目录把 P1 几何从「猜」变成「算得出 + 可机检」：CSV 偏移表 + 一个纯 C++ 校验程序
（与 gate2 的整 tile record 布局互相换算、与 oracle dump 的 record 逐段比对、算清多花字节），
外加一个用**引擎真头**在 3080 上求值 leading/偏移的探针。

**结论表、推导链与未确认清单见**：`..\P1几何与偏移表-20261013.md`

## 一句话结论

`leading_extent` 是**每 token 行**的元素数（不是每页总量）。P1 真值：

| plane | dtype | leading | 说明 |
|---|---|---|---|
| `k_code_pages` | U8 | `32·kb` | K4V2 ⇒ **128** |
| `v_code_pages` | U8 | `32·vb` | K4V2 ⇒ **64** |
| `k_scale_pages` | FP16 | `10` | 640 half/页 = 2×128 scale + 2×128 zp + 2×64 other |
| `v_scale_pages` | FP16 | `8` | 512 half/页 = 2×128 other + 2×64 scale + 2×64 zp |

（细案 §2.3 的 8192/4096/640/512 是这四个值 × 64 的**页总量**；其字节账 29184 B/tile 与本目录实测相同。）

每 tile 每 head 多花 **1536 B = 12 B/token/KV head**（维轴在 tile 两页各存一份），
B/token = `32·(Kb+Vb)+36`（record 口径 = `+24`）。
K4V2 = **228**（280 的 81.43%）、K4V4 = **292**（104.29%）、K4V3 = 260、K3V3 = 228、K5V5 = 356、K6V6 = 420。

## 复现

```bash
# 主校验（CPU，约 20 s；会在 data/ 重新生成 6 档 oracle dump）
docker run --rm -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/offsets-g128:/w" -w /w \
  nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run.sh"

# 引擎真头 offset 探针（需 GPU；引擎目录只读挂载，只 include）
docker run --rm --gpus all \
  -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/offsets-g128:/w" \
  -v "H:/Hermes工作区/Hermes/kvarn-任务A-20261012/engine/ninfer-all-kvarn:/eng:ro" \
  -w /w nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run_gpu.sh"
```

上一次运行：主校验 `总检查 = 3293568 不一致 = 0` → PASS；探针 `不一致总数 = 0` → PASS。
原始输出留在 `out/run.txt` / `out/run_gpu.txt`。
`data/*.bin` 与编译产物按 gate2 的先例**已清空**（`run.sh` 一分钟内全量重生成；两次跑出的
`out/*.csv` sha256 相同，说明可复现）。

## 文件

```
offsets_g128_check.cpp     主校验：record↔P1 双射 + oracle 逐段 + 字节账 + CSV（纯 C++17，无 CUDA）
probe_engine_offsets.cu    引擎真头 paged_kv_address.cuh 的 offset 探针（nvcc -std=c++20 -arch=sm_86）
run.sh / run_gpu.sh        一键复现
out/offsets_p1_g128.csv    36 档：4 个 leading、页字节、tile 字节、record 字节、多花、B/token、相对 280 %
out/p1_slot_map_g128.csv   6 档逐槽页内偏移表（9,984 行；13 列：plane/row/col/item/slice/index/record 分段/…）
out/run.txt, run_gpu.txt   原始输出
dump_oracle.cpp            gate2 的 dumprec oracle（只读复制，sha256 7fae780f…）
tiles/real_g128.bin        阶段 1 真实 tile（只读复制，sha256 77ae2a60…45a8e7）
data/                      本目录自产的 6 档 oracle dump（已清空，可重生成）
```

未触碰：`kvarn-任务A-20261012\`（只读 `:ro` include 引擎头）、`engine-g128\`、任何别人的 gate 目录、
`kvarn-build` 容器 / `kvarnbuild` 卷；本仓内未跑任何 git 命令。
