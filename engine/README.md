# 引擎 bundle（仓库内文件，不是外链）

本目录放的是**能把这套服务跑起来的引擎本体**：NInfer sm_86 Linux bundle ——
`bundle/bin/ninfer-serve` 加它自带的 loader/glibc/依赖库 `bundle/libs/`（共 148 个文件、1,424,817,968 B）。
**只有模型权重不在仓库里**（13,568,617,472 B，太大，见仓库根 README 的下载说明）。

## 里面到底是什么

| 项 | 值 |
|---|---|
| 打包内容 | `bundle/bin/ninfer-serve`（1,249,869,248 B，md5 `2cbd88ab06318c81b695ec64c72b433d`）+ `bundle/libs/`（其余 147 个文件，174,948,720 B，含 `lib64/ld-linux-x86-64.so.2`） |
| 打包方式 | `tar`（GNU 格式，保留权限）→ `zstd -19 -T8 --long=27`，源：x99 `/data/ninfer/bundle`，2026-10-14 14:18–14:21 |
| 整包 | `engine-bundle-2cbd88ab.tar.zst` 928,429,363 B（tar 流 1,425,182,720 B 的 65.2%；148 个文件净字节 1,424,817,968）<br>md5 `8ab568bf1f17621ee3d785bba6896849` · sha256 `4dcced3cfbb3513aeb228f4281dbc233e866c428c8d187a4a21751b74eb49b87` |
| 为什么切成 10 卷 | GitHub 单文件硬顶 100 MiB。每卷 94,371,840 B（90 MiB），末卷 79,082,803 B，**全部用普通 git 存，不需要 Git LFS** |
| 压缩为什么只有 35% | 实测引擎件中段（偏移 400 MiB / 800 MiB 各取 64 MiB）在 zstd-19 下比率 0.9976 / 0.9955，xz-9 也是 0.998+ —— 中段是高熵数据（内嵌预编译 cubin 类），**压不动**；只有头部（13%）、尾部与 libs 能压 |

## 怎么装

```bash
# 1) 只看校验（不落盘、不改任何东西）
bash engine/install-engine.sh --check-only

# 2) 真装（默认目标 /data/ninfer；旧引擎件与旧 libs 改名留档到 archive-engine-<时间戳>/，不删）
bash engine/install-engine.sh /data/ninfer

# 手动等价命令
cat engine/engine-bundle-*.tar.zst.part-* | zstd -dc | tar -xC /data/ninfer
chmod 755 /data/ninfer/bundle/bin/ninfer-serve
```

需要 `zstd`（Ubuntu 22.04：`sudo apt install zstd`）。解压 3 s，解完 `run.sh` 里那两条路径就都就位：

```
/data/ninfer/bundle/libs/lib64/ld-linux-x86-64.so.2
/data/ninfer/bundle/libs/lib/x86_64-linux-gnu
```

## 已经验过什么（不是"理论上能用"）

2026-10-14 在 x99 上做的闭环校验：

1. 分卷 `cat` 重接后尺寸与 md5 **等于**整包（928,429,363 B / `8ab568bf…`）；
2. 重接 → `zstd -dc` → `tar -x` 真解到 scratch，**148 个文件逐个 md5 与 `/data/ninfer/bundle` 源完全一致**，权限 755 保留、符号链接 0 个（源侧也是 0）；
3. 用解出来的 `loader + libs + ninfer-serve` 在**纯解包路径**下跑 `--help`，能吐完整参数表 ⇒ loader 与依赖库自足，不缺件。

`checksums.txt` 里就是这三层清单：整包 md5/sha256、10 个分卷 md5、解包后 148 个文件的 md5。

## 注意

- 包内引擎件就是**当前生产在跑的那一个**（与 `/data/ninfer/bundle/bin/ninfer-serve.vendor-original` 同一个 md5）。
- KVarN 试验件（`ninfer-serve.cand`、`.master` 等）**不在包里**——那是开发中的候选，不是生产。
- 模型权重请放到模型目录后，用 `run.sh` 里写的路径启动；权重与 model-id 等参数一律以 `run.sh` 为准（`run.sh` 的注释里有每项取舍的理由和回退依据）。
