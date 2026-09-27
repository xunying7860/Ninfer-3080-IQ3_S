# NInfer + Qwen3.8-27B IQ3_S —— RTX 3080 20GB 整合包

一个**完全离线、完全自带**的 27B 本地大模型推理整合包，专为 **RTX 3080 20GB（sm_86）** 调优。

解压双击就能用：不需要装 Python，不需要装 CUDA，不需要联网，不需要任何配置。

| | |
|---|---|
| 引擎 | [NInfer](https://github.com/Neroued/ninfer) —— 从零写的 C++20/CUDA 单卡推理引擎 |
| 模型 | Qwen3.8-27B GSQ-RCO IQ3_S（3.5 bit，13.99 GiB，含视觉 + MTP） |
| 平台 | Windows 10 / 11 x64 |
| 大小 | 压缩包 17.30 GiB，解压后 19.09 GiB |

---

## 下载

**去 [Releases](../../releases) 页面下载全部 10 个分卷**（GitHub 单个附件上限 2 GiB，所以拆了包）：

```
NInfer-Qwen3.8-27B-IQ3S-sm86-win64.7z.001
NInfer-Qwen3.8-27B-IQ3S-sm86-win64.7z.002
...
NInfer-Qwen3.8-27B-IQ3S-sm86-win64.7z.010
```

### 解压

1. 把 10 个分卷**放进同一个文件夹**（缺一个都解不开）
2. 用 [7-Zip](https://www.7-zip.org/) 右键点 **`.001`** → `7-Zip` → `提取到当前文件夹`
   - 命令行等价写法：`7z x NInfer-Qwen3.8-27B-IQ3S-sm86-win64.7z.001`
3. 得到 `ninfer-iq3s` 文件夹

> 只需要对 `.001` 操作，7-Zip 会自动按顺序找齐其余分卷。
> Windows 11（23H2 以后）原生支持 7z，但分卷还是建议用 7-Zip。

### 校验

分卷的 SHA256 见 Release 里的 `SHA256SUMS.txt`。

整个整合包解压后，模型文件的 SHA256 应为：

```
29e54467ca0ddae7e9aac04439f72ad20dbf6dc8fb5e84833b41ec7de9c46750
Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer
```

校验命令（在解压后的目录里执行）：

```bat
certutil -hashfile models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer SHA256
```

---

## 硬件要求

| 项目 | 要求 |
|---|---|
| **显卡** | **NVIDIA RTX 3080 20GB** |
| 系统 | Windows 10 / 11 64 位 |
| 驱动 | 支持 CUDA 13.3 的 NVIDIA 驱动（2025 年后的都行） |
| 内存 | 建议 32 GB 以上 |
| 硬盘 | 约 19 GB |
| 端口 | 占用 `127.0.0.1:18080`，只监听本地 |

本包的显存预算、设备路由表、上下文上限、投机解码档位，全部是在 RTX 3080 20GB / sm_86 上实测标定的。**同款卡解压双击即可，不需要调整。**

显存不足会启动失败并明确报错（日志里写 `requires ... but only ...`），不会静默出错。

---

## 使用

1. 解压（路径建议全英文，例如 `D:\ninfer`）
2. 双击 **`start.bat`**
3. 等到窗口里出现 `INFO  listening on http://127.0.0.1:18080`
   - 首次启动冷读 13.99 GiB 权重，需 1~3 分钟
   - 之后启动约 5 秒
4. 脚本会自动打开浏览器上的内置界面

**三种接入方式：**

| 方式 | 地址 / 配置 |
|---|---|
| 内置网页 | `http://127.0.0.1:18080/` |
| OpenAI 兼容客户端 | `http://127.0.0.1:18080/v1`，模型名 `qwen3.8-27b`，API Key 随便填非空值 |
| Anthropic Messages | 同上地址，双协议兼容 |

**停止服务**：关掉那个黑窗口，或双击 `stop.bat`。

---

## 包里已经调好了什么

- 上下文 **152K**（155,648 token），KV 用 **int8**
- 投机解码：**MTP 3 草稿 + lm-head-draft + adaptive-mtp**
- 思考档位拉到最高（`--default-reasoning-effort max`），且**无 token 上限**
- **视觉开启**（图像/视频输入），视觉塔放主机内存按需借用显存
- 两个对话同时驻留显存，来回切换不丢缓存
- 显存参数按 Windows WDDM 计费规则调过
- 聊天模板换成 [froggeric v22.5](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)（修掉了官方模板的工具调用和缓存问题）
- 自带 VC++ 运行时，对方机器不用另装 Redist

每一条都有实测依据，详细数据见包内 `README.txt`。

### 实测速度（RTX 3080 20GB）

| 项目 | 数值 |
|---|---|
| 预填充 | 约 1,300 ~ 1,400 tok/s |
| 生成 | 约 100 ~ 115 tok/s |
| 普通问题端到端 | 约 2.8 秒（含思考） |
| 模型加载 | 5 秒（已缓存）/ 1~3 分钟（冷读） |

### 质量

量化来自 ISTA-DASLab 的 GSQ-RCO —— 逐张量选类型的校准量化，平均只有 3.5 bpw，但质量高于多数 4.5 bpw 的量化：

| 基准 | IQ3_S | 官方 groupwise-int |
|---|---|---|
| IFBench (strict) | **80.33%** | 77.67% |
| AIME 2025 | **100%** | 96.67% |
| AIME 2026 | **100%** | 96.67% |
| GPQA-Diamond | **88.38%** | 87.37% |
| WikiText-2 PPL | **7.071** | — |

---

## 常见问题

**启动窗口一闪就没了？**
用命令行进目录执行 `start.bat` 看报错。最常见是显存不足，错误信息会说明需要多少、只剩多少。

**提示找不到 DLL？**
本包已自带所有依赖（含 VC++ 运行时）。若仍报错说明解压不完整，注意 10 个分卷都要下载。

**任务管理器显示共享显存有 1~2 GB，正常吗？**
正常，不影响性能。原因是 Windows 的 WDDM 会把 pinned 主机内存也算进显卡预算，包内 `README.txt` 的「为什么主机层要刻意开小」一节有完整分析。

**日志里 `prefill xx tok/s` 很低，是引擎慢吗？**
不是。那个字段的分母只算**新** token，缓存命中高的时候必然难看。看 `TTFT`、`total`、`mtp accepted` 三个才有意义。详见包内 `README.txt`。

**需要联网吗？**
完全不需要，跑起来后断网一样用。只有你主动让它读网络图片 URL 时才需要。

---

## 许可

- 模型底座 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)：Apache-2.0
- 引擎 [Neroued/ninfer](https://github.com/Neroued/ninfer)：Apache-2.0
  本包基于 [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all) 的 sm_86 + Windows 分支构建
- 聊天模板 [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
- 量化 [ISTA-DASLab](https://huggingface.co/ISTA-DASLab) GSQ-RCO

本仓库只做整合与分发，不对上述项目主张任何权利。
