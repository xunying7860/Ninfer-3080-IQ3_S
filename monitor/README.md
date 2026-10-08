# NInfer 监控面板（黑金）

把 **Strata 的 Monitor 监控面板**原样搬到 **NInfer** 上，**布局完全一致**，只换配色（黑金）。

- 访问：`http://169.254.146.59:18083/`（x99，直连网线）
- 引擎：NInfer 0.`*`（`ninfer.service`，GPU0，端口 18082，rk4v4，262144 ctx）
- 参考对象：`http://169.254.146.59:8080/#monitor`（Strata 0.1.40.3 的 Monitor 视图）

## 布局是「抄来的」，不是「重画的」

| 文件 | 来源 | 处理 |
|---|---|---|
| `web/components.css` | Strata `serve/web/components.css` | **逐字节复制**（md5 一致） |
| `web/app.css` | Strata `serve/web/app.css` | **逐字节复制**（md5 一致） |
| `web/sprite.svg` | Strata `serve/web/sprite.svg` | **逐字节复制**（md5 一致） |
| `web/tokens.css` | Strata `serve/web/tokens.css` | 结构逐字保留，**颜色变量全换成黑金** |
| `web/index.html` | Strata `serve/web/index.html` 的 Monitor 段 | 结构逐字克隆（只改品牌名/标题/表头 1 处文案） |
| `web/app.js` | Strata `serve/web/app.js` 的 Monitor 渲染部分 | 渲染逻辑同源，数据源改为 `/api/monitor` |

所以卡片、网格、进度条、仪表盘、火花线、请求表的**尺寸/间距/圆角/层级**都与 Strata 一模一样；
唯一变化是 `tokens.css` 里的颜色变量。

## 配色（黑金）

主色取自 `D:\Downloads\1.webp`（纯黑底 + 金属金 + 暖白高光）：

| 变量 | 值 | 用途 |
|---|---|---|
| `--st-bg` | `#000000` | 页面底（纯黑） |
| `--st-surface` / `-2` | `#0b0a08` / `#17140e` | 卡片、次级面（暖黑） |
| `--st-line` | `#3a2f18` | 描边（暗金） |
| `--st-ink` / `-soft` / `-muted` | `#f3e7c6` / `#d8c9a2` / `#9b8a63` | 暖米金文字三档 |
| `--st-accent` | `#d4af37` | 主金（进度条、仪表盘、Generating 徽标） |
| `--st-accent-hover` | `#ffd75e` | 高光金 |
| `--st-warn` / `--st-danger` | `#ffb02e` / `#ff5252` | 温度告警 / 错误 |

亮色主题是同一套金色的「香槟变体」（`#f8f4e9` + `#b8860b`），右上角按钮切换。

## 数据从哪来（全部只读）

`server/ninfer-monitor.py` 把 NInfer 的数据聚合成与 Strata `/metrics` **同形状**的 JSON，
前端因此可以沿用 Strata 的渲染逻辑。

| 面板元素 | Strata 的来源 | NInfer 的来源 |
|---|---|---|
| Model state 徽标 | `/metrics.live.state` | 日志 `throughput … running N (prefill N\|decode-ready N)` + `/metrics` 在飞计数 |
| Speed（Decode/Prefill） | `live.tok_s` / `live.prefill_tok_s_mean` | 同上（日志每 5 s 一行） |
| GPU load / VRAM / Temp / Power / PCIe | NVML | NVML（`libnvidia-ml.so.1` ctypes 直调，**GPU0**） |
| CPU / RAM / Disk read | psutil / OS | `/proc/stat` `/proc/meminfo` `/proc/diskstats` |
| Context fill 仪表盘 | `live.prompt_tokens+generated / max_context` | `/metrics` 的 `llamacpp:kv_cache_usage_ratio` + `kv_cache_tokens` |
| Context fill 第 1 条 bar | Experts in VRAM | **Model in VRAM**（`engine ready` 行的 `weights 11.3 GiB` ÷ 显存） |
| Recent requests 表 | `/metrics.requests[]` | 日志 `req#N done \| …`（prompt/reused/output/decode/TTFT/total/MTP） |
| 表第 7 列 | VRAM hit rate（专家命中） | **Cache hit rate**（`cache N (99.6%, …)` 前缀缓存命中）+ `MTP a/b` |
| Context cache 卡 | `conversation_cache` | 日志 `context cache \| …` + `/metrics` 的 reused/prefix 计数 |

**取不到的字段一律显示 `–`，绝不编数。** NInfer 没有「实时 prefill 进度百分比」和「专家缓存占用」，
所以那两处分别退化成「prefill 速率」和「权重占显存比例」，其余布局位置不变。

## 部署（x99）

```bash
# 本机 → x99
scp -r server web /tmp/            # 或直接 scp 到 /data/ninfer/monitor/
ssh xunying@169.254.146.59
sudo mkdir -p /data/ninfer/monitor && sudo chown xunying:xunying /data/ninfer/monitor
# 把 server/ web/ 放到 /data/ninfer/monitor/ 下
sudo cp /data/ninfer/monitor/server/ninfer-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now ninfer-monitor
```

验证：

```bash
curl -s http://169.254.146.59:18083/health          # {"status":"ok"}
curl -s http://169.254.146.59:18083/api/monitor | head -c 400
```

## 与 Strata 面板的差异（全部只有这些）

| # | 项目 | Strata | 本面板 | 原因 |
|---|---|---|---|---|
| 1 | 配色 | 亮/暗青绿 | **黑金** | 本次要求 |
| 2 | 品牌名 | Strata | NInfer | 目标引擎 |
| 3 | 顶栏标签页 | Chat / Monitor / About | 仅 Monitor | 只做监控面板（Chat/About 要另配接口） |
| 4 | 请求表第 7 列表头 | VRAM hit rate（专家命中） | **Cache hit rate**（前缀缓存命中）+ `MTP a/b` | NInfer 没有专家缓存/PCIe 分流 |
| 5 | Context fill 第 1 条 bar | Experts in VRAM | **Model in VRAM** | 同上（改用 `engine ready` 的权重体积） |
| 6 | 底部卡片标题 | Conversation cache | **Context cache** | NInfer 自己的叫法 |
| 7 | 底部卡片两条 bar | 显示 | 隐藏 | NInfer 没有实时停放计数接口，不编数 |
| 8 | Model state 进度条 | prefill 百分比 | 不填（0%） | NInfer 不吐 prefill 进度字段 |

其余——卡片顺序、8 个指标卡 2×4 网格、圆角/间距/字号/阴影、环形仪表盘、火花线、
请求表列顺序与对齐、底部卡片结构、状态徽标配色——**全部是 Strata 的原样**（见下）。

## 已验证（2026-10-08）

- **布局资产逐字节同源**：`components.css` / `app.css` / `sprite.svg` 的 md5 与
  `/data/workspace/strata01403/serve/web/` 下的原件**完全一致**（布局不是重画的）。
  ```
  a18c2efc0f32330e91d69c003aa6ed40  components.css
  f8d04860aebb34b50da4e88b46be764c  app.css
  7651d566a47d8e1b6bc4a8f1ff61738e  sprite.svg
  ```
- **两套主题都出图核对过**：黑金（默认）与香槟亮色均无对比度问题、无空白卡片、无图标缺失、无文字溢出。
- **真数字**（来自引擎，非造数）：`engine.up=true`，GPU0 99% / 299 W / 68 °C、Context fill 44%（114k/256K）、
  最近一条请求 `prompt 45,616 · reuse 98.1% · prefill 983 tok/s · decode 80.5 tok/s · MTP 1973/3783`。
- **故障路径**：把服务指向一个不存在的引擎端口 ⇒ `engine.up=false`，胶囊显示 `Engine not reachable`（不假装 Idle）。
- **解析器离线回归**：整份真实日志（29,556 行 / 200 条请求）灌进 `handle_line`，
  字段缺失只出现在「传输途中取消」的请求上（本就无读数）。
- **常驻**：`systemctl is-enabled ninfer-monitor` = `enabled`；重启服务后自动恢复，引擎重启时日志 tailer 自愈（跟踪 inode）。

## 环境变量（可选覆盖，默认即服务里那套）

`NINFER_HOST` `NINFER_PORT` `NINFER_LOG` `NINFER_MODEL_DIR` `NINFER_GPU`
`NINFER_MONITOR_BIND` `NINFER_MONITOR_PORT`（默认 18083）

## 版本

- v1.0 —— Monitor 视图全量搬迁，黑金配色，NInfer 数据接入，systemd 常驻自启。
- v1.0.1 —— 修静态资源 `web/` 前缀映射；prefill 兜底取「最近一条有读数的请求」；
  VRAM 副标题改 `256K ctx · rk4v4`；新增 `engine.up` 存活指示；端口/日志/GPU 支持环境变量覆盖。

