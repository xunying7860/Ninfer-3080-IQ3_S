#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NInfer 监控面板（服务端）—— 布局照抄 Strata 的 Monitor 视图，只换黑金配色。

职责
----
1. 托管静态页面（web/index.html + web/*.css + web/*.js + web/sprite.svg）。
2. 提供 /api/monitor —— 把 NInfer 的数据聚合成与 Strata /metrics 同形状的 JSON，
   前端渲染逻辑因此可以和 Strata 保持一致。

数据来源（全部只读，不打扰引擎）
--------------------------------
* /data/ninfer/logs/ninfer-serve.log（增量 tail）：
    - `req#N started | …`            → 请求开始（含 max output）
    - `req#N done | …`               → 请求完成（prompt/output/cache/TTFT/total/queue/prefill/decode/mtp）
    - `req#N cancelled`              → 取消
    - `throughput | 5.0s | … running N (prefill|decode-ready M) …` → 实时 phase / 速率
    - `engine ready | …` / `capacity | …` / `context cache | …` / `media | …` → 启动参数事实
* http://127.0.0.1:<ninfer 端口>/metrics  → 累计计数器（Prometheus 文本）
* http://127.0.0.1:<ninfer 端口>/v1/models → n_ctx（权威上下文长度）
* NVML（libnvidia-ml.so.1，ctypes 直调，无需 pynvml）→ 指定 GPU 的 util/显存/温度/功耗/PCIe
* /proc/stat、/proc/meminfo、/proc/diskstats、/proc/mounts → CPU/内存/磁盘

约定：任何取不到的读数一律 None（前端显示 `–`），绝不编数。
"""

from __future__ import annotations

import ctypes
import json
import mimetypes
import os
import re
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ────────────────────────────────────────────────────────────────── 配置
# 环境变量可覆盖（用于改端口 / 指向另一台引擎 / 跑测试实例），默认即下面这些值。
_env = os.environ.get
NINFER_HOST = _env("NINFER_HOST", "127.0.0.1")
NINFER_PORT = int(_env("NINFER_PORT", "18082"))
LOG_PATH = _env("NINFER_LOG", "/data/ninfer/logs/ninfer-serve.log")
MODEL_DIR = _env("NINFER_MODEL_DIR", "/data/ninfer/models")   # 在其中定位模型所在卷（磁盘读写行的口径）
GPU_INDEX = int(_env("NINFER_GPU", "0"))                      # ninfer 固定跑 GPU0
BIND = _env("NINFER_MONITOR_BIND", "0.0.0.0")
PORT = int(_env("NINFER_MONITOR_PORT", "18083"))
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
HISTORY = 60                               # 火花线保留的点数（1 s 一个）
REQ_KEEP = 200                             # 请求表保留的条数
LOG_SCAN_TAIL = 8 * 1024 * 1024            # 启动时最多回读多少日志（用于填最近请求表）
THRU_STALE = 7.0                           # throughput 行多久没来就认为引擎空闲

# ────────────────────────────────────────────────────────────────── NVML
class Nvml:
    """ctypes 直调 NVML —— x99 上没有 pynvml，也不打算装（与 nvtop-style.py 同法）。"""

    class _Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class _Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self, index):
        self.lib = self.dev = None
        for name in ("libnvidia-ml.so.1", "libnvidia-ml.so"):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if self.lib is None:
            return
        try:
            init = getattr(self.lib, "nvmlInit_v2", None) or self.lib.nvmlInit
            if init() != 0:
                self.lib = None
                return
            h = ctypes.c_void_p()
            get = getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2", None) or self.lib.nvmlDeviceGetHandleByIndex
            if get(ctypes.c_uint(index), ctypes.byref(h)) != 0:
                self.lib = None
                return
            self.dev = h
        except (AttributeError, OSError):
            self.lib = None

    def ok(self):
        return self.lib is not None and self.dev is not None

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        try:
            return v.value if getattr(self.lib, fn)(self.dev, *args, ctypes.byref(v)) == 0 else None
        except (AttributeError, OSError):
            return None

    def name(self):
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetName(self.dev, buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace")
        except (AttributeError, OSError):
            pass
        return None

    def read(self):
        out = {}
        u = self._Util()
        try:
            if self.lib.nvmlDeviceGetUtilizationRates(self.dev, ctypes.byref(u)) == 0:
                out["util"] = u.gpu
        except (AttributeError, OSError):
            pass
        m = self._Mem()
        try:
            if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) == 0:
                out["mem_used"], out["mem_total"] = m.used, m.total
        except (AttributeError, OSError):
            pass
        out["temp"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(0))
        mw = self._uint("nvmlDeviceGetPowerUsage")
        out["power"] = mw / 1000.0 if mw is not None else None
        lim = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        out["power_limit"] = lim / 1000.0 if lim is not None else None
        out["pcie_gen"] = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")
        out["pcie_gen_max"] = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        rx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(1))   # NVML_PCIE_UTIL_RX_BYTES, KB/s
        out["pcie_rx_mb"] = rx / 1024.0 if rx is not None else None
        return out


# ────────────────────────────────────────────────────────────────── 共享状态
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.reqs = deque(maxlen=REQ_KEEP)      # 最新在前
        self.max_output = {}                    # req# -> 该请求的输出上限
        self.metrics = {}                       # Prometheus 文本 -> dict
        self.counters = {}                      # 累计值
        self.capacity = {}                      # capacity 行
        self.engine_ready = {}                  # engine ready 行
        self.context_cache = {}                 # context cache 行
        self.media = {}                         # media 行
        self.thru = None                        # 最近一条 throughput 的解析结果
        self.thru_at = 0.0                      # 上面那条的时间（单调钟）
        self.ninfer_ok = False                  # 上一轮 /metrics 是否拿到（引擎存活指示）
        self.n_ctx = None
        self.hw = {}
        self.hw_static = {}
        self.hist = {k: deque(maxlen=HISTORY) for k in
                     ("tok_s", "prefill_tok_s_mean", "gpu_util", "gpu_mem_used", "gpu_temp",
                      "gpu_power", "gpu_pcie_rx_mb", "cpu", "disk_read_mb")}
        self.disk_prev = None
        self.cpu_prev = None


S = State()

# ────────────────────────────────────────────────────────────────── 日志解析
TS_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\s+(\w+)\s+(.*)$")
REQ_RE = re.compile(r"^req#(\d+) (started|done|cancelled|error)\b\s*\|?\s*(.*)$")
DUR_RE = re.compile(r"(\d+)m\s*([\d.]+)s|([\d.]+)s")


def inum(s):
    """'90,672' → 90672；失败 None。"""
    try:
        return int(str(s).replace(",", ""))
    except Exception:
        return None


def fnum(s):
    try:
        return float(s)
    except Exception:
        return None


def parse_duration(s):
    """'1m 5.6s' / '12.2s' → 秒（float）；失败 None。"""
    m = re.search(r"(?:(\d+)m\s*)?([\d.]+)s", s or "")
    if not m:
        return None
    return (int(m.group(1)) * 60 if m.group(1) else 0) + float(m.group(2))


def parse_ms(s):
    """'910 ms' / '1.1s' → 毫秒（float）；失败 None。"""
    m = re.search(r"([\d.]+)\s*(ms|s)\b", s or "")
    if not m:
        return None
    v = float(m.group(1))
    return v if m.group(2) == "ms" else v * 1000.0


def epoch(ts):
    try:
        return time.mktime(time.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")) + float("0" + ts[19:])
    except Exception:
        return time.time()


def classify_finish(reason, what):
    """把 NInfer 日志里的结束原因映射到面板的 4 种状态徽标（与 Strata 同一套）。
    实测取值：stop token / tool calls N / output limit / cancelled / context capacity。"""
    r = (reason or "").lower()
    if what == "cancelled" or "cancel" in r:
        return "cancel"
    if what == "error":
        return "error"
    if "output limit" in r:
        return "length"
    if "capacity" in r:
        return "error"
    return "stop"


def handle_line(ln):
    m = TS_RE.match(ln)
    if not m:
        return
    when, level, rest = m.group(1), m.group(2), m.group(3)

    # ---- 请求
    r = REQ_RE.match(rest)
    if r:
        num, what, tail = r.group(1), r.group(2), r.group(3)
        with S.lock:
            if what == "started":
                mo = re.search(r"max output ([\d,]+)", tail)
                S.max_output[num] = inum(mo.group(1)) if mo else None
            elif what in ("done", "cancelled", "error"):
                S.max_output.pop(num, None)
                parts = [x.strip() for x in tail.split("|")]
                reason = parts[1] if (what == "done" and len(parts) > 1) else tail.split("|")[0].strip()
                rec = {
                    "id": num,
                    "time": epoch(when),
                    "prompt_tokens": None, "reused": None, "output_tokens": None,
                    "decode_tok_s": None, "prefill_tok_s": None, "hit_rate": None,
                    "duration_s": None, "ttft_ms": None, "queue_ms": None,
                    "drafts": None, "accepted": None,
                    "reason": reason or what,
                    "finish": classify_finish(reason, what),
                }
                p = re.search(r"prompt ([\d,]+)", tail)
                o = re.search(r"output ([\d,]+)", tail)
                c = re.search(r"cache ([\d,]+) \(([\d.]+)%", tail)
                dd = re.search(r"decode ([\d.]+) tok/s", tail)
                pf = re.search(r"prefill ([\d.]+) tok/s", tail)
                da = re.search(r"accepted ([\d,]+)/([\d,]+)", tail)
                if p:
                    rec["prompt_tokens"] = inum(p.group(1))
                if o:
                    rec["output_tokens"] = inum(o.group(1))
                if c:
                    rec["reused"] = inum(c.group(1))
                    rec["hit_rate"] = fnum(c.group(2))
                if dd:
                    rec["decode_tok_s"] = fnum(dd.group(1))
                if pf:
                    rec["prefill_tok_s"] = fnum(pf.group(1))
                if da:
                    rec["accepted"] = inum(da.group(1))
                    rec["drafts"] = inum(da.group(2))
                rec["duration_s"] = parse_duration(re.search(r"total ([^|]*)", tail).group(1) if "total" in tail else "")
                rec["ttft_ms"] = parse_ms(re.search(r"TTFT ([^|]*)", tail).group(1) if "TTFT" in tail else "")
                rec["queue_ms"] = parse_ms(re.search(r"queue ([^|]*)", tail).group(1) if "queue" in tail else "")
                S.reqs.appendleft(rec)
        return

    # ---- 实时吞吐（每 5 s 一行）
    if rest.startswith("throughput |"):
        seg = rest
        pre = re.search(r"prefill ([\d.]+)(k?) tok/s", seg)
        dec = re.search(r"decode ([\d.]+)(k?) tok/s", seg)
        run = re.search(r"running (\d+)(?: \(([^)]*)\))?", seg)
        phase = None
        if run:
            if run.group(2):
                phase = "prefill" if "prefill" in run.group(2) else "decode"
            elif int(run.group(1)) > 0:
                phase = "decode"
        with S.lock:
            S.thru = {
                "prefill": fnum(pre.group(1)) * (1000.0 if pre and pre.group(2) else 1.0) if pre else None,
                "decode": fnum(dec.group(1)) * (1000.0 if dec and dec.group(2) else 1.0) if dec else None,
                "running": int(run.group(1)) if run else 0,
                "phase": phase,
                "at": epoch(when),
            }
            S.thru_at = time.monotonic()
        return

    # ---- 启动事实（每次重启都会重发；取最近一次）
    if rest.startswith("capacity |"):
        kv = re.search(r"KV ([\d,]+) tokens, (\w+), (\w+)", rest)
        rt = re.search(r"runtime ([\d.]+) GiB", rest)
        fr = re.search(r"free ([\d.]+) GiB", rest)
        pg = re.search(r"pages ([\d,]+)/([\d,]+)", rest)
        with S.lock:
            S.capacity = {"kv_tokens": inum(kv.group(1)) if kv else None,
                          "kv_dtype": kv.group(2) if kv else None,
                          "kv_mode": kv.group(3) if kv else None,
                          "runtime_gib": fnum(rt.group(1)) if rt else None,
                          "free_gib": fnum(fr.group(1)) if fr else None,
                          "pages_used": inum(pg.group(1)) if pg else None,
                          "pages_total": inum(pg.group(2)) if pg else None}
        return
    if rest.startswith("engine ready |"):
        w = re.search(r"weights ([\d.]+) GiB", rest)
        parts = [x.strip() for x in rest.split("|")]
        with S.lock:
            S.engine_ready = {"model": parts[1] if len(parts) > 1 else None,
                              "weights_gib": fnum(w.group(1)) if w else None,
                              "total_s": (parts[2].replace("total", "").strip() if len(parts) > 2 else None)}
        return
    if rest.startswith("context cache |"):
        a = re.search(r"(\d+) active \+ (\d+) cached device states", rest)
        h = re.search(r"host (\d+) states, ([\d.]+) MiB KV", rest)
        with S.lock:
            S.context_cache = {"active": inum(a.group(1)) if a else None,
                               "cached": inum(a.group(2)) if a else None,
                               "host_states": inum(h.group(1)) if h else None,
                               "host_kv_mib": fnum(h.group(2)) if h else None}
        return
    if rest.startswith("media |"):
        w = re.search(r"(\d+) preprocess workers", rest)
        c = re.search(r"cache ([\d.]+) MiB", rest)
        l = re.search(r"live ([\d.]+) GiB", rest)
        with S.lock:
            S.media = {"workers": inum(w.group(1)) if w else None,
                       "cache_mib": fnum(c.group(1)) if c else None,
                       "live_gib": fnum(l.group(1)) if l else None}
        return


def log_tailer():
    """增量 tail 引擎日志；重启/截断都能自愈。"""
    off = 0
    ino = None
    buf = ""
    while True:
        try:
            st = os.stat(LOG_PATH)
            if ino != st.st_ino or st.st_size < off:
                ino, off, buf = st.st_ino, 0, ""
                off = max(0, st.st_size - LOG_SCAN_TAIL)   # 首次/换文件：回读一段，填最近请求表
            if st.st_size > off:
                with open(LOG_PATH, "rb") as f:
                    f.seek(off)
                    chunk = f.read(st.st_size - off)
                    off = f.tell()
                buf += chunk.decode("utf-8", "replace")
                *lines, buf = buf.split("\n")
                for ln in lines:
                    try:
                        handle_line(ln)
                    except Exception:
                        pass
                if len(buf) > 1 << 20:
                    buf = ""
        except FileNotFoundError:
            pass
        except Exception:
            pass
        time.sleep(0.5)


# ────────────────────────────────────────────────────────────────── 主机读数
def http_text(url, timeout=2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return ""


def parse_prometheus(txt):
    out = {}
    for ln in (txt or "").splitlines():
        if not ln or ln.startswith("#"):
            continue
        parts = ln.rsplit(" ", 1)
        if len(parts) == 2:
            try:
                out[parts[0].strip()] = float(parts[1])
            except ValueError:
                pass
    return out


def cpu_percent():
    try:
        with open("/proc/stat") as f:
            p = [float(x) for x in f.readline().split()[1:]]
    except Exception:
        return None
    tot, idle = sum(p), p[3] + (p[4] if len(p) > 4 else 0.0)
    prev, now = S.cpu_prev, (tot, idle)
    S.cpu_prev = now
    if not prev:
        return None
    dt, di = now[0] - prev[0], now[1] - prev[1]
    return round(100.0 * (1.0 - di / dt), 1) if dt > 0 else None


def mem_info():
    try:
        vals = {}
        with open("/proc/meminfo") as f:
            for ln in f:
                k, _, rest = ln.partition(":")
                p = rest.split()
                if p:
                    vals[k] = float(p[0]) * 1024.0          # kB → B
        tot = vals.get("MemTotal", 0.0)
        avail = vals.get("MemAvailable", 0.0)
        return {"ram_total": tot, "ram_used": tot - avail}
    except Exception:
        return {"ram_total": None, "ram_used": None}


def block_base(dev):
    """sda2→sda，nvme0n1p2→nvme0n1（/proc/diskstats 里的物理盘名）。"""
    m = re.match(r"^(nvme\d+n\d+)(?:p\d+)?$", dev)
    if m:
        return m.group(1)
    m = re.match(r"^((?:sd|vd|xvd|hd)[a-z]+)\d*$", dev)
    return m.group(1) if m else dev


def disk_device_for(path):
    """把挂载点解析成 /proc/diskstats 里的盘名；LVM(dm-N) 再下钻到物理盘（与 nvtop 面板同口径）。"""
    target = os.path.realpath(path)
    best = None
    try:
        with open("/proc/mounts") as f:
            for ln in f:
                p = ln.split()
                if len(p) < 2 or not p[0].startswith("/dev/"):
                    continue
                mp = p[1].replace("\\040", " ")
                if (target == mp or target.startswith(mp.rstrip("/") + "/")) and (best is None or len(mp) > len(best[1])):
                    best = (os.path.realpath(p[0]), mp)      # /dev/mapper/xxx → /dev/dm-0
    except Exception:
        return None
    if not best:
        return None
    dev = block_base(os.path.basename(best[0]))
    if dev.startswith("dm-"):
        try:
            slaves = sorted(os.listdir(f"/sys/block/{dev}/slaves"))
            if slaves:
                dev = block_base(slaves[0])
        except OSError:
            pass
    return dev


def diskstats():
    out = {}
    try:
        with open("/proc/diskstats") as f:
            for ln in f:
                p = ln.split()
                if len(p) >= 10:
                    out[p[2]] = (int(p[5]), int(p[9]))          # 读扇区、写扇区（512 B）
    except Exception:
        pass
    return out


def sample_hardware():
    """采一帧主机读数。GPU 走 NVML，CPU/内存/磁盘走 /proc；取不到就是 None。"""
    g = NVML.read() if (NVML and NVML.ok()) else {}
    mem = mem_info()
    cpu = cpu_percent()
    read_mb = write_mb = None
    ds = diskstats()
    prev, pt = S.disk_prev or (None, None)
    cur = ds.get(DISK_DEV) if DISK_DEV else None
    now = time.monotonic()
    dt = (now - pt) if pt else 0.0
    if cur and prev and dt > 0.05 and cur[0] >= prev[0] and cur[1] >= prev[1]:
        read_mb = (cur[0] - prev[0]) * 512.0 / dt / 1e6
        write_mb = (cur[1] - prev[1]) * 512.0 / dt / 1e6
    S.disk_prev = (cur, now)
    hw = {
        "gpu_util": g.get("util"), "gpu_mem_used": g.get("mem_used"), "gpu_mem_total": g.get("mem_total"),
        "gpu_temp": g.get("temp"), "gpu_power": round(g["power"], 1) if g.get("power") is not None else None,
        "gpu_power_limit": g.get("power_limit"), "gpu_pcie_gen": g.get("pcie_gen"),
        "gpu_pcie_gen_max": g.get("pcie_gen_max"), "gpu_pcie_width": g.get("pcie_width"),
        "gpu_pcie_rx_mb": round(g["pcie_rx_mb"], 2) if g.get("pcie_rx_mb") is not None else None,
        "cpu": cpu, "disk_read_mb": round(read_mb, 2) if read_mb is not None else None,
        "disk_write_mb": round(write_mb, 2) if write_mb is not None else None,
    }
    hw.update(mem)
    return hw


NVML = None
DISK_DEV = None
CPU_NAME = None


def init_static():
    global NVML, DISK_DEV, CPU_NAME
    NVML = Nvml(GPU_INDEX)
    DISK_DEV = disk_device_for(MODEL_DIR) or disk_device_for("/")
    name = thread = cores = None
    try:
        with open("/proc/cpuinfo") as f:
            txt = f.read()
        m = re.search(r"^model name\s*:\s*(.+)$", txt, re.M)
        name = m.group(1).strip() if m else None
        thread = txt.count("processor\t:")
        if not thread:
            thread = len(re.findall(r"^processor\s*:", txt, re.M))
        cores = len({m.group(1) for m in re.finditer(r"^core id\s*:\s*(\d+)$", txt, re.M)}) or None
        if cores:
            cores *= len({m.group(1) for m in re.finditer(r"^physical id\s*:\s*(\d+)$", txt, re.M)}) or 1
    except Exception:
        pass
    CPU_NAME = name
    return {"gpu_name": (NVML.name() if (NVML and NVML.ok()) else None) or None,
            "cpu_name": name, "cores": cores, "threads": thread,
            "disk_dev": DISK_DEV, "gpu_index": GPU_INDEX}


def sampler():
    """每秒采一帧：主机读数 + 引擎计数器 + 火花线历史。"""
    while True:
        hw = sample_hardware()
        mets = parse_prometheus(http_text(f"http://{NINFER_HOST}:{NINFER_PORT}/metrics"))
        if mets:
            S.counters = mets
        S.ninfer_ok = bool(mets)
        if S.n_ctx is None:
            try:
                j = json.loads(http_text(f"http://{NINFER_HOST}:{NINFER_PORT}/v1/models", 3.0) or "{}")
                for m in (j.get("data") or []):
                    v = m.get("n_ctx") or m.get("max_model_len")
                    if v:
                        S.n_ctx = int(v)
                        break
            except Exception:
                pass
        with S.lock:
            S.hw = hw
            live = live_state()
            S.hist["tok_s"].append(live["tok_s"])
            S.hist["prefill_tok_s_mean"].append(live["prefill_tok_s_mean"])
            for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "disk_read_mb"):
                S.hist[k].append(hw.get(k))
        time.sleep(1.0)


# ────────────────────────────────────────────────────────────────── 组装 JSON
def live_state():
    """从最近一条 throughput 行 + 在飞请求推断实时状态。调用方持锁。"""
    now = time.monotonic()
    fresh = S.thru and (now - S.thru_at) <= THRU_STALE
    state, tok_s, pre = "idle", None, None
    if fresh and S.thru["running"] > 0:
        if S.thru["phase"] == "prefill":
            state, pre = "reading", S.thru["prefill"]
            tok_s = S.thru["decode"]
        else:
            state, tok_s = "generating", S.thru["decode"]
            pre = S.thru["prefill"]
    queued = int(S.counters.get("llamacpp:requests_deferred") or 0)
    running = int(S.counters.get("llamacpp:requests_processing") or 0)
    if state == "idle" and running > 0:
        state = "reading"                       # 刚起请求、还没到第一条 throughput 行
    return {"state": state, "queued": queued, "tok_s": tok_s, "prefill_tok_s_mean": pre,
            "prompt_tokens": None, "prompt_read": None, "prompt_total": None,
            "generated": None, "max_tokens": None}


def build_payload(show_all=False):
    now = time.time()
    with S.lock:
        live = live_state()
        hw = dict(S.hw)
        hist = {k: list(v) for k, v in S.hist.items()}
        reqs = list(S.reqs)
        counters = dict(S.counters)
        cap, er, cc, media = dict(S.capacity), dict(S.engine_ready), dict(S.context_cache), dict(S.media)
        n_ctx = S.n_ctx

    uptime = counters.get("ninfer:uptime_seconds")
    max_ctx = n_ctx or cap.get("kv_tokens")
    kv_tokens = counters.get("llamacpp:kv_cache_tokens")
    kv_ratio = counters.get("llamacpp:kv_cache_usage_ratio")

    engine = {
        "up": bool(S.ninfer_ok),
        "model": (er.get("model") or "qwen3.8-27b"),
        "max_context": max_ctx,
        "kv": cap.get("kv_dtype"),
        "weights_bytes": (er.get("weights_gib") or 0) * 1073741824 if er.get("weights_gib") else None,
        "images": True,
        "load_s": er.get("total_s"),
        "runtime_gib": cap.get("runtime_gib"),
        "free_gib": cap.get("free_gib"),
    }

    prompt_read = counters.get("llamacpp:prompt_tokens_total") or 0
    reused = counters.get("ninfer:reused_prompt_tokens_total") or 0
    prefix_hits = counters.get("ninfer:prefix_cache_hit_tokens_total") or 0
    output = counters.get("llamacpp:tokens_predicted_total") or 0
    decode_s = counters.get("llamacpp:tokens_predicted_seconds_total") or 0
    prompt_s = counters.get("llamacpp:prompt_seconds_total") or 0
    # 口径：引擎的 prompt_tokens_total 只算「真正 prefill 过的」，reused_prompt_tokens_total 是
    # 「从上下文缓存恢复、没有重读的」。两者相加 = 请求处理过的提示词总量，不重复计数。
    totals = {"requests": counters.get("ninfer:requests_total"),
              "since": (now - uptime) if uptime else None,
              "prompt_tokens": prompt_read + reused, "reused": reused,
              "output_tokens": output, "prompt_ms": prompt_s * 1000.0, "decode_ms": decode_s * 1000.0,
              "drafted": counters.get("ninfer:draft_tokens_total"),
              "accepted": counters.get("ninfer:draft_accepted_tokens_total")}

    reused_requests = sum(1 for r in reqs if (r.get("reused") or 0) > 0)
    last = next((r for r in reqs if r.get("prompt_tokens")), None)
    all_prompt = prompt_read + reused
    ds = cc.get("cached")
    cc_out = {
        "enabled": False,
        "requests": len(reqs), "requests_reused": reused_requests,
        "reused_tokens": reused, "prompt_tokens": all_prompt,
        "prefix_hits": prefix_hits,
        "exhausted": counters.get("ninfer:context_cache_exhausted_requests_total"),
        "last_prompt": (last or {}).get("prompt_tokens"), "last_reused": (last or {}).get("reused"),
        "device_states": (f"{cc.get('active')} active + {cc.get('cached')} cached"
                          if cc.get("active") is not None else None),
        "host_kv": (f"{cc.get('host_states')} states, {cc.get('host_kv_mib')} MiB KV"
                    if cc.get("host_states") is not None else None),
        "policy": (f"{cap.get('kv_mode') or 'rolling'} · "
                   f"device-state slots {ds if ds is not None else '?'} · "
                   f"host-state slots {cc.get('host_states') if cc.get('host_states') is not None else '?'}"),
        "note": ("NInfer keeps the state of the conversation it is in, so a follow-up reads only what is new; "
                 "device states held in VRAM are the ones it can resume without reading the prompt again."),
    }

    lh = lambda k: ([v for v in hist.get(k, [])] if hist.get(k) else None)
    return {
        "engine": engine,
        "live": live,
        "hardware": hw,
        "hardware_static": S.hw_static,
        "history": {k: lh(k) for k in ("tok_s", "prefill_tok_s_mean", "gpu_util", "gpu_mem_used",
                                       "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "disk_read_mb")},
        "context": {"used_tokens": kv_tokens, "max_context": max_ctx, "ratio": kv_ratio},
        "requests": reqs if show_all else reqs[:12],
        "requests_kept": len(reqs),
        "totals": totals,
        "context_cache": cc_out,
        "media": media or None,
    }


# ────────────────────────────────────────────────────────────────── HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "ninfer-monitor/1.0"

    def log_message(self, *a):        # 静默（systemd 日志里不要每 1 s 一行）
        pass

    def _send(self, code, body, ctype, cache="no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path in ("/api/monitor", "/metrics.json"):
            show_all = "requests=all" in query
            payload = json.dumps(build_payload(show_all), ensure_ascii=False, default=str).encode()
            return self._send(200, payload, "application/json; charset=utf-8")
        if path == "/health":
            return self._send(200, b'{"status":"ok"}', "application/json")
        if path in ("/", "/index.html"):
            path = "/index.html"
        return self._static(path)

    def _static(self, path):
        rel = path.lstrip("/")
        # index.html 里的引用与 Strata 保持一致（web/xxx.css、web/app.js、web/sprite.svg），
        # 而本服务的根目录就是 web/ ⇒ 把前导的 web/ 去掉再映射。
        if rel.startswith("web/"):
            rel = rel[4:]
        full = os.path.normpath(os.path.join(WEB_DIR, rel))
        if not full.startswith(os.path.normpath(WEB_DIR)) or not os.path.isfile(full):
            return self._send(404, b"not found", "text/plain; charset=utf-8")
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8"
        try:
            with open(full, "rb") as f:
                data = f.read()
        except OSError:
            return self._send(500, b"read error", "text/plain")
        return self._send(200, data, ctype, cache="no-cache")


def main():
    S.hw_static = init_static()
    threading.Thread(target=log_tailer, daemon=True).start()
    # 先跑一轮同步采样，保证首帧就有数字
    S.hw = sample_hardware()
    S.counters = parse_prometheus(http_text(f"http://{NINFER_HOST}:{NINFER_PORT}/metrics"))
    try:
        j = json.loads(http_text(f"http://{NINFER_HOST}:{NINFER_PORT}/v1/models", 3.0) or "{}")
        for m in (j.get("data") or []):
            v = m.get("n_ctx") or m.get("max_model_len")
            if v:
                S.n_ctx = int(v)
                break
    except Exception:
        pass
    threading.Thread(target=sampler, daemon=True).start()
    srv = ThreadingHTTPServer((BIND, PORT), Handler)
    print(f"ninfer-monitor: http://{BIND}:{PORT}/  (ninfer {NINFER_HOST}:{NINFER_PORT}, GPU {GPU_INDEX}, disk {DISK_DEV})",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
