#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ninfer 引擎看门狗（2026-10-04 定案 B）
=====================================

为什么需要它
------------
ninfer 有一种「活锁」：当某请求的 prompt_tokens + max_tokens 超过 KV 池 token 数时，
输出租约拿不到 KV 页，引擎**不报错**而是静默空转 ——

    GPU 100% 占用, 但功耗只有 ~96 W（真干活是 250~300 W）, 显存带宽 0%,
    一个 host 线程 ~93% CPU 空转, 指标计数器永不推进, 日志每 5 秒重复
    `throughput | 5.0s | running 1 (prefill 1) | host 0.0% (0 us)`。

systemd 的 `Restart=on-failure` 抓不到它（进程没崩、端口在听、/health 还返回 ok）。

判定方式：**指纹 + 计数停滞**，不发探针请求
------------------------------------------
刻意不做探针生成请求：探针会污染测速面板统计（prompt_tokens 跳变、最近一行被占）并占用
引擎 KV 状态。改用只读指纹：

    in_flight > 0  且  util >= 90%  且  power < 130W  且  计数器不推进   持续 >= 180s

三个量互相独立（util 高、功耗却低、计数器不动），合法的长 prefill 功耗在 250~300 W，
不会误判。命中即 `systemctl restart ninfer`，并记录前后证据。

兜底：/health 连续 >= 360s 拿不到 `status:ok` 也重启（覆盖其它死法，例如加载卡住）。

宽限：引擎 `engine ready` 后 360s 内不判定（加载期本来就慢，NCQ 关闭后 ~3 分钟）；
     每次重启后再等 600s 冷却，避免连环重启。

用法
----
    python3 engine-watchdog.py                 # 常驻循环（systemd 用）
    python3 engine-watchdog.py --dry-run       # 只记录不重启（试运行）
    python3 engine-watchdog.py --selftest      # 纯逻辑自检（不碰 GPU/引擎）
    python3 engine-watchdog.py --once          # 跑一轮就退出（人工排查用）
"""

import json
import os
import re
import subprocess
import sys
import time

# ── 目标与常量 ────────────────────────────────────────────────────────────────
SERVICE = "ninfer"
GPU = 0
PORT = 18082
ENGINE_LOG = "/data/ninfer/logs/ninfer-serve.log"      # 只读
WD_LOG = "/data/ninfer/logs/watchdog.log"
METRICS = "http://127.0.0.1:%d/metrics" % PORT
HEALTH = "http://127.0.0.1:%d/health" % PORT

SPIN_UTIL_MIN = 90          # 空转签名：占用 >= 此值
SPIN_POWER_MAX = 130.0      # 空转签名：功耗 <  此值（真干活 250~300W）
SPIN_HOLD_S = 60            # 指纹需持续这么久才动手（采样间隔就是 60s ⇒ 首次命中即触发）
                            # 为什么 60s 够安全：真干活的 prefill 是 250~300W，只有活锁才会
                            # 「100% 占用 + ~96W + 显存带宽 0 + 计数停滞」四者同时成立；
                            # 计数一旦推进（flat=False）就不算命中，所以不会误杀正常请求。
HEALTH_FAIL_S = 360         # /health 连续失败这么久也重启
BOOT_GRACE_S = 360          # engine ready 后的宽限期
COOLDOWN_S = 600            # 重启后的冷却期
INTERVAL = 60               # 采样间隔
HEARTBEAT_S = 600           # 一切正常时的心跳日志周期（避免日志刷屏）


def log(msg, force=True):
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    if force:
        try:
            with open(WD_LOG, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass


def sh(cmd, timeout=20):
    """执行命令，失败/超时一律返回空串（看门狗自身绝不因为一次失败而挂掉）"""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "").strip()
    except Exception:
        return ""


# ── 采样 ─────────────────────────────────────────────────────────────────────
def sample_gpu():
    """返回 (util%, powerW)，取不到返回 (None, None)"""
    out = sh(["nvidia-smi", "-i", str(GPU),
              "--query-gpu=utilization.gpu,power.draw",
              "--format=csv,noheader,nounits"])
    try:
        u, p = [x.strip() for x in out.split(",")[:2]]
        return float(u), float(p)
    except Exception:
        return None, None


def sample_counters():
    """返回 (prompt_tokens_total, tokens_predicted_total)；取不到返回 None"""
    out = sh(["curl", "-s", "-m", "5", METRICS])
    if not out:
        return None
    vals = {}
    for key in ("prompt_tokens_total", "tokens_predicted_total"):
        m = re.search(r"^llamacpp:%s\s+(\d+)" % key, out, re.M)
        if m:
            vals[key] = int(m.group(1))
    if "prompt_tokens_total" not in vals:
        return None
    return (vals.get("prompt_tokens_total"), vals.get("tokens_predicted_total"))


def kv_usage():
    """KV 池占用率（0~1）。活锁时读这个值能直接判「是不是池被占满导致租约拿不到」"""
    out = sh(["curl", "-s", "-m", "5", METRICS])
    m = re.search(r"^llamacpp:kv_cache_usage_ratio\s+([0-9.]+)", out or "", re.M)
    return float(m.group(1)) if m else None


def health_ok():
    out = sh(["curl", "-s", "-m", "5", HEALTH])
    return '"status":"ok"' in (out or "").replace(" ", "")


def in_flight():
    """读引擎日志末段的最近一条 throughput 行，判断是否有请求在跑"""
    out = sh(["bash", "-lc", "tail -c 8000 %s" % ENGINE_LOG])
    runs = re.findall(r"running (\d+)", out)
    return int(runs[-1]) > 0 if runs else False


def engine_ready_age():
    """距最近一次 'engine ready' 的秒数（None = 找不到该行）"""
    out = sh(["bash", "-lc", "tail -n 4000 %s | grep -a 'engine ready' | tail -1" % ENGINE_LOG])
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", out or "")
    if not m:
        return None
    try:
        return time.time() - time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
    except Exception:
        return None


def loading_in_progress():
    """
    最近一条引擎日志是否处于「启动/加载」流程。

    必须单独判这一步：加载期（NCQ 关闭后动辄 3~4 分钟）`/health` 必然返回
    `model_loading`（非 ok），而 `engine_ready_age()` 此刻读到的是**上一次** engine ready 的时间
    ⇒ 只看 age 会让加载期被计入 health 失败 ⇒ **误杀正在正常加载的引擎**（实测踩过一次）。
    空转（活锁）时末行是 throughput 行，不会命中这里的模式 ⇒ 不影响主判定。
    """
    out = sh(["bash", "-lc", "tail -n 8 %s" % ENGINE_LOG])
    lines = [l for l in (out or "").strip().splitlines() if l.strip()]
    if not lines:
        return False
    return bool(re.search(r"loading weights|starting engine|CUDA graphs|calibrat|"
                          r"reading the model|preparing|server stopped", lines[-1]))


# ── 判定（纯函数，便于 selftest）──────────────────────────────────────────────
def verdict(util, power, counters, prev_counters, in_flight_flag, health_ok_flag,
            spin_hold, health_fail, grace):
    """
    输入一次采样 + 累积量，输出 (动作, 新 spin_hold, 新 health_fail)
    动作 ∈ {"ok", "grace", "restart-spin", "restart-health"}
    """
    flat = (counters is not None and prev_counters is not None and counters == prev_counters)
    spinning = bool(in_flight_flag and util is not None and util >= SPIN_UTIL_MIN
                    and power is not None and power < SPIN_POWER_MAX and flat)
    spin_hold = spin_hold + INTERVAL if spinning else 0
    health_fail = 0 if health_ok_flag else health_fail + INTERVAL

    if grace:
        return "grace", spin_hold, health_fail
    if spin_hold >= SPIN_HOLD_S:
        return "restart-spin", 0, health_fail
    if health_fail >= HEALTH_FAIL_S:
        return "restart-health", spin_hold, 0
    return "ok", spin_hold, health_fail


def do_restart(reason, evidence, dry):
    log("!! 判定需要重启 [%s] 证据: %s" % (reason, evidence))
    if dry:
        log("   [dry-run] 跳过实际重启（本应执行: systemctl restart %s）" % SERVICE)
        return
    t0 = time.time()
    out = sh(["systemctl", "restart", SERVICE], timeout=180)
    log("   systemctl restart %s 已执行（%.1fs）%s" % (SERVICE, time.time() - t0, out[:120]))


# ── 自检：不碰 GPU/引擎，只验判定逻辑 ────────────────────────────────────────
def selftest():
    cases = [
        # 名称, util, power, 本次计数, 上次计数, in_flight, health_ok, 已累积hold, 已累积fail, 宽限, 期望动作
        ("空闲（0% 占用）",            0,  88.0, (100, 10), (100, 10), False, True,  0,   0, False, "ok"),
        ("真干活（高功耗）",          100, 288.0, (200, 20), (100, 10), True,  True,  0,   0, False, "ok"),
        ("真干活但计数恰未刷新",      100, 280.0, (100, 10), (100, 10), True,  True,  120, 0, False, "ok"),
        ("空转指纹首采即到 60s",      100, 96.5, (100, 10), (100, 10), True,  True,  0,   0, False, "restart-spin"),
        ("低功耗但在飞且计数在动",    100, 96.5, (200, 20), (100, 10), True,  True,  0,   0, False, "ok"),
        ("加载期（宽限）",            100, 300.0, (0, 0),  (0, 0),   True,  False, 600, 600, True,  "grace"),
        ("/health 长时间失败",        0,  90.0, (100, 10), (99, 9),  False, False, 0,   300, False, "restart-health"),
        ("在飞但功耗正常（长 prefill）", 100, 265.0, (100, 10), (100, 10), True, True, 900, 0, False, "ok"),
    ]
    bad = 0
    for (name, u, p, ct, pct, fl, hok, hold, hf, gr, want) in cases:
        got, _, _ = verdict(u, p, ct, pct, fl, hok, hold, hf, gr)
        flag = "PASS" if got == want else "FAIL"
        if got != want:
            bad += 1
        print("  [%s] %-26s 期望=%-15s 实际=%s" % (flag, name, want, got))
    print("自检结果: %d/%d 通过" % (len(cases) - bad, len(cases)))
    return 1 if bad else 0


# ── 主循环 ───────────────────────────────────────────────────────────────────
def main():
    dry = "--dry-run" in sys.argv
    once = "--once" in sys.argv
    log("看门狗启动 dry_run=%s service=%s 间隔=%ds 指纹=in-flight & util>=%d%% & power<%.0fW & 计数停滞 >=%ds"
        % (dry, SERVICE, INTERVAL, SPIN_UTIL_MIN, SPIN_POWER_MAX, SPIN_HOLD_S))

    prev_counters = None
    spin_hold = 0
    health_fail = 0
    last_restart = 0.0
    last_beat = 0.0

    while True:
        try:
            util, power = sample_gpu()
            counters = sample_counters()
            hok = health_ok()
            kv = kv_usage()
            fl = in_flight()
            age = engine_ready_age()
            loading = loading_in_progress()
            grace = ((age is not None and age < BOOT_GRACE_S)
                     or (time.time() - last_restart < COOLDOWN_S)
                     or loading)

            action, spin_hold, health_fail = verdict(
                util, power, counters, prev_counters, fl, hok, spin_hold, health_fail, grace)
            if counters is not None:
                prev_counters = counters

            ev = ("util=%s power=%s in_flight=%s kv_usage=%s counters=%s flat_hold=%ds "
                  "health_ok=%s loading=%s ready_age=%s"
                  % (util, power, fl, kv, counters, spin_hold, hok, loading,
                     "n/a" if age is None else int(age)))

            if action == "restart-spin":
                do_restart("活锁指纹（空转 + 计数停滞）", ev, dry)
                last_restart = time.time()
            elif action == "restart-health":
                do_restart("/health 长时间非 ok", ev, dry)
                last_restart = time.time()
            elif action == "grace":
                if time.time() - last_beat >= HEARTBEAT_S:
                    log("… 宽限期（加载/重启后），跳过判定 | %s" % ev, force=False)
                    last_beat = time.time()
            else:
                if time.time() - last_beat >= HEARTBEAT_S:
                    log("ok | %s" % ev, force=False)
                    last_beat = time.time()
        except Exception as exc:      # 看门狗自身绝不能挂
            log("循环异常（忽略并继续）: %s: %s" % (type(exc).__name__, exc))

        if once:
            return 0
        time.sleep(INTERVAL)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main())
