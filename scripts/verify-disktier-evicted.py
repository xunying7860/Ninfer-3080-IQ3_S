#!/usr/bin/env python3
# ★ 定案实验：磁盘层「重启后复用」到底成不成
# 上游 benchmark 原话：“a 17,444-token prompt **evicted by two others** comes back from disk in 1.2 s
#   instead of 9.6 s, and in 1.0 s **after a restart**” ⇒ 判据必须先让 P 被挤出。
# 我们的 --max-private-continuations = 8 ⇒ 需要 9 个别的会话把 P 挤出去（用小会话省时间）。
# 判据：重启后回 P 的 cache%（高 = 磁盘层生效）。
import json
import re
import subprocess
import time
import urllib.request

LOG = "/data/ninfer/logs/ninfer-serve.log"
SYS = "You are a careful assistant. Answer briefly and exactly."


def doc(n, marker):
    """题面内容只由 marker 决定（同 marker = 完全同一题面 ✓）"""
    return "\n".join("Line %06d: %s" % (i, "MARK-%s at %d" % (marker, i) if i == n // 2
                     else "Line %06d: routine maintenance notes, inventory records and log entries follow." % i)
                     for i in range(n))


def ask(tag, text, max_tokens=12, timeout=1800):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "system", "content": SYS},
                                                 {"role": "user", "content": text + "\n\nQuestion: how many lines? Answer the number only."}],
            "max_tokens": max_tokens, "temperature": 1, "top_k": 20, "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:18082/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    el = time.time() - t
    pt = d.get("usage", {}).get("prompt_tokens")
    time.sleep(0.7)
    line = ""
    for ln in reversed(open(LOG, errors="replace").readlines()[-400:]):
        if "req#" in ln and "done" in ln and format(pt, ",") in ln:
            line = ln.strip()
            break
    m = re.search(r"cache ([\d,]+) \(([\d.]+)%, ([a-z ]+)\)", line)
    print("  [%s] %5.1fs prompt=%s cache=%s (%s%%, %s)" % (
        tag, el, pt, m.group(1) if m else "?", m.group(2) if m else "?", m.group(3).strip() if m else line[-60:]),
        flush=True)
    return float(m.group(2)) if m else None


def disk(tag):
    du = subprocess.run(["du", "-sh", "--apparent-size", "/data/nvme/ninfer-disktier"],
                        capture_output=True, text=True).stdout.split()[0]
    real = subprocess.run(["du", "-sh", "/data/nvme/ninfer-disktier"],
                          capture_output=True, text=True).stdout.split()[0]
    free = subprocess.run(["df", "-h", "/data/nvme"], capture_output=True, text=True).stdout.strip().split("\n")[-1]
    mt = subprocess.run("find /data/nvme/ninfer-disktier -type f -newermt '-20 minutes' | wc -l",
                        shell=True, capture_output=True, text=True).stdout.strip()
    print("  [磁盘 %s] 名义 %s / 实际 %s｜20 分钟内被写过的文件数=%s｜%s" % (tag, du, real, mt, free), flush=True)


def wait_ready(secs=300):
    for _ in range(secs):
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://127.0.0.1:18082/v1/models", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
    return False


print("### 0) 现况", flush=True)
disk("实验前")
print("  private-continuations:", subprocess.run(
    "tr '\\0' '\\n' < /proc/$(systemctl show -p MainPID --value ninfer.service)/cmdline | grep -A1 private-continuations | tail -1",
    shell=True, capture_output=True, text=True).stdout.strip(), flush=True)

P = doc(1200, "P")          # ~36K token
print("### 1) 发 P（先冷后热）", flush=True)
ask("P-第1发", P)
ask("P-第2发", P)           # 应 100% 命中（转轮检查点）

print("### 2) 发 9 个别的会话，把 P 挤出私有目录（上游 benchmark 的做法）", flush=True)
for i in range(1, 10):
    ask("Q%d" % i, doc(160, "Q%d" % i))   # ~5K token 一个，快
disk("挤出后")

print("### 3) 重启 ninfer", flush=True)
t0 = time.time()
subprocess.run(["sudo", "-n", "systemctl", "restart", "ninfer.service"], check=True)
ok = wait_ready()
print("  重启到就绪 %.0f 秒 ready=%s" % (time.time() - t0, ok), flush=True)
print("  重启后引擎日志里的磁盘层相关行：", flush=True)
subprocess.run("grep -aiE 'diskv|disk kv|restore.*disk|disk.*restore' /data/ninfer/logs/ninfer-serve.log | tail -4",
               shell=True)

print("### 4) 回 P（判据）", flush=True)
ask("P-重启后", P)
disk("实验后")
print("=== 完 ===", flush=True)
