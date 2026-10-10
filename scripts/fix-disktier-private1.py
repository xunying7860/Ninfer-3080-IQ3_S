#!/usr/bin/env python3
# 方案(a)：--max-private-continuations 1  ⇒ 只要出现第二个会话，正在用的那条立刻被挤出并落盘
# 然后按判据验证：P(36K) → Q(小) → 重启 → 回 P ⇒ 看 cache%（高 = 生效）
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request

RUN = "/data/ninfer/run.sh"
LOG = "/data/ninfer/logs/ninfer-serve.log"
SYS = "You are a careful assistant. Answer briefly and exactly."


def doc(n, marker):
    return "\n".join("Line %06d: %s" % (i, "MARK-%s at %d" % (marker, i) if i == n // 2
                     else "Line %06d: routine maintenance notes, inventory records and log entries follow." % i)
                     for i in range(n))


def ask(tag, text, timeout=1800):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "system", "content": SYS},
            {"role": "user", "content": text + "\n\nQuestion: how many lines? Answer the number only."}],
            "max_tokens": 12, "temperature": 1, "top_k": 20, "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:18082/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    el = time.time() - t
    pt = d.get("usage", {}).get("prompt_tokens")
    time.sleep(0.6)
    line = ""
    for ln in reversed(open(LOG, errors="replace").readlines()[-400:]):
        if "req#" in ln and "done" in ln and format(pt, ",") in ln:
            line = ln.strip()
            break
    m = re.search(r"cache ([\d,]+) \(([\d.]+)%, ([a-z ]+)\)", line)
    print("  [%s] %6.1fs prompt=%s cache=%s (%s%%, %s)" % (
        tag, el, pt, m.group(1) if m else "?", m.group(2) if m else "?", m.group(3).strip() if m else "?"), flush=True)
    return m.group(2) if m else None


# ---------- 1) 改 run.sh（幂等）----------
s = open(RUN, encoding="utf-8").read()
pid = subprocess.run("systemctl show -p MainPID --value ninfer.service", shell=True,
                     capture_output=True, text=True).stdout.strip()
cur = open("/proc/%s/cmdline" % pid, "rb").read().replace(b"\0", b"\n").decode("utf-8", "replace")
now = re.search(r"--max-private-continuations\s+(\d+)", cur)
print("改动前 argv 里 --max-private-continuations =", now.group(1) if now else "(未设置，取默认 2)", flush=True)
if "--max-private-continuations 1 " in s or "--max-private-continuations 1\\" in s or re.search(r"--max-private-continuations\s+1(\s|\\)", s):
    print("run.sh 已经是 1，不动", flush=True)
else:
    shutil.copy2(RUN, RUN + ".bak-%s-pre-private1" % time.strftime("%Y%m%d-%H%M%S"))
    if re.search(r"--max-private-continuations\s+\d+", s):
        s = re.sub(r"--max-private-continuations\s+\d+", "--max-private-continuations 1", s)
    else:
        # 插到 --host-state-slots 那行后面（保持续行风格：行尾反斜杠、不写行内注释）
        m = re.search(r"([^\n]*--max-pending-requests\s+\d+[^\n]*?)(\s*\\)\n", s)
        if not m:
            m = re.search(r"([^\n]*--max-concurrency\s+\d+[^\n]*?)(\s*\\)\n", s)
        if not m:
            print("找不到插入锚点（--max-pending-requests / --max-concurrency），放弃改文件", flush=True)
            print("--- run.sh 里相关行（诊断）", flush=True)
            for ln in s.splitlines():
                if any(k in ln for k in ("max-concurrency", "max-pending", "state-slots", "private")):
                    print("   " + ln, flush=True)
            sys.exit(1)
        s = s[:m.end(1)] + " --max-private-continuations 1" + s[m.end(1):]
    open(RUN, "w", encoding="utf-8").write(s)
    print("已改 run.sh", flush=True)
subprocess.run(["bash", "-n", RUN], check=True)
print("bash -n OK", flush=True)

# ---------- 2) 重启 ----------
t0 = time.time()
subprocess.run(["sudo", "-n", "systemctl", "restart", "ninfer.service"], check=True)
for _ in range(300):
    time.sleep(1)
    try:
        with urllib.request.urlopen("http://127.0.0.1:18082/v1/models", timeout=3) as r:
            if r.status == 200:
                break
    except Exception:
        pass
print("重启到就绪 %.0f 秒" % (time.time() - t0), flush=True)

# ---------- 3) 判据 ----------
P = doc(1200, "P2")
print("### 发 P（大）", flush=True)
ask("P-第1发", P)
print("### 发一个别的会话 Q（应把 P 挤出并落盘）", flush=True)
ask("Q", doc(160, "Qx"))
print("### 磁盘实际占用：", subprocess.run(["du", "-sh", "/data/nvme/ninfer-disktier"],
                                            capture_output=True, text=True).stdout.strip(), flush=True)
print("### 重启", flush=True)
t0 = time.time()
subprocess.run(["sudo", "-n", "systemctl", "restart", "ninfer.service"], check=True)
for _ in range(300):
    time.sleep(1)
    try:
        with urllib.request.urlopen("http://127.0.0.1:18082/v1/models", timeout=3) as r:
            if r.status == 200:
                break
    except Exception:
        pass
print("重启到就绪 %.0f 秒" % (time.time() - t0), flush=True)
print("### 回 P（判据：高命中 = 修好）", flush=True)
ask("P-重启后", P)
print("=== 完 ===", flush=True)
