#!/usr/bin/env python3
# A) ninfer 磁盘层有效性：A/B/C 三会话把 A 挤到磁盘 → 重启 → 再发 A ⇒ 看 cache%（高=磁盘层真能跨重启）
# B) strata 会话文件是否跨重启：同一题面 → 重启 strata → 再发 ⇒ 看 reused/read
import json
import re
import subprocess
import sys
import time
import urllib.request

NLOG = "/data/ninfer/logs/ninfer-serve.log"
SLOG = "/data/workspace/strata01403/strata-iq3_xxs.1card.log"


def mkdoc(n, tag):
    return "\n".join("Line %06d: %s" % (i, "SENTINEL-%s line %d" % (tag, i) if i == n // 2
                     else "Line %06d: routine maintenance notes, inventory records and log entries follow." % i)
                     for i in range(n))


def ninfer(tag, n, port=18082, model="qwen3.8-27b"):
    b = {"model": model, "messages": [{"role": "user", "content": mkdoc(n, tag) +
         "\n\nQuestion: how many lines? Answer the number only."}], "max_tokens": 32,
         "temperature": 1, "top_k": 20, "reasoning_effort": "none",
         "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    el = time.time() - t
    pt = d.get("usage", {}).get("prompt_tokens")
    line = ""
    for ln in reversed(open(NLOG, errors="replace").readlines()[-600:]):
        if "req#" in ln and "done" in ln and ("prompt %s" % format(pt, ",")) in ln:
            line = ln.strip()
            break
    m = re.search(r"cache ([\d,]+) \(([\d.]+)%, ([a-z ]+)\)", line)
    print("[ninfer %s] %.1fs prompt=%s cache=%s%% (%s)" % (tag, el, pt, m.group(2) if m else "?", m.group(3).strip() if m else line[:60]))
    return float(m.group(2)) if m else None


def strata(tag, n, port=8080, model="qwen3.8-flash-next-iq3_xxs"):
    b = {"model": model, "messages": [{"role": "user", "content": mkdoc(n, tag) +
         "\n\nQuestion: how many lines? Answer the number only."}], "max_tokens": 32,
         "temperature": 1, "top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    el = time.time() - t
    pt = d.get("usage", {}).get("prompt_tokens")
    time.sleep(0.6)
    line = ""
    for ln in reversed(open(SLOG, errors="replace").readlines()[-400:]):
        if "tokens =" in ln and "generated in" in ln:
            line = ln.strip()
            break
    print("[strata %s] %.1fs prompt=%s  引擎行: %s" % (tag, el, pt, line[:150]))
    return line


def wait(port, secs=90):
    for _ in range(secs * 2):
        time.sleep(2)
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % port, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
    return False


print("############ A) ninfer 磁盘层：A/B/C 挤出 → 重启 → 回 A ############")
print("— 发 A（第 1 个会话）")
ninfer("A", 1200)
print("— 发 B、C（把 A 挤出私有目录：max-private-continuations 默认 2）")
ninfer("B", 1200)
ninfer("C", 1200)
print("— 磁盘层目录大小：")
print(subprocess.run(["du", "-sh", "/data/nvme/ninfer-disktier"], capture_output=True, text=True).stdout.strip())
print("— 重启 ninfer")
subprocess.run(["sudo", "-n", "systemctl", "restart", "ninfer.service"], check=True)
wait(18082)
print("— 回 A（同题面）：cache 高 ⇒ 磁盘层跨重启生效 ✓")
ca = ninfer("A-重启后", 1200)

print()
print("############ B) strata 会话文件是否跨重启 ############")
print("— 发一次（建立会话）")
strata("S1", 1200)
print("— 重启 strata（约 1 分钟）")
t0 = time.time()
subprocess.run(["sudo", "-n", "systemctl", "restart", "strata"], check=True)
wait(8080, 150)
print("  重启到就绪 %.0f 秒" % (time.time() - t0))
print("— 再发同题面：引擎行的 reused 高 ⇒ 会话文件跨重启生效 ✓")
strata("S1-重启后", 1200)
