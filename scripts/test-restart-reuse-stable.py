#!/usr/bin/env python3
# 修正版：用**稳定前缀**（带 system prompt）重测「重启后复用」
#  A) ninfer 磁盘层：A → B/C（把 A 挤出私有目录）→ 重启 → 回 A ⇒ cache% 高 = 磁盘层生效
#  B) strata 会话文件：S1 → S2/S3 → 重启 strata → 回 S1 ⇒ reused 高 = 会话文件生效
import json
import re
import subprocess
import sys
import time
import urllib.request

NLOG = "/data/ninfer/logs/ninfer-serve.log"
SLOG = "/data/workspace/strata01403/strata-iq3_xxs.1card.log"
SYS = "You are a careful assistant. Answer briefly and exactly."


def mkdoc(n, tag=""):
    """★ 文档内容**不含 tag**（tag 只用于打印）—— 否则"同题面"其实是不同题面，
    前缀对不上，磁盘层/会话文件的"重启后复用"根本测不到（我第一次就踩了这个坑）。"""
    return "\n".join("Line %06d: %s" % (i, "DOC-MARKER %d" % i if i == n // 2
                     else "Line %06d: routine maintenance notes, inventory records and log entries follow." % i)
                     for i in range(n))


def call(port, model, tag, n, log, regex, doc_variant=None):
    body = {"model": model, "messages": [{"role": "system", "content": SYS},
            {"role": "user", "content": mkdoc(n, doc_variant or tag) + "\n\nQuestion: how many lines? Answer the number only."}],
            "max_tokens": 16, "temperature": 1, "top_k": 20, "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    el = time.time() - t
    pt = d.get("usage", {}).get("prompt_tokens")
    time.sleep(0.8)
    hit = ""
    for ln in reversed(open(log, errors="replace").readlines()[-500:]):
        if format(pt, ",") in ln and ("done" in ln or "tokens =" in ln):
            hit = ln.strip()
            break
    print("[%s] %.1fs prompt=%s\n      %s" % (tag, el, pt, hit[:185]))
    return el, hit


def wait(port, secs=200):
    for _ in range(secs):
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % port, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
    return False


print("############ A) ninfer 磁盘层（稳定前缀）############")
call(18082, "qwen3.8-27b", "A1", 1200, NLOG, None, "A")
call(18082, "qwen3.8-27b", "B1", 1200, NLOG, None, "B")
call(18082, "qwen3.8-27b", "C1", 1200, NLOG, None, "C")
print("— 磁盘层目录：", subprocess.run(["du", "-sh", "/data/nvme/ninfer-disktier"],
                                       capture_output=True, text=True).stdout.strip())
print("— 重启 ninfer")
subprocess.run(["sudo", "-n", "systemctl", "restart", "ninfer.service"], check=True)
wait(18082)
print("— 回 A（同题面，前缀稳定 ⇒ 若磁盘层生效应命中）")
call(18082, "qwen3.8-27b", "A2-重启后", 1200, NLOG, None, "A")

print()
print("############ B) strata 会话文件（稳定前缀）############")
call(8080, "qwen3.8-flash-next-iq3_xxs", "S1", 1200, SLOG, None, "S1")
call(8080, "qwen3.8-flash-next-iq3_xxs", "S2", 1200, SLOG, None, "S2")
call(8080, "qwen3.8-flash-next-iq3_xxs", "S3", 1200, SLOG, None, "S3")
print("— 重启 strata")
t0 = time.time()
subprocess.run(["sudo", "-n", "systemctl", "restart", "strata"], check=True)
wait(8080, 200)
print("  重启到就绪 %.0f 秒" % (time.time() - t0))
print("— 回 S1")
call(8080, "qwen3.8-flash-next-iq3_xxs", "S1-重启后", 1200, SLOG, None, "S1")
print("=== 完 ===")
