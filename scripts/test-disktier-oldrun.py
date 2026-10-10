#!/usr/bin/env python3
# 验证旧引擎磁盘层：同题面 → 重启引擎 → 再发同一题面，看第二发的 cache% 是否来自磁盘层
# 判据：N1（首问）cache 应为 0；N2（重启后）若 cache 高（>50%）⇒ 磁盘层恢复生效 ✓
import json
import re
import subprocess
import sys
import time
import urllib.request

PORT = 18082
MODEL = "qwen3.8-27b"
LOG = "/data/ninfer/logs/ninfer-serve.log"
DOC = 1200          # 行数（≈25K token）
NEEDLE = "IMPORTANT MEMO: the vault access code is BLUE-FALCON-7419, keep it secret."


def doc():
    return "\n".join("Line %06d: %s" % (i, NEEDLE) if i == DOC // 2
                     else "Line %06d: routine maintenance notes, inventory records and log entries follow." % i
                     for i in range(DOC))


def ask(tag):
    b = {"model": MODEL, "messages": [{"role": "user", "content": doc() +
         "\n\nQuestion: what is the vault access code? Answer the code only."}],
         "max_tokens": 32, "temperature": 1, "top_k": 20, "presence_penalty": 0,
         "frequency_penalty": 0, "reasoning_effort": "none",
         "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                                 data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    el = time.time() - t
    u = d.get("usage", {})
    c = (d["choices"][0]["message"].get("content") or "").strip()
    # 从引擎日志里取这次请求的权威行（cache% / prefill）
    line = ""
    try:
        for ln in reversed(open(LOG, errors="replace").readlines()[-400:]):
            if "req#" in ln and "done" in ln and ("prompt %s" % format(u.get("prompt_tokens", 0), ",")) in ln:
                line = ln.strip()
                break
    except Exception as e:
        line = "(读日志失败 %s)" % e
    m = re.search(r"cache ([\d,]+) \(([\d.]+)%, ([a-z ]+)\)", line)
    print("[%s] %.1fs prompt=%s out=%s 命中暗号=%s" % (tag, el, u.get("prompt_tokens"), u.get("completion_tokens"),
                                                      "BLUE-FALCON-7419" in c))
    print("    引擎行: %s" % (line[:190] if line else "(未找到)"))
    print("    cache: %s%% (%s)" % (m.group(2), m.group(3)) if m else "    cache: (解析不到)")
    return m.group(2) if m else None


print("=== 第一发（会话冷启，cache 应为 0）===")
c1 = ask("N1-冷")
print()
print("=== 重启引擎（磁盘层应把常驻 continuation 落盘）===")
t0 = time.time()
subprocess.run(["sudo", "-n", "systemctl", "restart", "ninfer.service"], check=True)
for _ in range(60):
    time.sleep(2)
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % PORT, timeout=3) as r:
            if r.status == 200:
                break
    except Exception:
        pass
print("重启到 models=200：%.0f 秒" % (time.time() - t0))
print()
print("=== 第二发（同题面；若磁盘层生效 ⇒ cache 高、TTFT 小）===")
c2 = ask("N2-重启后")
print()
print("=== 结论 ===")
print("第一发 cache = %s%% ；第二发 cache = %s%%" % (c1, c2))
try:
    print("判据：第二发 cache 明显 >0（尤其 >50%%）⇒ 磁盘层恢复生效 ✓；若 ≈0 ⇒ 未生效 ✗")
except Exception:
    pass
