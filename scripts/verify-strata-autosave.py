#!/usr/bin/env python3
# ★ strata 重启不丢上下文 —— 判据实测
# 稳定题面（带 system prompt，文档与 tag 解耦）→ 重启 strata（触发停机 SAVE + 启动 RESTORE）→ 回同一题面
# 判据：重启后同题面 reused ≈ 全长（>90%）= 修好；0 = 没修好
import json
import subprocess
import time
import urllib.request

SLOG = "/data/workspace/strata01403/strata-iq3_xxs.1card.log"
SYS = "You are a careful assistant. Answer briefly and exactly."


def doc(n=1200):
    return "\n".join("Line %06d: %s" % (i, "DOC MARKER %d" % i if i == n // 2
                     else "Line %06d: routine maintenance notes, inventory records and log entries follow." % i)
                     for i in range(n))


def ask(tag, timeout=1800):
    body = {"model": "qwen3.8-flash-next-iq3_xxs",
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": doc() + "\n\nQuestion: how many lines? Answer the number only."}],
            "max_tokens": 16, "temperature": 1, "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    el = time.time() - t
    pt = d.get("usage", {}).get("prompt_tokens")
    time.sleep(0.8)
    line = ""
    for ln in reversed(open(SLOG, errors="replace").readlines()[-400:]):
        if "tokens =" in ln and "generated in" in ln:
            line = ln.strip()
            break
    print("[%s] %.1fs prompt=%s\n      %s" % (tag, el, pt, line[:190]), flush=True)
    return line


def wait_ready(secs=400):
    for _ in range(secs):
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
    return False


print("### 1) 发稳定题面（建立常驻会话）", flush=True)
ask("T1-重启前")
print("### 2) 重启 strata（停机应 SAVE、启动应 RESTORE）", flush=True)
t0 = time.time()
subprocess.run(["sudo", "-n", "systemctl", "restart", "strata"], check=True)
ok = wait_ready()
print("    重启到就绪：%.0f 秒，ready=%s" % (time.time() - t0, ok), flush=True)
print("### 3) 停机/启动日志（找 autosaved / restored）", flush=True)
subprocess.run("sudo journalctl -u strata --since '-8 min' --no-pager 2>/dev/null | "
               "grep -aiE 'autosav|restored at startup|restore failed|starting with an empty' | tail -5", shell=True)
print("### 4) 回同一题面 ⇒ 判据 reused", flush=True)
ask("T2-重启后")
print("=== 完 ===", flush=True)
