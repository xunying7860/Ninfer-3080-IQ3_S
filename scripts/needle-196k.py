#!/usr/bin/env python3
# 大海捞针（三档）：与 /data/ninfer/needle-test.py 同一题面口径，唯一差别是
#   —— 每个请求显式带 reasoning_effort=none + chat_template_kwargs.enable_thinking=false。
# 为什么必须这样：生产是 --default-reasoning-effort max，而 stock needle-test.py 只给 max_tokens=64，
#   64 个 token 全被 xhigh 思考吃掉（引擎日志 finish=output limit、content 为空）⇒ 针答不出来。
#   这里是**客户端**改动，生产参数一字未动。
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18082
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b-gsq-rco-iq3s"
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT
NEEDLE = "IMPORTANT MEMO: the secret access code is BLUE-FALCON-7419, do not share it."


def build(n_lines):
    lines = []
    for i in range(n_lines):
        if i == n_lines // 2:
            lines.append("Line %06d: %s" % (i, NEEDLE))
        else:
            lines.append("Line %06d: routine maintenance notes, inventory records and log entries follow." % i)
    return "\n".join(lines)


def ask(text, label):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content":
                          text + "\n\nQuestion: what is the secret access code mentioned above? "
                                 "Answer with the code only, nothing else."}],
            "max_tokens": 64, "temperature": 0,
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=2400) as r:
            d = json.load(r)
        el = time.time() - t
        ch = d["choices"][0]
        c = (ch["message"].get("content") or "").strip()
        pt = d.get("usage", {}).get("prompt_tokens")
        print("[%s] prompt_tokens=%s 回答=%r 命中=%s 墙钟=%.2fs"
              % (label, pt, c[:120], "BLUE-FALCON-7419" in c, el))
        return pt, el
    except Exception as e:
        print("[%s] 失败: %r" % (label, e))
        return None, None


for lines, label in ((9300, "196K"),):
    print("=== 大海捞针：%s 档 ===" % label)
    ask(build(lines), label)
