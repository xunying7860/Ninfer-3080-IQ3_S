#!/usr/bin/env python3
# 大海捞针（needle-in-a-haystack）：既有测速口径之一。
# 两档：~32K 与 ~196K 题面；针插在 50% 深度，唯一暗号 BLUE-FALCON-7419。
# 判据：回答里出现暗号 = 命中；同时抄引擎控制台的 prompt 数 / TTFT / prefill / decode。
import json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18083
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b-dflash2"
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
    body = {"model": MODEL, "messages": [{"role": "user", "content":
            text + "\n\nQuestion: what is the secret access code mentioned above? "
                   "Answer with the code only, nothing else."}],
            "max_tokens": 64, "temperature": 0}
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


print("=== 大海捞针：32K 档 ===")
ask(build(1550), "32K")
print("=== 大海捞针：196K 档 ===")
ask(build(9300), "196K")
