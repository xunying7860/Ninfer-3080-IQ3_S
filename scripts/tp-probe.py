#!/usr/bin/env python3
# 吞吐/复用/检索 三合一探针（x99 本机跑）
# 用法: tp-probe.py <port> <model-id> [arm_label]
#  - 第 1 段：数数字语料（decode 友好）3 轮 → 看 decode tok/s 与第 2/3 轮 cache% / TTFT
#  - 第 2 段：~32K 填充 + 中段针 → 看检索（环启用时才有意义）
# 权威读数在引擎自己的日志行（req#N done / capacity / prompt ... tokens），本脚本只给客户端侧墙钟与 usage。
import json, sys, time, urllib.request, urllib.error

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18084
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b-kvmem"
ARM = sys.argv[3] if len(sys.argv) > 3 else "arm"
BASE = "http://127.0.0.1:%d" % PORT
URL = BASE + "/v1/chat/completions"


def post(body, timeout=2400):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return time.time() - t, json.load(r), None
    except urllib.error.HTTPError as e:
        return time.time() - t, None, "HTTP %s: %s" % (e.code, e.read()[:300])
    except Exception as e:
        return time.time() - t, None, repr(e)


def one(label, body):
    el, d, err = post(body)
    if err:
        print("[%s/%s] 失败: %s" % (ARM, label, err)); return
    u = d.get("usage", {})
    ch = d["choices"][0]
    c = (ch["message"].get("content") or "").strip()
    rc = (ch["message"].get("reasoning_content") or "")
    print("[%s/%s] 墙钟=%.2fs prompt=%s out=%s finish=%s content[:60]=%r" %
          (ARM, label, el, u.get("prompt_tokens"), u.get("completion_tokens"),
           ch.get("finish_reason"), c[:60]))
    if rc and not c:
        print("        (全进 reasoning，%d 字符)" % len(rc))


# ---------- 段 1：数数字（decode 口径，关思考） ----------
nums = " ".join(str(i) for i in range(1, 301))
ask = nums + "\n\nContinue the same number sequence from the next number. Output numbers only, separated by spaces."
body = {"model": MODEL, "messages": [{"role": "user", "content": ask}],
        "max_tokens": 400, "temperature": 0,
        "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
print("=== 段1 数数字 decode（同题面 3 轮，看 cache%/TTFT）===")
for i in range(3):
    one("nums-r%d" % (i + 1), body)
    time.sleep(2)

# ---------- 段 2：~32K 填充 + 中段针 ----------
print("=== 段2 32K 填充 + 中段针（检索判据）===")
NEEDLE = "IMPORTANT MEMO: the secret access code is BLUE-FALCON-7419, do not share it."
N = 1550
lines = []
for i in range(N):
    lines.append("Line %06d: %s" % (i, NEEDLE) if i == N // 2
                 else "Line %06d: routine maintenance notes, inventory records and log entries follow." % i)
text = "\n".join(lines) + "\n\nQuestion: what is the secret access code mentioned above? Answer with the code only."
one("needle-32K", {"model": MODEL, "messages": [{"role": "user", "content": text}],
                   "max_tokens": 64, "temperature": 0,
                   "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}})

# ---------- 段 3：追问一次（同一 32K 题面 + 追加一句）→ 环的复用判据 ----------
print("=== 段3 32K 追问（前缀复用判据）===")
one("needle-32K-followup", {"model": MODEL, "messages": [
    {"role": "user", "content": text},
    {"role": "assistant", "content": "BLUE-FALCON-7419"},
    {"role": "user", "content": "Repeat the code you just gave."}],
    "max_tokens": 64, "temperature": 0,
    "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}})
print("=== 段结束：权威数字看引擎日志 req#N done 行 ===")
