#!/usr/bin/env python3
# 512K 档回归 v2（带超时护栏；风险档放最后，便于定位悬崖点）
#   段A 触发条件（§1.2）：max_tokens=131072 + 65K 题面 + tools
#   段B 多轮快照压测：10 轮（turn-closure 快照 / host-backed publish）
#   段C 中段针阶梯：400/800/1600/3100/4500 行（≈8K/17K/34K/65K/94K）
#   段D 悬崖阶梯（每档 240s 超时；超时=卡死，停止后续档）：5450(114K)/6100(128K)/7100(149K)
# 用法: verify-512k2.py <port> <model> <tag>
import json, sys, time, urllib.request, urllib.error

PORT = int(sys.argv[1]); MODEL = sys.argv[2]; TAG = sys.argv[3]
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT
NEEDLE = "IMPORTANT MEMO: the secret access code is BLUE-FALCON-7419, do not share it."
BASE = {"model": MODEL, "temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0,
        "presence_penalty": 0, "repetition_penalty": 1,
        "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "run",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
ISSUES = []


def post(body, timeout):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return time.time() - t, json.load(r), None
    except urllib.error.HTTPError as e:
        return time.time() - t, None, "HTTP %s" % e.code
    except Exception as e:
        return time.time() - t, None, type(e).__name__


def doc(n):
    return "\n".join("Line %06d: %s" % (i, NEEDLE) if i == n // 2
                     else "Line %06d: routine maintenance notes, inventory records and log entries follow." % i
                     for i in range(n))


def run(tag, body, timeout=600, needle=True):
    el, d, err = post(body, timeout)
    if err:
        print("[%s/%s] %.1fs  ** %s **" % (TAG, tag, el, err), flush=True)
        ISSUES.append("%s:%s" % (tag, err)); return None
    u = d.get("usage", {}); ch = d["choices"][0]; c = ch["message"].get("content") or ""
    hit = "BLUE-FALCON-7419" in c
    flag = ""
    if needle and not hit:
        flag = " ← 针未命中（§1.1 风险）"; ISSUES.append(tag + ":miss")
    print("[%s/%s] %.1fs prompt=%s out=%s finish=%s 命中针=%s%s content[:50]=%r" %
          (TAG, tag, el, u.get("prompt_tokens"), u.get("completion_tokens"),
           ch.get("finish_reason"), (hit if needle else "n/a"), flag, c.strip()[:50]), flush=True)
    return d


print("===== 段A 触发条件：max_tokens=131072 + 65K + tools（12:49 就是这条崩的）=====", flush=True)
b = dict(BASE); b["max_tokens"] = 131072; b["tools"] = TOOLS
b["messages"] = [{"role": "user", "content": doc(3100) + "\n\nQuestion: how many lines does the document have? Answer with the number only."}]
run("A1-maxtok131072-65K-tools", b, needle=False)

print("===== 段B 多轮快照压测（10 轮 turn-closure）=====", flush=True)
convo = [{"role": "user", "content": doc(3100) + "\n\nQuestion: how many lines does the document have? Answer with the number only."}]
for turn in range(1, 11):
    q = dict(BASE); q["messages"] = convo[:]; q["max_tokens"] = 128
    if turn == 1: q["tools"] = TOOLS
    d = run("B-turn%02d" % turn, q, 300, needle=False)
    if d is None: break
    convo.append({"role": "assistant", "content": (d["choices"][0]["message"].get("content") or "")[:200]})
    convo.append({"role": "user", "content": "Turn %d: what is the secret access code? Answer with the code only." % turn})

print("===== 段C 中段针阶梯（≤94K）=====", flush=True)
for n in (400, 800, 1600, 3100, 4500):
    q = dict(BASE); q["max_tokens"] = 64
    q["messages"] = [{"role": "user", "content": doc(n) + "\n\nQuestion: what is the secret access code mentioned above? Answer with the code only."}]
    run("C-needle-%dL(~%dKtok)" % (n, round(n * 21 / 1000)), q, 600)

print("===== 段D 悬崖阶梯（240s 超时；超时即判定卡死并停止）=====", flush=True)
for n, name in ((5450, "114K"), (6100, "128K"), (7100, "149K>池")):
    q = dict(BASE); q["max_tokens"] = 64
    q["messages"] = [{"role": "user", "content": doc(n) + "\n\nQuestion: what is the secret access code mentioned above? Answer with the code only."}]
    d = run("D-%dL(%s)" % (n, name), q, 240)
    if d is None and ISSUES and ISSUES[-1].endswith("TimeoutError"):
        print("  → 该档超时（卡死），停止后续档", flush=True)
        break

print("===== 汇总 =====", flush=True)
print("ISSUES = %s" % (ISSUES if ISSUES else "无"), flush=True)
