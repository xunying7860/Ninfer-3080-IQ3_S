#!/usr/bin/env python3
# 512K 档回归测试（x99 本机跑）
#   T1 触发条件回归（§1.2）：显式 max_tokens=131072 + 65K 题面 + tools  ⇒ 必须 200 且不崩
#   T2 快照路径压测：多轮会话 10 轮（每轮 = 一次 turn-closure 快照 / host-backed publish，就是 12:49 崩的那条路径）
#   T3 §1.1 中段针按档扫描：400/800/1600/3100/4500/6100 行（≈8K…128K），针在正中；池 = 131,072 token ≈ 6,240 行
#   T4 超池：7,100 行（≈149K，> 池）⇒ 观察"尽力而为"区行为（可能静默丢针，但不许崩）
# 用法: verify-512k.py <port> <model-id> <标签>
import json, sys, time, urllib.request, urllib.error

PORT = int(sys.argv[1]); MODEL = sys.argv[2]; TAG = sys.argv[3]
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT
NEEDLE = "IMPORTANT MEMO: the secret access code is BLUE-FALCON-7419, do not share it."
BASE = {"model": MODEL, "temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0,
        "presence_penalty": 0, "repetition_penalty": 1,
        "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "run a shell command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
FAILS = []


def post(body, timeout=3600):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return time.time() - t, json.load(r), None
    except urllib.error.HTTPError as e:
        return time.time() - t, None, "HTTP %s: %s" % (e.code, e.read()[:200].decode("utf-8", "replace"))
    except Exception as e:
        return time.time() - t, None, repr(e)


def doc(n, needle_pos=None):
    pos = n // 2 if needle_pos is None else needle_pos
    return "\n".join("Line %06d: %s" % (i, NEEDLE) if i == pos
                     else "Line %06d: routine maintenance notes, inventory records and log entries follow." % i
                     for i in range(n))


def report(tag, el, d, err, needle=True, note=""):
    if err:
        print("[%s/%s] %.1fs FAIL %s %s" % (TAG, tag, el, err, note), flush=True)
        FAILS.append(tag); return None
    u = d.get("usage", {}); ch = d["choices"][0]
    c = ch["message"].get("content") or ""
    hit = "BLUE-FALCON-7419" in c
    if needle and not hit:
        FAILS.append(tag)
    print("[%s/%s] %.1fs prompt=%s out=%s finish=%s 命中针=%s %s content[:60]=%r" %
          (TAG, tag, el, u.get("prompt_tokens"), u.get("completion_tokens"), ch.get("finish_reason"),
           (hit if needle else "n/a"), note, c.strip()[:60]), flush=True)
    return d


print("========== T1 触发条件回归：max_tokens=131072 + 65K 题面 + tools ==========", flush=True)
txt = doc(3100) + "\n\nQuestion: what is the secret access code mentioned above? Answer with the code only."
b = dict(BASE); b["messages"] = [{"role": "user", "content": txt}]; b["max_tokens"] = 131072; b["tools"] = TOOLS
report("t1-maxtok131072-65K-tools", *post(b), note="（12:49 就是这条崩的）")
time.sleep(2)

print("========== T2 快照路径压测：多轮会话 10 轮（turn-closure 快照 / host-backed publish）==========", flush=True)
base_txt = doc(3100)
convo = [{"role": "user", "content": base_txt + "\n\nQuestion: how many lines does the document have? Answer with the number only."}]
for turn in range(1, 11):
    q = dict(BASE); q["messages"] = convo[:]; q["max_tokens"] = 128
    if turn == 1:
        q["tools"] = TOOLS
    el, d, err = post(q)
    d2 = report("turn%02d" % turn, el, d, err, needle=False)
    if d2 is None:
        break
    convo.append({"role": "assistant", "content": (d2["choices"][0]["message"].get("content") or "")[:200]})
    convo.append({"role": "user", "content": "Turn %d: restate the secret access code if you have it, else say UNKNOWN." % turn})
    time.sleep(1)
time.sleep(2)

print("========== T3 §1.1 中段针按档扫描（针在正中；池=131,072 token ≈ 6,240 行）==========", flush=True)
for n in (400, 800, 1600, 3100, 4500, 6100):
    q = dict(BASE)
    q["messages"] = [{"role": "user", "content": doc(n) + "\n\nQuestion: what is the secret access code mentioned above? Answer with the code only."}]
    q["max_tokens"] = 64
    report("needle-%dL(~%dKtok)" % (n, round(n * 21 / 1000)), *post(q))
    time.sleep(1)

print("========== T4 超池：7,100 行（≈149K token > 池 131,072）= 尽力而为区 ==========", flush=True)
q = dict(BASE)
q["messages"] = [{"role": "user", "content": doc(7100) + "\n\nQuestion: what is the secret access code mentioned above? Answer with the code only."}]
q["max_tokens"] = 64
report("overpool-7100L(~149Ktok)", *post(q))

print("========== 汇总 ==========", flush=True)
print("FAILS = %s" % (FAILS if FAILS else "无（全部通过）"), flush=True)
