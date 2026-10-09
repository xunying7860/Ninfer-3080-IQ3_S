#!/usr/bin/env python3
# KVMem 环 + 超长上下文探针（x99 本机跑）
# 用法: kvmem-probe.py <port> <model-id> <标签> [针档行数...]
#   默认档位：3100(≈65K) 6100(≈128K) 9300(≈196K)
# 每档：长题面 + 中段针（暗号 BLUE-FALCON-7419）→ 判检索是否真把中段搬回来；随后同题面追问看复用
# 另跑一段数数字 decode（3 轮，关思考）看稳态 decode 与复用 TTFT
import json, sys, time, urllib.request, urllib.error

PORT = int(sys.argv[1]); MODEL = sys.argv[2]; TAG = sys.argv[3]
SIZES = [int(x) for x in sys.argv[4:]] or [3100, 6100, 9300]
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT
NEEDLE = "IMPORTANT MEMO: the secret access code is BLUE-FALCON-7419, do not share it."
BASE = {"model": MODEL, "temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0,
        "presence_penalty": 0, "repetition_penalty": 1,
        "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}


def post(body, timeout=3600):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return time.time() - t, json.load(r), None
    except urllib.error.HTTPError as e:
        return time.time() - t, None, "HTTP %s: %s" % (e.code, e.read()[:300].decode("utf-8", "replace"))
    except Exception as e:
        return time.time() - t, None, repr(e)


def build(n):
    L = []
    for i in range(n):
        L.append("Line %06d: %s" % (i, NEEDLE) if i == n // 2
                 else "Line %06d: routine maintenance notes, inventory records and log entries follow." % i)
    return "\n".join(L)


def show(label, el, d, err, needle=True):
    if err:
        print("[%s/%s] %.1fs  %s" % (TAG, label, el, err)); return
    u = d.get("usage", {}); ch = d["choices"][0]
    c = (ch["message"].get("content") or "")
    hit = "BLUE-FALCON-7419" in c
    print("[%s/%s] %.1fs prompt=%s out=%s finish=%s 命中针=%s content[:70]=%r" %
          (TAG, label, el, u.get("prompt_tokens"), u.get("completion_tokens"),
           ch.get("finish_reason"), (hit if needle else "n/a"), c.strip()[:70]))


# ---------- 1) 数数字 decode（3 轮）----------
nums = " ".join(str(i) for i in range(1, 301))
b = dict(BASE); b["messages"] = [{"role": "user", "content": nums + "\n\nContinue the same number sequence."}]
b["max_tokens"] = 400
print("=== 段1 数数字 decode（3 轮）===")
for i in range(3):
    el, d, err = post(b, 600); show("nums-r%d" % (i + 1), el, d, err, needle=False)
    time.sleep(2)

# ---------- 2) 超长题面 + 中段针（逐档）----------
for n in SIZES:
    txt = build(n) + "\n\nQuestion: what is the secret access code mentioned above? Answer with the code only."
    print("=== 段2 题面 %d 行（≈%dK token）中段针 ===" % (n, round(n * 21 / 1000)))
    q = dict(BASE); q["messages"] = [{"role": "user", "content": txt}]; q["max_tokens"] = 64
    el, d, err = post(q, 3600); show("needle-%dL" % n, el, d, err)
    # 追问（前缀复用判据）
    q2 = dict(BASE); q2["messages"] = [{"role": "user", "content": txt},
                                       {"role": "assistant", "content": "BLUE-FALCON-7419"},
                                       {"role": "user", "content": "Repeat the code and say how many lines the document had."}]
    q2["max_tokens"] = 128
    el, d, err = post(q2, 3600); show("followup-%dL" % n, el, d, err)
    # 长上下文 decode 口径（同题面前缀 + 512 token 续写，纯 decode）
    q3 = dict(BASE)
    q3["messages"] = [{"role": "user", "content": txt},
                      {"role": "assistant", "content": "BLUE-FALCON-7419"},
                      {"role": "user", "content": "Now output the line numbers from 1 upward, space separated, as many as you can."}]
    q3["max_tokens"] = 512
    el, d, err = post(q3, 3600); show("longctx-decode-%dL" % n, el, d, err, needle=False)
print("=== 结束（权威行看引擎日志 req#N done）===")
