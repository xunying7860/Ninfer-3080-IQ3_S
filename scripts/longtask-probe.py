#!/usr/bin/env python3
# 长任务吞吐对比探针（x99 本机跑；对同一台引擎发同一份长任务）
# 用法: longtask-probe.py <port> <model-id> <标签> [prefill_lines] [out_tokens]
# 任务形状：长题面（默认 3100 行 ≈ 32K token，中段埋针）+ 长输出（默认 2000 token）
#   → 引擎权威行给出 prefill tok/s、TTFT、decode tok/s；随后同题面追问一次看 cache%/TTFT
# 注意：输出长度必须 prompt + max_tokens <= 池（否则活锁）；本探针默认 32K+2000，池 256K 安全。
import json, sys, time, urllib.request, urllib.error

PORT = int(sys.argv[1]); MODEL = sys.argv[2]; ARM = sys.argv[3]
NLINES = int(sys.argv[4]) if len(sys.argv) > 4 else 3100
OUT = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
BASE = "http://127.0.0.1:%d" % PORT
URL = BASE + "/v1/chat/completions"
NEEDLE = "IMPORTANT MEMO: the secret access code is BLUE-FALCON-7419, do not share it."

lines = []
for i in range(NLINES):
    lines.append("Line %06d: %s" % (i, NEEDLE) if i == NLINES // 2
                 else "Line %06d: routine maintenance notes, inventory records and log entries follow." % i)
TEXT = "\n".join(lines)


def post(body, timeout=3600):
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


def one(label, msgs, max_tokens):
    el, d, err = post({"model": MODEL, "messages": msgs, "max_tokens": max_tokens,
                       "temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0,
                       "presence_penalty": 0, "repetition_penalty": 1,
                       "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}})
    if err:
        print("[%s/%s] 失败: %s" % (ARM, label, err)); return
    u = d.get("usage", {}); ch = d["choices"][0]
    c = (ch["message"].get("content") or "")
    print("[%s/%s] 墙钟=%.1fs prompt=%s out=%s finish=%s 命中针=%s content[:80]=%r" %
          (ARM, label, el, u.get("prompt_tokens"), u.get("completion_tokens"),
           ch.get("finish_reason"), "BLUE-FALCON-7419" in c, c.strip()[:80]))


print("=== %s · 长任务：%d 行题面 + %d token 长输出 ===" % (ARM, NLINES, OUT))
q1 = [{"role": "user", "content": TEXT + "\n\nQuestion: what is the secret access code above? Then continue listing the line numbers you saw, as many as you can."}]
one("long-1", q1, OUT)
time.sleep(3)
q2 = q1 + [{"role": "assistant", "content": "BLUE-FALCON-7419"}, {"role": "user", "content": "Repeat the code and give a 200-word summary of the document."}]
one("long-2-followup", q2, 512)
print("=== 权威数字看引擎日志 req#N done 行（prefill/decode/TTFT/cache）===")
