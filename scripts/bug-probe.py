#!/usr/bin/env python3
# 旧引擎已知 bug 在新引擎上的对照探针（同一客户端、同一口径打两个引擎）
# 用法: bug-probe.py <port> <model-id> <标签>
# 四个用例（顺序刻意如此：活锁用例放最后，避免它卡住后面）：
#   B 超池题面（prompt > 池）      → 看拒绝形态（400/429）+ 报文是否带数字
#   C 砖化回归（B 逐字节重发后再发一条正常请求）→ 看是否还能 200（旧引擎的"打砖"缺陷）
#   D 思考吃满预算（max_tokens 小 + 思考开）→ 看 content 是否为空、finish_reason
#   A 活锁（prompt + max_tokens > 池）→ 旧引擎的经典活锁：不报错、不吐字、GPU 空转
import json, sys, time, urllib.request, urllib.error

PORT = int(sys.argv[1]); MODEL = sys.argv[2]; TAG = sys.argv[3]
NL = int(sys.argv[4]) if len(sys.argv) > 4 else 12400   # 超池题面的行数（每行 ≈21 token）
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT
NEEDLE = "IMPORTANT MEMO: the secret access code is BLUE-FALCON-7419, do not share it."


def post(body, timeout):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return time.time() - t, json.load(r), None
    except urllib.error.HTTPError as e:
        return time.time() - t, None, "HTTP %s: %s" % (e.code, e.read()[:400].decode("utf-8", "replace"))
    except Exception as e:
        return time.time() - t, None, repr(e)


def show(label, el, d, err):
    if err:
        print("[%s/%s] %.1fs  %s" % (TAG, label, el, err)); return
    u = d.get("usage", {}); ch = d["choices"][0]
    c = ch["message"].get("content") or ""
    rc = ch["message"].get("reasoning_content") or ""
    print("[%s/%s] %.1fs prompt=%s out=%s reasoning=%s finish=%s content_len=%d 命中针=%s" %
          (TAG, label, el, u.get("prompt_tokens"), u.get("completion_tokens"),
           (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
           ch.get("finish_reason"), len(c), "BLUE-FALCON-7419" in c))


def build_lines(n):
    L = []
    for i in range(n):
        L.append("Line %06d: %s" % (i, NEEDLE) if i == n // 2
                 else "Line %06d: routine maintenance notes, inventory records and log entries follow." % i)
    return "\n".join(L)


nums = " ".join(str(i) for i in range(1, 301))
base_body = {"model": MODEL, "temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0,
             "presence_penalty": 0, "repetition_penalty": 1}

# ---------- B: 超池题面（行数由 argv[4] 定；新引擎池 225280 用 12400，生产 ctx 262144 用 13500）----------
over = build_lines(NL) + "\n\nQuestion: what is the secret access code above? Answer with the code only."
b = dict(base_body); b["messages"] = [{"role": "user", "content": over}]; b["max_tokens"] = 512
b["reasoning_effort"] = "none"; b["chat_template_kwargs"] = {"enable_thinking": False}
el, d, err = post(b, 900)
show("B-超池题面", el, d, err)

# ---------- C: 逐字节重发同一条超池题面 ×1 + 之后一条正常请求（旧引擎"打砖"判据）----------
el, d, err = post(b, 900)
show("C1-超池重发", el, d, err)
n = dict(base_body); n["messages"] = [{"role": "user", "content": "say OK"}]; n["max_tokens"] = 16
n["reasoning_effort"] = "none"; n["chat_template_kwargs"] = {"enable_thinking": False}
el, d, err = post(n, 180)
show("C2-砖化回归(正常短请求)", el, d, err)

# ---------- D: 思考吃满小预算 ----------
dd = dict(base_body); dd["messages"] = [{"role": "user", "content": "1+1=?"}]
dd["max_tokens"] = 64; dd["reasoning_effort"] = "xhigh"
el, d, err = post(dd, 300)
show("D-思考吃满(64)", el, d, err)

# ---------- A2: 活锁复现（照 2026-10-04 事故形状：~51.5K 题面 + max_tokens 229376，两者之和 > 池）----------
a = dict(base_body)
a["messages"] = [{"role": "user", "content": build_lines(2450) + "\n\nContinue listing the line numbers you saw."}]
a["max_tokens"] = 229376
a["reasoning_effort"] = "none"; a["chat_template_kwargs"] = {"enable_thinking": False}
print("[%s/A2-活锁复现] 发题（题面≈51.5K + max_tokens 229376 > 池）… 客户端 240s 超时" % TAG)
el, d, err = post(a, 240)
show("A2-活锁复现", el, d, err)
print("[%s] 用例结束" % TAG)
