#!/usr/bin/env python3
# 质量判据探针 v2（追加式会话线程：一次 prefill，后续问题走前缀复用）
# 用法: quality-probe2.py <port> <model-id> <标签> <行数> <深度百分比逗号分隔>
#   Q1..Qk：第 i 根针的 code 是什么（追加式 ⇒ 第 2 轮起复用，只算新增 token）
#   Qloc：第 N 行原文是什么（逐字判定）· Qcnt：文档多少行（计数判定）
#   末尾再发一条"同文档换问题的新会话"请求，记 cache%（验证"复用只吃逐轮追加"口径）
import json, random, re, sys, time, urllib.request, urllib.error

PORT = int(sys.argv[1]); MODEL = sys.argv[2]; TAG = sys.argv[3]
LINES = int(sys.argv[4]) if len(sys.argv) > 4 else 9300
DEPTHS = [int(x) for x in (sys.argv[5].split(",") if len(sys.argv) > 5 else ["5", "25", "50", "75", "95"])]
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT
BASE = {"model": MODEL, "temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0,
        "presence_penalty": 0, "repetition_penalty": 1,
        "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}

random.seed(20261009)
CODES = {}
docs = []
for i in range(LINES):
    docs.append("Line %06d: routine maintenance notes, inventory records and log entries follow." % i)
for k, d in enumerate(DEPTHS):
    pos = max(1, min(LINES - 2, LINES * d // 100))
    code = "ZQ-%d-%04d" % (d, random.randint(1000, 9999))
    CODES[k] = (d, pos, code)
    docs[pos] = "Line %06d: IMPORTANT MEMO #%d: the access code for depth %d is %s, keep it confidential." % (pos, k + 1, d, code)
DOC = "\n".join(docs)
LOC_LINE = LINES // 3


def post(msgs, max_tokens=96, timeout=3600):
    body = dict(BASE); body["messages"] = msgs; body["max_tokens"] = max_tokens
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        return time.time() - t, (d["choices"][0]["message"].get("content") or ""), d.get("usage", {}), None
    except urllib.error.HTTPError as e:
        return time.time() - t, "", {}, "HTTP %s: %s" % (e.code, e.read()[:200].decode("utf-8", "replace"))
    except Exception as e:
        return time.time() - t, "", {}, repr(e)


print("=== %s · 质量判据 v2（追加式会话）：%d 行（≈%dK token）· 针深度 %s ===" % (TAG, LINES, round(LINES * 21 / 1000), DEPTHS), flush=True)
msgs = []
hits = 0
for k, (d, pos, code) in enumerate(CODES.values()):
    q = "What is the access code in IMPORTANT MEMO #%d (the memo at depth %d%% of the document)? Answer with the code only." % (k + 1, d)
    msgs = ([{"role": "user", "content": DOC + "\n\n" + q}] if k == 0 else msgs + [{"role": "user", "content": q}])
    el, c, u, err = post(msgs, 96)
    ok = code in c; hits += ok
    print("[%s/Q%d 深度%d%%] %.1fs prompt=%s out=%s 期望=%s 命中=%s 答=%r %s" %
          (TAG, k + 1, d, el, u.get("prompt_tokens"), u.get("completion_tokens"), code, ok, c.strip()[:70], err or ""), flush=True)
    msgs = msgs + [{"role": "assistant", "content": c.strip()[:200]}]
print("[%s] 多针召回：%d/%d" % (TAG, hits, len(CODES)), flush=True)

msgs = msgs + [{"role": "user", "content": "What is the exact full text of line %06d of the document above? Quote it verbatim." % LOC_LINE}]
el, c, u, err = post(msgs, 96)
_loc_ok = ("Line %06d" % LOC_LINE) in c and "routine maintenance notes" in c and "not present" not in c.lower()
print("[%s/Qloc 行 %06d] %.1fs 命中=%s 答=%r" % (TAG, LOC_LINE, el, _loc_ok, c.strip()[:120]), flush=True)
msgs = msgs + [{"role": "assistant", "content": c.strip()[:200]},
               {"role": "user", "content": "How many lines does the document above have? Answer with the number only."}]
el, c, u, err = post(msgs, 64)
_cnt_ok = str(LINES) in re.sub(r"[,\s]", "", c)
print("[%s/Qcnt] %.1fs 期望=%d 命中=%s 答=%r" % (TAG, el, LINES, _cnt_ok, c.strip()[:60]), flush=True)

el, c, u, err = post([{"role": "user", "content": DOC + "\n\nName one thing that the document explicitly tells you NOT to share. One short sentence."}], 96)
print("[%s/Qfresh 新会话同文档换问题] %.1fs prompt=%s cached_tokens=%s 答=%r" %
      (TAG, el, u.get("prompt_tokens"), u.get("prompt_tokens_details", {}).get("cached_tokens"), c.strip()[:60]), flush=True)
print("=== 结束（权威行看引擎日志 req#N done）===", flush=True)
