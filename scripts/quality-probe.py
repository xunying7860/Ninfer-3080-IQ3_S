#!/usr/bin/env python3
# 质量判据探针（多针召回 + 定位题 + 前缀复用）
# 用法: quality-probe.py <port> <model-id> <标签> <行数> <深度百分比逗号分隔> [额外问题模式]
#   例: quality-probe.py 18084 qwen3.8-27b Q195K 9300 5,25,50,75,95
# 做法：造一份 S 行文档，在指定深度各埋一根**唯一暗号针**（MEMO-i: code <CODE-i>），
#   然后逐问（每题都带同一份文档前缀 ⇒ 第 2 题起走前缀复用）：
#     Q_i：第 i 根针的 code 是什么？        → 逐针召回率
#     Q_loc：第 N 行的原文是什么？          → 定位题（可逐字判定）
#     Q_cnt：文档一共多少行？               → 计数题
# 判据：召回 = 答案里出现对应 CODE；定位 = 答案里出现 "Line <N>"；计数 = 答案里出现正确行数。
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


def ask(question, max_tokens=96, timeout=3600):
    body = dict(BASE)
    body["messages"] = [{"role": "user", "content": DOC + "\n\n" + question}]
    body["max_tokens"] = max_tokens
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        el = time.time() - t
        c = (d["choices"][0]["message"].get("content") or "")
        return el, c, d.get("usage", {}), None
    except urllib.error.HTTPError as e:
        return time.time() - t, "", {}, "HTTP %s: %s" % (e.code, e.read()[:200].decode("utf-8", "replace"))
    except Exception as e:
        return time.time() - t, "", {}, repr(e)


print("=== %s · 质量判据：%d 行（≈%dK token）· 针深度 %s ===" % (TAG, LINES, round(LINES * 21 / 1000), DEPTHS))
hits = 0
for k, (d, pos, code) in enumerate(CODES.values()):
    el, c, u, err = ask("What is the access code in IMPORTANT MEMO #%d (the memo at depth %d%% of the document)? Answer with the code only." % (k + 1, d))
    ok = (code in c)
    hits += ok
    print("[%s/Q%d 深度%d%%] %.1fs prompt=%s out=%s 期望=%s 命中=%s 答=%r" %
          (TAG, k + 1, d, el, u.get("prompt_tokens"), u.get("completion_tokens"), code, ok, c.strip()[:80]))
    if err: print("    err:", err)
print("[%s] 多针召回：%d/%d" % (TAG, hits, len(CODES)))

el, c, u, err = ask("What is the exact full text of line %06d of the document above? Quote it verbatim." % LOC_LINE)
print("[%s/Qloc 行 %06d] %.1fs 命中=%s 答=%r" % (TAG, LOC_LINE, el, ("Line %06d" % LOC_LINE) in c, c.strip()[:100]))

el, c, u, err = ask("How many lines does the document above have? Answer with the number only.")
ok = str(LINES) in re.sub(r"[,\s]", "", c)
print("[%s/Qcnt] %.1fs 期望=%d 命中=%s 答=%r" % (TAG, LINES, ok, c.strip()[:60]))
print("=== 结束（权威行看引擎日志 req#N done）===")
