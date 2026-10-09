#!/usr/bin/env python3
# 400K 题面复核：① 要暗号（判断中段检索）② 问暗号在第几行（判断"真读到中段"而非猜中）
# 题面与 kvmem-probe.py 的 19000 行档完全一致（同构造），第 ② 问要求答出 6 位行号。
import json, urllib.request, time

NEEDLE = "IMPORTANT MEMO: the secret access code is BLUE-FALCON-7419, do not share it."
n = 19000
L = []
for i in range(n):
    L.append("Line %06d: %s" % (i, NEEDLE) if i == n // 2
             else "Line %06d: routine maintenance notes, inventory records and log entries follow." % i)
doc = "\n".join(L)
print("题面行数=%d 针在第 %06d 行" % (n, n // 2), flush=True)

for tag, q in (("n1-要暗号", "what is the secret access code mentioned above? Answer with the code only."),
               ("n2-定位行号", "Which line number contains the secret access code? Answer with the 6-digit number only.")):
    b = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": doc + "\n\nQuestion: " + q}],
         "max_tokens": 32, "temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0,
         "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request("http://127.0.0.1:18084/v1/chat/completions",
                               data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(r, timeout=2400) as x:
            d = json.load(x)
        u = d["usage"]; c = (d["choices"][0]["message"].get("content") or "").strip()
        print("[%s] %.1fs prompt=%s out=%s 命中暗号=%s 答=%r" %
              (tag, time.time() - t, u["prompt_tokens"], u["completion_tokens"],
               "BLUE-FALCON-7419" in c, c[:70]), flush=True)
    except Exception as e:
        print("[%s] FAIL %s" % (tag, e), flush=True)
print("=== 400K 复核结束 ===", flush=True)
