#!/usr/bin/env python3
# 长会话压力（复刻 2026-10-09 三次崩溃的形态）：多轮 → max_tokens 12288 + thinking xhigh + tools 5，
# 会话从 ~60K 涨到 ~100K；全程记录 200/非200、崩溃后是否自愈。
# 用法: stress-longsession.py <port> <model> [轮数] [首轮文档行数]
import json, sys, time, urllib.request, urllib.error

PORT = int(sys.argv[1]); MODEL = sys.argv[2]
TURNS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
DOCLINES = int(sys.argv[4]) if len(sys.argv) > 4 else 2900
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT
TOOLS = [{"type": "function", "function": {"name": n, "description": "tool " + n,
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                         "required": ["command"]}}}
         for n in ("terminal", "read_file", "search_files", "write_file", "patch")]
BASE = {"model": MODEL, "temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0,
        "presence_penalty": 0, "repetition_penalty": 1, "reasoning_effort": "xhigh"}


def post(msgs, mt, timeout=900, tools=TOOLS):
    b = dict(BASE, messages=msgs, max_tokens=mt)
    if tools:
        b["tools"] = tools
    req = urllib.request.Request(URL, data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        u = d.get("usage", {}); ch = d["choices"][0]
        return time.time() - t, 200, u.get("prompt_tokens"), u.get("completion_tokens"), \
            (ch["message"].get("content") or "")[:180]
    except urllib.error.HTTPError as e:
        return time.time() - t, e.code, None, None, e.read()[:120].decode("utf-8", "replace")
    except Exception as e:
        return time.time() - t, "ERR", None, None, "%s %s" % (type(e).__name__, str(e)[:90])


doc = "\n".join("Line %06d: routine maintenance notes, inventory records and log entries follow." % i
                for i in range(DOCLINES))
convo = [{"role": "user", "content": doc + "\n\nQuestion: how many lines does this document have? Answer the number only."}]
el, code, pt, ot, txt = post(convo, 12288)
print("[turn00 建会话] %.1fs http=%s prompt=%s out=%s %r" % (el, code, pt, ot, txt[:60]), flush=True)
convo.append({"role": "assistant", "content": txt or "3100"} if txt else {"role": "assistant", "content": "2900"})

bad = 0
for turn in range(1, TURNS + 1):
    convo.append({"role": "user",
                  "content": "第 %d 步：先写不少于 200 字的分析，再给出一行结论（以 CONCLUSION: 开头）。" % turn})
    el, code, pt, ot, txt = post(convo, 12288)
    ok = (code == 200)
    if not ok:
        bad += 1
    print("[turn%02d] %.1fs http=%s prompt=%s out=%s %s %r" %
          (turn, el, code, pt, ot, "OK" if ok else "**FAIL**", txt[:70]), flush=True)
    convo.append({"role": "assistant", "content": txt if ok else "(failed)"})
    if not ok:
        # 若引擎退出（自愈开关生效）→ 等它回来再继续，验证"自愈"
        for wait in range(60):
            time.sleep(10)
            el2, c2, _, _, _ = post([{"role": "user", "content": "ping"}], 4, timeout=30, tools=None)
            if c2 == 200:
                print("    ... 引擎已自愈（等待 %ds 后恢复）" % ((wait + 1) * 10), flush=True)
                break
    time.sleep(1)

print("=== 结束：FAIL 次数 = %d / %d 轮 ===" % (bad, TURNS), flush=True)
