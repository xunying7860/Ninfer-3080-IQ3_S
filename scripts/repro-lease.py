#!/usr/bin/env python3
# 复现 2026-10-09 15:54 的崩溃形状并验证 --kv-lease-growth 修法
#   崩点：req#44 | 7 messages | max output 12,288 | thinking xhigh | tools 5（stream）
#   前置状态：池里已有一个 ~78K token 的大会话（req#40/#41）
# 用法: repro-lease.py <port> <model>
import json, sys, time, urllib.request, urllib.error

PORT = int(sys.argv[1]); MODEL = sys.argv[2]
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT
TOOLS = [{"type": "function", "function": {"name": n, "description": "tool " + n,
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                         "required": ["command"]}}}
         for n in ("terminal", "read_file", "search_files", "write_file", "patch")]


def doc(n, tag="routine maintenance notes, inventory records and log entries follow."):
    return "\n".join("Line %06d: %s" % (i, tag) for i in range(n))


def post(body, timeout=900, stream=False):
    body = dict(body, stream=stream)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if not stream:
                d = json.load(r)
                u = d.get("usage", {})
                return time.time() - t, "ok prompt=%s out=%s finish=%s" % (
                    u.get("prompt_tokens"), u.get("completion_tokens"), d["choices"][0].get("finish_reason"))
            chunks, n = [], 0
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    j = json.loads(payload)
                except Exception:
                    continue
                n += 1
                if n <= 3 or n % 50 == 0:
                    chunks.append(json.dumps(j, ensure_ascii=False)[:90])
            return time.time() - t, "ok(stream) chunks=%d 前几块=%s" % (n, chunks[:2])
    except urllib.error.HTTPError as e:
        return time.time() - t, "HTTP %s: %s" % (e.code, e.read()[:160].decode("utf-8", "replace"))
    except Exception as e:
        return time.time() - t, "%s %s" % (type(e).__name__, str(e)[:80])


COMMON = {"model": MODEL, "temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0,
          "presence_penalty": 0, "repetition_penalty": 1}

print("== 前置：建一个 ~78K token 的大会话（占住池 + 银行化检查点）==", flush=True)
big = doc(3714)
el, r = post(dict(COMMON, messages=[{"role": "user", "content": big + "\n\nSay OK only."}],
                  max_tokens=64, reasoning_effort="none",
                  chat_template_kwargs={"enable_thinking": False}))
print("  [预热] %.1fs %s" % (el, r), flush=True)

print("== 复现：7 消息 + thinking xhigh + 5 tools + max 12,288 + stream（崩点形状）×3 轮 ==", flush=True)
msgs = [{"role": "user", "content": "You are asked to inspect a repository."}]
for i in range(1, 7):
    msgs.append({"role": "assistant", "content": "Step %d noted." % i})
    msgs.append({"role": "user", "content": "Continue step %d: read the file then patch it." % i})
for k in range(3):
    el, r = post(dict(COMMON, messages=msgs, max_tokens=12288, tools=TOOLS,
                      reasoning_effort="xhigh"), stream=True)
    print("  [%d] %.1fs %s" % (k + 1, el, r), flush=True)
    time.sleep(2)
print("== 结束（崩没崩看引擎日志 worker crash）==", flush=True)
