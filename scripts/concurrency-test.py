#!/usr/bin/env python3
# 双服务并发验证：ninfer(GPU0:18082) 与 strata(GPU1:8080) 同口径数数字语料。
# 用法：concurrency-test.py <mode> [rounds_ninfer] [rounds_strata]
#   mode = ninfer  : 只打 ninfer（测单独基线）
#   mode = strata  : 只打 strata（测单独基线）
#   mode = both    : 两边用 barrier 同时开跑（测并发是否掉速）
# 口径：数数字 1..300 接着数、max_tokens=1000、temperature=0、思考关。
#   ninfer 生产默认 --default-reasoning-effort max ⇒ 逐请求 reasoning_effort=none；
#   strata 必须 chat_template_kwargs.enable_thinking=false。
# 权威读数取各引擎自己的日志行（本脚本只打印客户端侧墙钟，用于对齐时间窗）。
import json, sys, threading, time, urllib.request, urllib.error

MODE = sys.argv[1] if len(sys.argv) > 1 else "both"
RN = int(sys.argv[2]) if len(sys.argv) > 2 else 10
RS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

NUMS = " ".join(str(i) for i in range(1, 301))
PROMPT = NUMS + ("\n\nContinue the same number sequence from the next number. Output numbers "
                 "only, in the same space-separated format, and keep going until the length limit.")

T0 = time.time()


def stamp():
    return "T+%7.2fs" % (time.time() - T0)


def post(port, body, timeout=2400):
    url = "http://127.0.0.1:%d/v1/chat/completions" % port
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return time.time() - t, json.load(r), None
    except urllib.error.HTTPError as e:
        return time.time() - t, None, "HTTP %s: %s" % (e.code, e.read()[:300])
    except Exception as e:
        return time.time() - t, None, repr(e)


def ninfer_rounds(n, barrier=None):
    if barrier is not None:
        barrier.wait()
    for k in range(1, n + 1):
        el, d, err = post(18082, {"model": "qwen3.8-27b",
                                  "messages": [{"role": "user", "content": PROMPT}],
                                  "max_tokens": 1000, "temperature": 0,
                                  "reasoning_effort": "none"})
        if err:
            print("%s [ninfer r%d] 失败: %s" % (stamp(), k, err))
            continue
        u = d.get("usage", {})
        ch = d["choices"][0]
        c = ch["message"].get("content") or ""
        print("%s [ninfer r%d] finish=%s prompt=%s output=%s 墙钟=%.2fs 末值=%r"
              % (stamp(), k, ch.get("finish_reason"), u.get("prompt_tokens"),
                 u.get("completion_tokens"), el, c.strip()[-30:]))


def strata_rounds(n, barrier=None):
    if barrier is not None:
        barrier.wait()
    for k in range(1, n + 1):
        el, d, err = post(8080, {"model": "qwen3.8-flash-next-iq3_xxs",
                                 "messages": [{"role": "user", "content": PROMPT}],
                                 "max_tokens": 1000, "temperature": 0,
                                 "chat_template_kwargs": {"enable_thinking": False}})
        if err:
            print("%s [strata r%d] 失败: %s" % (stamp(), k, err))
            continue
        u = d.get("usage", {})
        ch = d["choices"][0]
        c = ch["message"].get("content") or ""
        print("%s [strata r%d] finish=%s prompt=%s output=%s 墙钟=%.2fs 末值=%r"
              % (stamp(), k, ch.get("finish_reason"), u.get("prompt_tokens"),
                 u.get("completion_tokens"), el, c.strip()[-30:]))


print("模式=%s | ninfer 轮数=%d | strata 轮数=%d | 起始时刻 %s"
      % (MODE, RN, RS, time.strftime("%H:%M:%S")))
if MODE == "ninfer":
    ninfer_rounds(RN)
elif MODE == "strata":
    strata_rounds(RS)
else:
    bar = threading.Barrier(2)
    tn = threading.Thread(target=ninfer_rounds, args=(RN, bar))
    ts = threading.Thread(target=strata_rounds, args=(RS, bar))
    tn.start()
    ts.start()
    tn.join()
    ts.join()
print("结束 %s（总 %.1fs）" % (stamp(), time.time() - T0))
