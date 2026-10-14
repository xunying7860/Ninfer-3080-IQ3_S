#!/usr/bin/env python3
# 双服务「掉不掉速」判定：A-B-A 交替采样（消除漂移与单次噪声）
#   A 段：strata 单独 5 轮（ninfer 空闲）
#   B 段：strata 5 轮 + ninfer 16 轮 同时开跑（barrier 同时起）
#   C 段：strata 单独 5 轮（ninfer 空闲）
# 读数：直接解析两台引擎自己的日志（权威），按段切分后打印。
#   两个日志都在本机（x99），所以脚本自己按行号切片，不靠时间戳（strata 日志无时间戳）。
import json, sys, threading, time, urllib.request, urllib.error

NINFER_LOG = "/data/ninfer/logs/ninfer-serve.log"
STRATA_LOG = "/data/workspace/strata0137/strata-iq3_xxs.1card.log"
RS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
RN = int(sys.argv[2]) if len(sys.argv) > 2 else 16

NUMS = " ".join(str(i) for i in range(1, 301))
PROMPT = NUMS + ("\n\nContinue the same number sequence from the next number. Output numbers "
                 "only, in the same space-separated format, and keep going until the length limit.")
T0 = time.time()


def stamp():
    return "T+%7.2fs" % (time.time() - T0)


def snap():
    out = []
    for p in (NINFER_LOG, STRATA_LOG):
        with open(p, "rb") as f:
            out.append(len(f.read().splitlines()))
    return out


def slice_lines(path, start):
    with open(path, "r", errors="replace") as f:
        lines = f.read().splitlines()
    return lines[start:]


def show(tag, ns, ss):
    print("\n----- %s：引擎权威读数 -----" % tag)
    print("[ninfer]")
    got = False
    for ln in slice_lines(NINFER_LOG, ns):
        if "req#" in ln and " done |" in ln:
            print("  " + ln.strip())
            got = True
    if not got:
        print("  （无）")
    print("[strata]")
    got = False
    for ln in slice_lines(STRATA_LOG, ss):
        if "generated in" in ln or "decode expert cache hit rate" in ln:
            print("  " + ln.strip())
            got = True
    if not got:
        print("  （无）")


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


def strata_rounds(n, barrier=None, tag=""):
    if barrier is not None:
        barrier.wait()
    for k in range(1, n + 1):
        el, d, err = post(8080, {"model": "qwen3.8-flash-next-iq3_xxs",
                                 "messages": [{"role": "user", "content": PROMPT}],
                                 "max_tokens": 1000, "temperature": 0,
                                 "chat_template_kwargs": {"enable_thinking": False}})
        if err:
            print("%s [strata%s r%d] 失败: %s" % (stamp(), tag, k, err))
            continue
        u = d.get("usage", {})
        ch = d["choices"][0]
        print("%s [strata%s r%d] finish=%s output=%s 墙钟=%.2fs 末值=%r"
              % (stamp(), tag, k, ch.get("finish_reason"), u.get("completion_tokens"),
                 el, (ch["message"].get("content") or "").strip()[-30:]))


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
        print("%s [ninfer r%d] finish=%s output=%s 墙钟=%.2fs"
              % (stamp(), k, ch.get("finish_reason"), u.get("completion_tokens"), el))


print("=== A-B-A 交替：strata 每段 %d 轮，B 段 ninfer 同时 %d 轮 | 开始 %s ==="
      % (RS, RN, time.strftime("%H:%M:%S")))

print("\n########## A 段：strata 单独 ##########")
ns, ss = snap()
strata_rounds(RS, tag="-A")
show("A 段（strata 单独）", ns, ss)

print("\n########## B 段：strata + ninfer 并发 ##########")
ns, ss = snap()
bar = threading.Barrier(2)
tn = threading.Thread(target=ninfer_rounds, args=(RN, bar))
ts = threading.Thread(target=strata_rounds, args=(RS, bar, "-B"))
tn.start()
ts.start()
tn.join()
ts.join()
show("B 段（并发）", ns, ss)

print("\n########## C 段：strata 单独 ##########")
ns, ss = snap()
strata_rounds(RS, tag="-C")
show("C 段（strata 单独）", ns, ss)

print("\n=== 完 %s（总 %.1fs）===" % (stamp(), time.time() - T0))
