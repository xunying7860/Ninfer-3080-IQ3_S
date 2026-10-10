#!/usr/bin/env python3
# 在 strata 一次真实生成期间，采样：①每 CPU 占用 ②RAPL 整包功耗 ③引擎的专家命中率
# 用法: cpu-during-decode.py <port> <model> [输出token数]
import json
import subprocess
import sys
import threading
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-flash-next"
OUT = int(sys.argv[3]) if len(sys.argv) > 3 else 400


def cpu_times():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = [int(x) for x in line.split()[1:]]
            out[line.split()[0]] = sum(f), f[3] + f[4]      # total, idle
    return out


def rapl_uj():
    try:
        r = subprocess.run(["sudo", "-n", "cat", "/sys/class/powercap/intel-rapl:0/energy_uj"],
                           capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip().isdigit() else None
    except Exception:
        return None


res = {}


def gen():
    doc = "\n".join("Line %06d: routine maintenance notes, inventory records and log entries follow." % i
                    for i in range(300))
    b = {"model": MODEL, "messages": [{"role": "user", "content": doc +
         "\n\nWrite at least 300 words explaining monsoon climate formation, then give a conclusion."}],
         "max_tokens": OUT, "temperature": 1, "top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                                 data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        res["el"] = time.time() - t
        res["out"] = d.get("usage", {}).get("completion_tokens")
    except Exception as e:
        res["err"] = str(e)[:80]


th = threading.Thread(target=gen)
th.start()
time.sleep(1.0)                       # 等 prefill 过去，进入 decode
base = cpu_times()
base_j = rapl_uj()
t0 = time.time()
samples = []
while th.is_alive() and time.time() - t0 < 25:
    time.sleep(0.5)
    cur = cpu_times()
    j = rapl_uj()
    if j is not None and base_j is not None:
        samples.append((time.time() - t0, j - base_j))
th.join(timeout=5)
cur = cpu_times()
span = time.time() - t0
per = {}
for k in base:
    dt, di = cur[k][0] - base[k][0], cur[k][1] - base[k][1]
    per[k] = 100.0 * (dt - di) / dt if dt else 0.0
busy = sorted((v, k) for k, v in per.items() if k != "cpu")
w = None
if samples and base_j is not None:
    j2 = rapl_uj()
    if j2 is not None:
        w = (j2 - base_j) / 1e6 / span
print("窗口 %.1fs  输出 %s tok  用时 %s" % (span, res.get("out"), ("%.1fs" % res["el"]) if "el" in res else res.get("err")))
print("整包功耗(RAPL package): %s W" % ("%.1f" % w if w else "读不到(需 sudo)"))
print("忙核 TOP8: " + "  ".join("%s=%.0f%%" % (k.replace("cpu", "c"), v) for v, k in reversed(busy[-8:])))
idle = [k.replace("cpu", "c") for v, k in busy if v < 15]
print("基本闲置的核: %s" % (" ".join(idle[:20]) if idle else "无"))
print("整机 28 逻辑核平均: %.1f%%" % (sum(per.values()) / len(per)))
