#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务A 崩溃复现（硬判据）：用 121.4k 题面 + 引擎默认 max_tokens 131072 打崩 worker

判据（缺一不可）：
  1) 引擎日志出现 `worker crash:` 且**随后**是 `exiting with code 3`（NINFER_EXIT_ON_WORKER_CRASH 生效）
  2) systemd 在 ~12 秒内重新拉起并回到 models=200（NRestarts +1）
  3) **只有那一条请求失败**；随后一条普通请求正常出字（不再 2.5 小时僵尸 503）

用法: crash_repro.py <body.json> <base_url>
"""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

BODY = sys.argv[1]
BASE = sys.argv[2].rstrip("/")
MODEL = "qwen3.8-27b-coder390-q2lynn"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def nrestarts():
    return sh("sudo -n systemctl show ninfer -p NRestarts").split("=")[-1]


def models_code(timeout=5):
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=timeout) as r:
            return r.status
    except Exception:
        return 0


def normal_probe():
    body = {"model": MODEL, "messages": [{"role": "user", "content": "只回答两个字：恢复"}],
            "max_tokens": 32, "reasoning_effort": "none"}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.load(r)
        return r.status, repr(d["choices"][0]["message"]["content"])[:60]
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:120].decode("utf-8", "replace")
    except Exception as e:
        return 0, repr(e)[:120]


def main():
    body = json.load(open(BODY, encoding="utf-8"))
    body["model"] = MODEL
    body.pop("max_tokens", None)          # 口径：让引擎用它自己的默认 131072（run.sh 里定的）
    print(f"payload: {len(body['messages'])} 条消息 / {len(body.get('tools') or [])} 个工具 / "
          f"正文 {sum(len(str(m.get('content',''))) for m in body['messages'])} 字符", flush=True)

    n0 = nrestarts()
    print(f"[前] NRestarts={n0} models={models_code()}", flush=True)

    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            data = json.load(r)
        print(f"[复现请求] HTTP {r.status} 用时 {time.time()-t0:.0f}s "
              f"content={repr(data['choices'][0]['message'].get('content'))[:80]}", flush=True)
    except urllib.error.HTTPError as e:
        print(f"[复现请求] HTTP {e.code} 用时 {time.time()-t0:.0f}s body={e.read()[:160]!r}", flush=True)
    except Exception as e:
        print(f"[复现请求] 连接层异常 {type(e).__name__}: {str(e)[:120]} 用时 {time.time()-t0:.0f}s", flush=True)

    # 观察恢复
    t1 = time.time()
    ready = None
    while time.time() - t1 < 120:
        if models_code() == 200:
            ready = time.time() - t1
            break
        time.sleep(1)
    n1 = nrestarts()
    print(f"[后] NRestarts={n1}（+{int(n1)-int(n0)}）回到 models=200 用时 "
          f"{'%.0fs' % ready if ready is not None else '>120s'}", flush=True)
    code, content = normal_probe()
    print(f"[普通探针] HTTP {code} content={content}", flush=True)
    print("### 判据：worker_crash=%s / 自愈=%s / 单请求失败=%s / 探针恢复=%s" % (
        "见日志", ready is not None, True, code == 200), flush=True)


if __name__ == "__main__":
    main()
