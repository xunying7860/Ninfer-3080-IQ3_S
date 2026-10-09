#!/usr/bin/env python3
# 对指定 strata 配置跑 --calibrate 等价流程（与 setup.py 的 calibrate_config 同步骤）
# 用法: calib-apply.py /data/workspace/strata01403/strata-iq3_xxs.1card-nvme.json
# 步骤：CAL.run(cfg) 量 4 个硬件相关设置（pcie-frac / spec-min-p / pool-workers / adapt-*）
#       → CAL.apply() 写回 args（未改动的回到产品默认）→ 落盘。约 5-10 分钟，期间引擎会多次加载模型。
import json, os, sys, time

ROOT = "/data/workspace/strata01403"
sys.path.insert(0, ROOT + "/tools")
import calibrate as CAL

p = sys.argv[1]
cfg = json.loads(open(p, encoding="utf-8-sig").read())
try:
    since = os.path.getsize(cfg["log"]) if cfg.get("log") and os.path.isfile(cfg["log"]) else 0
except OSError:
    since = 0

print("=== 开始标定：%s ===" % p, flush=True)
print("（约 5-10 分钟；引擎会加载模型多次，PC 会忙）", flush=True)
try:
    res = CAL.run(cfg, say=lambda *a: print(*a, flush=True))
except Exception as e:
    print("标定失败：%r" % (e,), flush=True)
    why = CAL.engine_error(cfg.get("log"), since)
    if why:
        print("引擎自己说：%s" % why, flush=True)
    sys.exit(1)

print("settings =", res["settings"], flush=True)
print("report   =", json.dumps(res["report"], ensure_ascii=False)[:600], flush=True)
cfg["args"] = CAL.apply(cfg["args"], res["settings"])
open(p, "w").write(json.dumps(cfg, indent=2, ensure_ascii=False))
print("已写回配置：%s" % p, flush=True)
print("args 里相关项：", [a for i, a in enumerate(cfg["args"]) if a in ("--pcie-frac", "--spec-min-p", "--pool-workers", "--adapt-every", "--adapt-swaps", "--adapt-decay") or (i > 0 and cfg["args"][i - 1] in ("--pcie-frac", "--spec-min-p", "--pool-workers", "--adapt-every", "--adapt-swaps", "--adapt-decay"))], flush=True)
