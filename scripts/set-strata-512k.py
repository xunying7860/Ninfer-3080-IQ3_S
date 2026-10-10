#!/usr/bin/env python3
# 把 strata xxs 生产档的 --max-context 262144 → 524288（512K），改前自动备份（幂等）
import json
import shutil
import sys
import time

CFG = "/data/workspace/strata01403/strata-iq3_xxs.1card-nvme.json"
TARGET = 524288

d = json.load(open(CFG, encoding="utf-8"))
a = d["args"]
i = a.index("--max-context")
cur = int(a[i + 1])
if cur == TARGET:
    print("已经是 %d，跳过" % TARGET)
    sys.exit(0)
bak = CFG + ".bak-%s-pre512k" % time.strftime("%Y%m%d-%H%M%S")
shutil.copy2(CFG, bak)
print("备份: %s" % bak)
a[i + 1] = str(TARGET)
json.dump(d, open(CFG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.load(open(CFG, encoding="utf-8"))          # 校验能解析
d2 = json.load(open(CFG, encoding="utf-8"))
print("已改: --max-context %d → %s" % (cur, d2["args"][i + 1]))
print("其余相关项: kv=%s  vram-reserve=%s  expert-cache=%s  conversation-cache-mib=%s  ple-io=%s" % (
    d2["args"][d2["args"].index("--kv") + 1],
    d2["args"][d2["args"].index("--vram-reserve-mib") + 1],
    d2["args"][d2["args"].index("--expert-cache") + 1],
    d2["args"][d2["args"].index("--conversation-cache-mib") + 1],
    d2["args"][d2["args"].index("--ple-io") + 1]))
