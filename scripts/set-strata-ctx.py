#!/usr/bin/env python3
# 改 strata xxs 生产档的 --max-context（幂等；改前自动备份为 .bak-<时间>-pre<旧值>）
import json
import shutil
import sys
import time

CFG = "/data/workspace/strata01403/strata-iq3_xxs.1card-nvme.json"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 1048576

d = json.load(open(CFG, encoding="utf-8"))
a = d["args"]
i = a.index("--max-context")
cur = int(a[i + 1])
if cur == TARGET:
    print("已经是 %d，跳过" % TARGET)
    sys.exit(0)
bak = "%s.bak-%s-pre%d" % (CFG, time.strftime("%Y%m%d-%H%M%S"), cur)
shutil.copy2(CFG, bak)
print("备份: %s" % bak)
a[i + 1] = str(TARGET)
json.dump(d, open(CFG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.load(open(CFG, encoding="utf-8"))
print("已改: --max-context %d → %d（预测 KV ≈ %.1f GiB）" % (cur, TARGET, TARGET * 9.5 / 1048576.0 / 1024))
print("相关项: kv=%s reserve=%s expert-cache=%s conv-cache-mib=%s ple-io=%s ple-gguf=%s" % (
    a[a.index("--kv") + 1], a[a.index("--vram-reserve-mib") + 1], a[a.index("--expert-cache") + 1],
    a[a.index("--conversation-cache-mib") + 1], a[a.index("--ple-io") + 1],
    a[a.index("--ple-gguf") + 1].split("/")[-1]))
