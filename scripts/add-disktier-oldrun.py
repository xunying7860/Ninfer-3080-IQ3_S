#!/usr/bin/env python3
# 给旧引擎 run.sh 加磁盘层（幂等；改前备份）
# 依据：引擎自身 --help —— "disk-kv-path DIR adds a disk tier: evicted continuations write their
#       KV and StateImages there (per artifact and profile, --disk-kv-gib total, default 64)
#       and survive restarts; --disk-kv-restore seeds a new request's matching prefix from it"
# ★ 注意：只插一整行、行尾必须带反斜杠，且**绝不能**在续行中间写注释（shell 不把非词首 # 当注释）。
import shutil
import sys
import time

P = "/data/ninfer/run.sh"
ANCHOR = '  --gdn-state-fp16 --spec mtp --draft-tokens 3 --adaptive-mtp \\\n'
NEW = ('  --gdn-state-fp16 --spec mtp --draft-tokens 3 --adaptive-mtp \\\n'
       '  --disk-kv-path /data/nvme/ninfer-disktier --disk-kv-gib 64 --disk-kv-restore \\\n')

s = open(P, encoding="utf-8").read()
if "--disk-kv-path" in s:
    print("已经是启用状态，跳过")
    sys.exit(0)
n = s.count(ANCHOR)
if n != 1:
    print("锚点命中 %d 次（期望 1）—— 不写盘" % n)
    sys.exit(1)
shutil.copy2(P, P + ".bak-%s-pre-diskkv" % time.strftime("%Y%m%d-%H%M%S"))
open(P, "w", encoding="utf-8").write(s.replace(ANCHOR, NEW))
print("已加：--disk-kv-path /data/nvme/ninfer-disktier --disk-kv-gib 64 --disk-kv-restore")
