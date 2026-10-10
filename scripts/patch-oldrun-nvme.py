#!/usr/bin/env python3
# 把旧引擎 run.sh 的模型路径从 SATA 改到 NVMe（幂等；失败即报错不写盘）
import sys

P = "/data/ninfer/run.sh"
OLD = '  "$ROOT/models/gsq_rco_iq3_s_dflash2_prop.ninfer"'
NEW = ('  # ★ 2026-10-09 用户要求：模型搬到 NVMe；SATA 原件保留到验证通过再清\n'
       '  "/data/nvme/ninfer-models/gsq_rco_iq3_s_dflash2_prop.ninfer"')

s = open(P, encoding="utf-8").read()
if NEW.splitlines()[-1].strip() in s:
    print("已经是 NVMe 路径，跳过")
    sys.exit(0)
n = s.count(OLD)
if n != 1:
    print("锚点命中 %d 次（期望 1）—— 不写盘" % n)
    sys.exit(1)
open(P, "w", encoding="utf-8").write(s.replace(OLD, NEW))
print("已改：模型路径 → /data/nvme/ninfer-models/")
