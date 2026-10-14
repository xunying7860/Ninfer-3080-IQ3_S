#!/usr/bin/env python3
# dmon-summary.py —— 汇总 nvidia-smi dmon -s pucvmet 文件的每卡统计（列序见 analyze-dmon.py）
# 用法：python3 dmon-summary.py <file>
import sys
from collections import defaultdict

path = sys.argv[1]
rows = defaultdict(list)
for line in open(path, errors="replace"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    f = line.split()
    if len(f) < 14 or not f[0].isdigit():
        continue
    rows[int(f[0])].append(f)

for g in sorted(rows):
    r = rows[g]
    pwr = [float(x[1]) for x in r]
    sm = [float(x[4]) for x in r]
    pclk = [float(x[11]) for x in r]
    mclk = [float(x[10]) for x in r]
    rx = [float(x[20]) for x in r]
    gt = [float(x[2]) for x in r]
    active = [x for x in r if float(x[4]) > 5]
    print(f"GPU{g}: n={len(r)} | pwr mean={sum(pwr)/len(pwr):.1f} max={max(pwr):.0f} "
          f"| sm mean={sum(sm)/len(sm):.1f} max={max(sm):.0f} | pclk min={min(pclk):.0f} max={max(pclk):.0f} "
          f"| mclk min={min(mclk):.0f} | rx mean={sum(rx)/len(rx):.0f} max={max(rx):.0f} "
          f"| gtemp mean={sum(gt)/len(gt):.1f} max={max(gt):.0f} | active(sm>5%) n={len(active)}")
    if active:
        ap = [float(x[1]) for x in active]
        asm = [float(x[4]) for x in active]
        apc = [float(x[11]) for x in active]
        arx = [float(x[20]) for x in active]
        print(f"       active 子集: pwr mean={sum(ap)/len(ap):.1f} | sm mean={sum(asm)/len(asm):.1f} "
              f"| pclk min={min(apc):.0f} max={max(apc):.0f} | rx mean={sum(arx)/len(arx):.0f} max={max(arx):.0f}")
