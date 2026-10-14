#!/usr/bin/env python3
# 解析 A-B-A 期间的 GPU 采样（ab-gpu.csv）与 top 采样（ab-cpu.txt），定位 strata 并发掉速的通道。
import re

A = ("22:04:05", "22:04:40")
B = ("22:04:47", "22:05:34")
C = ("22:05:45", "22:06:15")


def hhmmss(ts):
    m = re.search(r"(\d\d:\d\d:\d\d)", ts)
    return m.group(1) if m else ""


def inwin(t, w):
    return w[0] <= t <= w[1]


print("=== GPU 采样窗口统计（ab-gpu.csv）===")
rows = []
for ln in open("/data/ninfer/logs/ab-gpu.csv", errors="replace"):
    p = [x.strip() for x in ln.split(",")]
    if len(p) != 6:
        continue
    t = hhmmss(p[0])
    try:
        rows.append((t, int(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    except ValueError:
        continue
print("总样本 %d，时间范围 %s ~ %s" % (len(rows), rows[0][0] if rows else "-", rows[-1][0] if rows else "-"))
for name, w in (("A 单独", A), ("B 并发", B), ("C 单独", C)):
    for gpu in (0, 1):
        sel = [r for r in rows if r[1] == gpu and inwin(r[0], w)]
        if not sel:
            print("  %s GPU%d: 无样本" % (name, gpu))
            continue
        n = len(sel)
        temp = [r[2] for r in sel]
        pw = [r[3] for r in sel]
        ut = [r[4] for r in sel]
        mem = [r[5] for r in sel]
        print("  %s GPU%d: n=%d 温度 %d/%.1f/%d  功耗 %d/%.1f/%d W  util %d/%.1f/%d %%  显存峰值 %d MiB"
              % (name, gpu, n, min(temp), sum(temp) / n, max(temp),
                 min(pw), sum(pw) / n, max(pw), min(ut), sum(ut) / n, max(ut), max(mem)))

print()
print("=== CPU 采样：各采样点 CPU 占用前 3 名（ab-cpu.txt）===")
blocks = []
cur = None
for ln in open("/data/ninfer/logs/ab-cpu.txt", errors="replace"):
    if ln.startswith("top - "):
        if cur:
            blocks.append(cur)
        cur = {"t": hhmmss(ln), "load": ln.strip(), "procs": []}
    elif cur is not None:
        p = ln.split(None, 11)
        if len(p) >= 12 and p[0].isdigit():
            try:
                cur["procs"].append((float(p[8]), p[11].strip()))
            except ValueError:
                pass
if cur:
    blocks.append(cur)

for name, w in (("A 单独", A), ("B 并发", B), ("C 单独", C)):
    sel = [b for b in blocks if inwin(b["t"], w)]
    print("\n--- %s（%d 个采样点）---" % (name, len(sel)))
    if sel:
        print("  负载：", sel[len(sel) // 2]["load"][:110])
    agg = {}
    for b in sel:
        for cpu, cmd in b["procs"]:
            if any(k in cmd for k in ("ninfer", "strata", "python", "server")):
                agg[cmd] = agg.get(cmd, [])
                agg[cmd].append(cpu)
    for cmd, v in sorted(agg.items(), key=lambda kv: -max(kv[1])):
        print("  %-28s 峰值 %6.1f%%  均值 %6.1f%%  n=%d" % (cmd[:28], max(v), sum(v) / len(v), len(v)))
    top3 = sorted([p for b in sel for p in b["procs"]], key=lambda x: -x[0])[:3]
    print("  全局最高 3：", ["%.1f%% %s" % (c, n[:22]) for c, n in top3])
