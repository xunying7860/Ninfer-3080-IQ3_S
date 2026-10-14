#!/usr/bin/env python3
# 解析 nvidia-smi dmon -s pucvmet 采样文件，按 GPU 汇总功耗/SM时钟/利用率/PCIe 收发。
# 列：0 gpu 1 pwr 2 gtemp 3 mtemp 4 sm 5 mem 6 enc 7 dec 8 jpg 9 ofa 10 mclk 11 pclk
#     12 pviol 13 tviol 14 fb 15 bar1 16 ccpm 17 sbecc 18 dbecc 19 pci 20 rxpci 21 txpci
import sys


def num(x):
    try:
        return float(x)
    except ValueError:
        return None


def analyze(path, label):
    rows = []
    for ln in open(path, errors="replace"):
        if ln.startswith("#"):
            continue
        p = ln.split()
        if len(p) < 22:
            continue
        g = num(p[0])
        if g is None:
            continue
        rows.append({"gpu": int(g), "pwr": num(p[1]), "gtemp": num(p[2]), "sm": num(p[4]),
                     "mclk": num(p[10]), "pclk": num(p[11]), "rx": num(p[20]), "tx": num(p[21])})
    print("\n### %s（%s）共 %d 条" % (label, path, len(rows)))
    for g in (0, 1):
        sel = [r for r in rows if r["gpu"] == g]
        if not sel:
            print("  GPU%d: 无数据" % g)
            continue
        act = [r for r in sel if (r["sm"] or 0) > 20 or (r["pwr"] or 0) > 50]
        print("  GPU%d 全部 n=%d：功耗 均值%.0f 峰%.0f W | 温度峰 %.0f C | sm均值%.0f%% | pclk 均值%.0f 峰%.0f MHz | PCIe rx 均%.0f 峰%.0f MB/s | tx 均%.0f 峰%.0f MB/s"
              % (g, len(sel), avg(sel, "pwr"), mx(sel, "pwr"), mx(sel, "gtemp"), avg(sel, "sm"),
                 avg(sel, "pclk"), mx(sel, "pclk"), avg(sel, "rx"), mx(sel, "rx"),
                 avg(sel, "tx"), mx(sel, "tx")))
        if act:
            print("     活跃子集 n=%d：功耗 均值%.0f 峰%.0f W | sm均值%.0f%% | pclk 均值%.0f 最低%.0f 峰%.0f MHz | PCIe rx 均%.0f 峰%.0f | tx 均%.0f 峰%.0f MB/s"
                  % (len(act), avg(act, "pwr"), mx(act, "pwr"), avg(act, "sm"),
                     avg(act, "pclk"), mn(act, "pclk"), mx(act, "pclk"),
                     avg(act, "rx"), mx(act, "rx"), avg(act, "tx"), mx(act, "tx")))


def vals(rows, k):
    return [r[k] for r in rows if r[k] is not None]


def avg(rows, k):
    v = vals(rows, k)
    return sum(v) / len(v) if v else 0.0


def mx(rows, k):
    v = vals(rows, k)
    return max(v) if v else 0.0


def mn(rows, k):
    v = vals(rows, k)
    return min(v) if v else 0.0


for i in range(1, len(sys.argv), 2):
    analyze(sys.argv[i], sys.argv[i + 1] if i + 1 < len(sys.argv) else sys.argv[i])
