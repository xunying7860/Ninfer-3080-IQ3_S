#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_fake_dump.py —— 手搓一条最小 KVD1 dump + 配套 KVO1（引擎解码输出）。

目的：证明 --from-dump / --decoded 两个入口不是纸面的 —— 字节由本脚本按格式逐字段
写死，解码结果可以解析式预判，任何工具侧读错都会立刻显形。

构造（解析式可预期）：
  bits=4, rows=cols=128
  q[r][c] = (r + c + 16*slice) & 0xF      # 低位先打包（beellama/引擎同一约定）
  scale_axis[r] = 1.0, zp_axis[r] = 0.0, other_axis[c] = 2.0   # 三者都是 fp16 精确值
  => 解码值 = (q*1 + 0)*2 = 2q           # fp32 精确
  => 主机反量化 == record 解码 == 2q；原值也置 2q ⇒ 三个 cosine 全 1.0

产物：
  <out>.dump   KVD1：4 条 record = [K s0, K s1, V s0, V s1]
  <out>.dec    KVO1：2 个 head（K/V），值为"正确解码"，domain=0（旋转域）  ⇒ 应 PASS
  <out>.bad    KVO1：把 head0 的 token 7 整体取反 ⇒ cos=-1，应被判定为解码端病
"""
import argparse
import struct
import sys

ROWS = COLS = 128
BITS = 4
PAYLOAD = ROWS * COLS * BITS // 8          # 8192
KVD1_MAGIC = b"KVD1"
KVO1_MAGIC = b"KVO1"


def h16(x):
    """fp16 舍入再回 fp32（只需要 1.0 / 0.0 / 2.0 这类精确值，用 struct 即可）"""
    return struct.unpack("<e", struct.pack("<e", x))[0]


def pack_codes(slice_id):
    """per-row 低位先打包：byte[r*64 + c/2]，偶 c 在低半字节"""
    buf = bytearray(PAYLOAD)
    for r in range(ROWS):
        base = r * (COLS // 2)
        for c in range(COLS):
            q = q_of(r, c, slice_id)
            if c % 2 == 0:
                buf[base + c // 2] |= q          # 低半字节
            else:
                buf[base + c // 2] |= q << 4     # 高半字节
    return bytes(buf)


def q_of(r, c, slice_id):
    return (r * 3 + c * 5 + 7 * slice_id + 1) & 0xF


def rec_bytes(slice_id):
    """KVD1 一条 record 的两段：code bytes + scale halfs(rows sc | rows zp | cols other)"""
    code = pack_codes(slice_id)
    scale = struct.pack("<%de" % (2 * ROWS + COLS),
                        *([1.0] * ROWS + [0.0] * ROWS + [2.0] * COLS))
    return code, scale


def orig_floats(is_v, slice_id):
    """量化前原值（旋转域）。统一口径：tile[row][col] 的码 = q_of(row,col,s)，
    原值 = 2q，按 tile 行主序存。is_v 只影响 row/col 的语义，不影响这里。"""
    return [2.0 * q_of(i, j, slice_id) for i in range(ROWS) for j in range(COLS)]


def write_kvd1(path):
    recs = []                                  # (name, kind, orig)
    for kind, is_v in ((0, False), (1, True)):
        for s in range(2):
            recs.append(("fake_%s_s%d" % ("V" if is_v else "K", s), kind, s, orig_floats(is_v, s)))
    with open(path, "wb") as f:
        f.write(KVD1_MAGIC + struct.pack("<II", 1, len(recs)))
        for name, kind, s, orig in recs:
            code, scale = rec_bytes(s)
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)) + nb)
            f.write(struct.pack("<7i", kind, BITS, ROWS, COLS, ROWS, 2, 2))
            f.write(struct.pack("<I", 1))                       # flags: bit0 = 带原值
            f.write(struct.pack("<I", len(code)) + code)
            f.write(struct.pack("<I", len(scale) // 2) + scale)
            f.write(struct.pack("<I", len(orig)) + struct.pack("<%df" % len(orig), *orig))
    return len(recs)


def engine_decoded_heads():
    """引擎口径解码：K 的 k[t][d]，V 的 v[t][d]，d = s*128+dl"""
    heads = []
    for kind, is_v in ((0, False), (1, True)):
        data = []
        for t in range(ROWS):
            for d in range(256):
                s, dl = d // 128, d % 128
                if is_v:                                        # V: rows=token, cols=dim
                    data.append(2.0 * q_of(t, dl, s))
                else:                                           # K: rows=dim, cols=token
                    data.append(2.0 * q_of(dl, t, s))
        heads.append((("fake_%s" % ("V" if is_v else "K")), kind, data))
    return heads


def write_kvo1(path, poison_head0):
    heads = engine_decoded_heads()
    with open(path, "wb") as f:
        f.write(KVO1_MAGIC + struct.pack("<II", 1, len(heads)))
        for name, kind, data in heads:
            d = list(data)
            if poison_head0 and kind == 0:                      # 把 K 的 token 7 整体取反
                for i in range(7 * 256, 8 * 256):
                    d[i] = -d[i]
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)) + nb)
            f.write(struct.pack("<4i", kind, ROWS, 256, 0))     # domain=0（旋转域）
            f.write(struct.pack("<I", len(d)) + struct.pack("<%df" % len(d), *d))
    return len(heads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="KVD1 dump 输出路径（.dec/.bad 由它派生）")
    a = ap.parse_args()
    n = write_kvd1(a.out)
    dec = a.out + ".dec"
    bad = a.out + ".bad"
    h = write_kvo1(dec, poison_head0=False)
    hb = write_kvo1(bad, poison_head0=True)
    print("[fake] %s : %d records（K s0/s1 + V s0/s1，bits=4，rows=cols=128）" % (a.out, n))
    print("[fake] %s : %d heads（正确引擎解码，domain=0）      -> 应 PASS" % (dec, h))
    print("[fake] %s : %d heads（K 的 token7 取反）           -> 应 FAIL(解码端)" % (bad, hb))
    print("[fake] 解析式预期：解码值 = (q*1 + 0)*2 = 2q，q=(3r+5c+7s+1)&15；"
          "三路 cosine 全 1.0，maxabs=0，nonfinite=0")


if __name__ == "__main__":
    main()
