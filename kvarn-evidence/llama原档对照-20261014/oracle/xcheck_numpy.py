#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xcheck_numpy.py —— 独立交叉校验：把 ggml-quants.c 的 6 个 KV 量化档用 numpy 重写一遍，
与 llama_oracle dump 出来的「码流 hex + 反量化向量」逐位/逐元素对拍。

用法：
    ./llama_oracle dump out/dump.txt      # 容器里
    python xcheck_numpy.py out/dump.txt   # 宿主（需要 numpy）
"""
import sys
import numpy as np

f32 = np.float32
F = np.float32


def fp16(v):
    """__float2half_rn 的等价：float32 -> float16（round-to-nearest-even）"""
    return np.float16(f32(v))


def f16f(h):
    return f32(np.float16(h))


def nearest_int(v):
    """ggml-quants.c:1002"""
    val = f32(f32(v) + f32(12582912.0))
    i = val.view(np.int32) if hasattr(val, "view") else np.int32(0)
    return int((int(i) & 0x007fffff) - 0x00400000)


def i8trunc(v):
    """(int8_t)(float) —— 截断向零"""
    return int(np.trunc(f32(v)))


def rnd_half_away(v):
    """roundf() 的等价（round-half-away-from-zero）"""
    v = f32(v)
    return int(np.trunc(f32(v + np.copysign(f32(0.5), v))))


# ---------------------------------------------------------------- q4_0
def q4_0(x):
    qk = 32
    nb = len(x) // qk
    buf = bytearray()
    deq = np.zeros(len(x), f32)
    for i in range(nb):
        xb = x[i * qk:(i + 1) * qk]
        amax = f32(0.0); mx = f32(0.0)
        for j in range(qk):
            v = xb[j]
            if amax < abs(v):
                amax = abs(v); mx = v
        d = f32(mx / f32(-8.0))
        idd = f32(1.0) / d if d != 0 else f32(0.0)
        dh = fp16(d)
        qs = bytearray()
        for j in range(qk // 2):
            x0 = f32(xb[j] * idd)
            x1 = f32(xb[qk // 2 + j] * idd)
            xi0 = min(15, i8trunc(f32(x0 + f32(8.5))))
            xi1 = min(15, i8trunc(f32(x1 + f32(8.5))))
            qs.append((xi0 & 15) | ((xi1 & 15) << 4))
        buf += np.array([dh], dtype="<f2").tobytes() + bytes(qs)
        df = f16f(dh)
        for j in range(qk // 2):
            q = qs[j]
            deq[i * qk + j] = f32(((q & 15) - 8) * df)
            deq[i * qk + qk // 2 + j] = f32(((q >> 4) - 8) * df)
    return bytes(buf), deq


# ---------------------------------------------------------------- q4_1
def q4_1(x):
    qk = 32
    nb = len(x) // qk
    buf = bytearray()
    deq = np.zeros(len(x), f32)
    for i in range(nb):
        xb = x[i * qk:(i + 1) * qk]
        mn = f32(np.inf); mx = f32(-np.inf)
        for j in range(qk):
            v = xb[j]
            if v < mn: mn = v
            if v > mx: mx = v
        d = f32(f32(mx - mn) / f32(15.0))
        idd = f32(1.0) / d if d != 0 else f32(0.0)
        dh = fp16(d); mh = fp16(mn)
        qs = bytearray()
        for j in range(qk // 2):
            x0 = f32(f32(xb[j] - mn) * idd)
            x1 = f32(f32(xb[qk // 2 + j] - mn) * idd)
            xi0 = min(15, i8trunc(f32(x0 + f32(0.5))))
            xi1 = min(15, i8trunc(f32(x1 + f32(0.5))))
            qs.append((xi0 & 15) | ((xi1 & 15) << 4))
        buf += np.array([dh, mh], dtype="<f2").tobytes() + bytes(qs)
        df = f16f(dh); mf = f16f(mh)
        for j in range(qk // 2):
            q = qs[j]
            deq[i * qk + j] = f32((q & 15) * df + mf)
            deq[i * qk + qk // 2 + j] = f32((q >> 4) * df + mf)
    return bytes(buf), deq


# ---------------------------------------------------------------- q5_0
def q5_0(x):
    qk = 32
    nb = len(x) // qk
    buf = bytearray()
    deq = np.zeros(len(x), f32)
    for i in range(nb):
        xb = x[i * qk:(i + 1) * qk]
        amax = f32(0.0); mx = f32(0.0)
        for j in range(qk):
            v = xb[j]
            if amax < abs(v):
                amax = abs(v); mx = v
        d = f32(mx / f32(-16.0))
        idd = f32(1.0) / d if d != 0 else f32(0.0)
        dh = fp16(d)
        qh = 0
        qs = bytearray()
        for j in range(qk // 2):
            x0 = f32(xb[j] * idd)
            x1 = f32(xb[qk // 2 + j] * idd)
            xi0 = min(31, i8trunc(f32(x0 + f32(16.5))))
            xi1 = min(31, i8trunc(f32(x1 + f32(16.5))))
            qs.append((xi0 & 0x0F) | ((xi1 & 0x0F) << 4))
            qh |= ((xi0 & 0x10) >> 4) << (j + 0)
            qh |= ((xi1 & 0x10) >> 4) << (j + qk // 2)
        buf += np.array([dh], dtype="<f2").tobytes() + np.uint32(qh).astype("<u4").tobytes() + bytes(qs)
        df = f16f(dh)
        for j in range(qk // 2):
            xh0 = ((qh >> (j + 0)) << 4) & 0x10
            xh1 = ((qh >> (j + 12))) & 0x10
            a = ((qs[j] & 0x0F) | xh0) - 16
            b = ((qs[j] >> 4) | xh1) - 16
            deq[i * qk + j] = f32(a * df)
            deq[i * qk + qk // 2 + j] = f32(b * df)
    return bytes(buf), deq


# ---------------------------------------------------------------- q5_1
def q5_1(x):
    qk = 32
    nb = len(x) // qk
    buf = bytearray()
    deq = np.zeros(len(x), f32)
    for i in range(nb):
        xb = x[i * qk:(i + 1) * qk]
        mn = f32(np.inf); mx = f32(-np.inf)
        for j in range(qk):
            v = xb[j]
            if v < mn: mn = v
            if v > mx: mx = v
        d = f32(f32(mx - mn) / f32(31.0))
        idd = f32(1.0) / d if d != 0 else f32(0.0)
        dh = fp16(d); mh = fp16(mn)
        qh = 0
        qs = bytearray()
        for j in range(qk // 2):
            x0 = f32(f32(xb[j] - mn) * idd)
            x1 = f32(f32(xb[qk // 2 + j] - mn) * idd)
            xi0 = i8trunc(f32(x0 + f32(0.5))) & 0xFF   # (uint8_t)(x0+0.5f)
            xi1 = i8trunc(f32(x1 + f32(0.5))) & 0xFF
            qs.append((xi0 & 0x0F) | ((xi1 & 0x0F) << 4))
            qh |= ((xi0 & 0x10) >> 4) << (j + 0)
            qh |= ((xi1 & 0x10) >> 4) << (j + qk // 2)
        buf += np.array([dh, mh], dtype="<f2").tobytes() + np.uint32(qh).astype("<u4").tobytes() + bytes(qs)
        df = f16f(dh); mf = f16f(mh)
        for j in range(qk // 2):
            xh0 = ((qh >> (j + 0)) << 4) & 0x10
            xh1 = ((qh >> (j + 12))) & 0x10
            a = (qs[j] & 0x0F) | xh0
            b = (qs[j] >> 4) | xh1
            deq[i * qk + j] = f32(a * df + mf)
            deq[i * qk + qk // 2 + j] = f32(b * df + mf)
    return bytes(buf), deq


# ---------------------------------------------------------------- q8_0
def q8_0(x):
    qk = 32
    nb = len(x) // qk
    buf = bytearray()
    deq = np.zeros(len(x), f32)
    for i in range(nb):
        xb = x[i * qk:(i + 1) * qk]
        amax = f32(0.0)
        for j in range(qk):
            amax = max(amax, abs(xb[j]))
        d = f32(amax / f32(127.0))
        idd = f32(1.0) / d if d != 0 else f32(0.0)
        dh = fp16(d)
        qs = np.zeros(qk, np.int8)
        for j in range(qk):
            qs[j] = np.int8(rnd_half_away(f32(xb[j] * idd)))
        buf += np.array([dh], dtype="<f2").tobytes() + qs.tobytes()
        df = f16f(dh)
        for j in range(qk):
            deq[i * qk + j] = f32(int(qs[j]) * df)
    return bytes(buf), deq


# ---------------------------------------------------------------- q4_K
def make_qkx2_quants(n, nmax, x, weights, rmin, rdelta, nstep, use_mad):
    """ggml-quants.c:1180 逐行"""
    mn = x[0]; mx = x[0]
    sum_w = weights[0]
    sum_x = f32(sum_w * x[0])
    for i in range(1, n):
        if x[i] < mn: mn = x[i]
        if x[i] > mx: mx = x[i]
        w = weights[i]
        sum_w = f32(sum_w + w)
        sum_x = f32(sum_x + f32(w * x[i]))
    if mn > 0:
        mn = f32(0.0)
    if mx == mn:
        return f32(0.0), f32(-mn), np.zeros(n, np.uint8)
    iscale = f32(f32(nmax) / f32(mx - mn))
    scale = f32(f32(1.0) / iscale)
    best_error = f32(0.0)
    L = np.zeros(n, np.uint8)
    for i in range(n):
        l = nearest_int(f32(iscale * f32(x[i] - mn)))
        L[i] = max(0, min(nmax, l))
        diff = f32(f32(scale * L[i]) + mn - x[i])
        diff = abs(diff) if use_mad else f32(diff * diff)
        best_error = f32(best_error + f32(weights[i] * diff))
    if nstep < 1:
        return scale, f32(-mn), L
    for is_ in range(nstep + 1):
        iscale = f32(f32(f32(rmin) + f32(f32(rdelta) * f32(is_)) + f32(nmax)) / f32(mx - mn))
        sum_l = f32(0.0); sum_l2 = f32(0.0); sum_xl = f32(0.0)
        Laux = np.zeros(n, np.uint8)
        for i in range(n):
            l = nearest_int(f32(iscale * f32(x[i] - mn)))
            l = max(0, min(nmax, l))
            Laux[i] = l
            w = weights[i]
            sum_l = f32(sum_l + f32(w * l))
            sum_l2 = f32(sum_l2 + f32(w * l * l))
            sum_xl = f32(sum_xl + f32(w * f32(l * x[i])))
        D = f32(f32(sum_w * sum_l2) - f32(sum_l * sum_l))
        if D > 0:
            this_scale = f32(f32(f32(sum_w * sum_xl) - f32(sum_x * sum_l)) / D)
            this_min = f32(f32(f32(sum_l2 * sum_x) - f32(sum_l * sum_xl)) / D)
            if this_min > 0:
                this_min = f32(0.0)
                this_scale = f32(sum_xl / sum_l2)
            cur_error = f32(0.0)
            for i in range(n):
                diff = f32(f32(this_scale * Laux[i]) + this_min - x[i])
                diff = abs(diff) if use_mad else f32(diff * diff)
                cur_error = f32(cur_error + f32(weights[i] * diff))
            if cur_error < best_error:
                L = Laux.copy()
                best_error = cur_error
                scale = this_scale
                mn = this_min
    return scale, f32(-mn), L


def get_scale_min_k4(j, q):
    """ggml-quants.c:1261"""
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def q4_K(x):
    assert len(x) == 256
    L = np.zeros(256, np.uint8)
    Laux = np.zeros(32, np.uint8)
    mins = np.zeros(8, f32)
    scales = np.zeros(8, f32)
    max_scale = f32(0.0); max_min = f32(0.0)
    for j in range(8):
        xb = x[32 * j:32 * j + 32]
        sum_x2 = f32(0.0)
        for l in range(32):
            sum_x2 = f32(sum_x2 + f32(xb[l] * xb[l]))
        av_x = f32(np.sqrt(f32(sum_x2 / f32(32.0))))
        weights = np.array([f32(av_x + abs(xb[l])) for l in range(32)], f32)
        sc, m, Ll = make_qkx2_quants(32, 15, xb, weights, -1.0, 0.1, 20, False)
        scales[j] = sc; mins[j] = m
        L[32 * j:32 * j + 32] = Ll
        if sc > max_scale: max_scale = sc
        if m > max_min: max_min = m
    inv_scale = f32(f32(63.0) / max_scale) if max_scale > 0 else f32(0.0)
    inv_min = f32(f32(63.0) / max_min) if max_min > 0 else f32(0.0)
    scp = np.zeros(12, np.uint8)
    for j in range(8):
        ls = min(63, nearest_int(f32(inv_scale * scales[j])))
        lm = min(63, nearest_int(f32(inv_min * mins[j])))
        if j < 4:
            scp[j] = ls; scp[j + 4] = lm
        else:
            scp[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4)
            scp[j - 4] |= (ls >> 4) << 6
            scp[j] |= (lm >> 4) << 6
    dh = fp16(f32(max_scale / f32(63.0)))
    dmh = fp16(f32(max_min / f32(63.0)))
    for j in range(8):
        sc, m = get_scale_min_k4(j, scp)
        dd = f32(f16f(dh) * sc)
        if dd == 0:
            continue
        dm = f32(f16f(dmh) * m)
        for ii in range(32):
            l = nearest_int(f32(f32(x[32 * j + ii] + dm) / dd))
            L[32 * j + ii] = max(0, min(15, l))
    qs = np.zeros(128, np.uint8)
    for j in range(0, 256, 64):
        for l in range(32):
            qs[(j // 64) * 32 + l] = L[j + l] | (L[j + l + 32] << 4)
    buf = np.array([dh, dmh], dtype="<f2").tobytes() + scp.tobytes() + qs.tobytes()
    # dequant
    deq = np.zeros(256, f32)
    dd = f16f(dh); dmin = f16f(dmh)
    is_ = 0; out = 0
    for j in range(0, 256, 64):
        sc, m = get_scale_min_k4(is_ + 0, scp); d1 = f32(dd * sc); m1 = f32(dmin * m)
        sc, m = get_scale_min_k4(is_ + 1, scp); d2 = f32(dd * sc); m2 = f32(dmin * m)
        for l in range(32):
            deq[out] = f32(f32(d1 * (qs[(j // 64) * 32 + l] & 0xF)) - m1); out += 1
        for l in range(32):
            deq[out] = f32(f32(d2 * (qs[(j // 64) * 32 + l] >> 4)) - m2); out += 1
        is_ += 2
    return buf, deq


CODECS = {"q4_0": q4_0, "q4_1": q4_1, "q5_0": q5_0, "q5_1": q5_1, "q8_0": q8_0, "q4_K": q4_K}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out/dump.txt"
    x = None
    out = {}
    hexs = {}
    for ln in open(path, encoding="utf-8"):
        t = ln.split()
        if not t:
            continue
        if t[0] == "in":
            x = np.array([f32(v) for v in t[1:]], f32)
        elif t[0] == "out":
            out[t[1]] = (int(t[2]), np.array([f32(v) for v in t[3:]], f32))
        elif t[0] == "hex":
            hexs[t[1]] = t[2]
    if x is None:
        print("没读到 in 行"); return 2
    print("n=%d，档=%s\n" % (len(x), ",".join(out)))
    bad = 0
    for name, fn in CODECS.items():
        buf, deq = fn(x)
        nbytes, ref = out[name]
        hexok = buf.hex() == hexs[name]
        d = np.abs(deq.astype(np.float64) - ref.astype(np.float64))
        maxd = float(d.max())
        rel = maxd / max(1e-30, float(np.abs(ref.astype(np.float64)).max()))
        ok = hexok and maxd < 1e-6
        if not ok:
            bad += 1
        print("%-5s bytes=%3d(参考 %3d) 码流逐位相同=%s  max|Δdeq|=%.3e  相对峰值=%.3e  %s"
              % (name, len(buf), nbytes, "OK" if hexok else "DIFF", maxd, rel, "PASS" if ok else "FAIL"))
        if not hexok:
            for i in range(min(len(buf), len(hexs[name]) // 2)):
                if buf[i] != int(hexs[name][2 * i:2 * i + 2], 16):
                    print("      首个不同字节 @%d: numpy=%02x oracle=%s" % (i, buf[i], hexs[name][2 * i:2 * i + 2]))
                    break
    print("\n失败档数 = %d" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
