#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_tiles.py —— 把真实 K/V 抓取按 beellama 的 D256 口径旋好、切成 oracle 能读的 tile。

口径严格照 kvarn.cu 的 store 路径（kvarn_store_kernel_*，行 1105-1130）：
  current 形状 [tokens][n_slice_units][128]（即 head_dim=256 已按 record_dim=128 切成 2 片）
  每片先做 128 点 WHT（×1/sqrt(128) = 0.08838834764831845）   -> kvarn_wht_128
  再跨片做长度为 head_slices 的 Hadamard（2 片 ×0.7071067811865475） -> kvarn_wht_cross_slices
  结果写进 F16 stage（F16 舍入由 oracle 的 --stage-f16 负责，这里保持 fp32）

输出（oracle 读的二进制，小端）：
  int32 n_tiles
  repeat: int32 name_len, char name[name_len], int32 rows, int32 cols, int32 token_is_row, float32 data[rows*cols]
K tile: rows=128(dim), cols=G(token), token_is_row=0
V tile: rows=G(token), cols=128(dim), token_is_row=1
"""
import argparse
import hashlib
import os
import struct
import sys

import torch

SRC = r"C:\Users\xunying\AppData\Local\hermes\cache\scratch\kvarn-tiles"
LAYERS = [35, 63]
SRC_FILES = {35: ("layer35_k_proj.pt", "layer35_v_proj.pt"),
             63: ("layer63_k_proj.pt", "layer63_v_proj.pt")}
D = 256
RECORD_DIM = 128           # D256 -> head_slices = 2, record_dim = 128
HEAD_SLICES = 2
INV_SQRT_128 = 0.08838834764831845
INV_SQRT_2 = 0.7071067811865475


def wht128(x: torch.Tensor) -> torch.Tensor:
    """沿最后一维（长度 128）做归一化 WHT，与 kvarn_wht_128 逐行等价。"""
    y = x.clone()
    stride = 1
    while stride < RECORD_DIM:
        a = y[..., 0::2 * stride][..., :stride].clone()
        # 显式 butterfly（按 base 分组），与源码写法一致
        for base in range(0, RECORD_DIM, 2 * stride):
            aa = y[..., base:base + stride].clone()
            bb = y[..., base + stride:base + 2 * stride].clone()
            y[..., base:base + stride] = aa + bb
            y[..., base + stride:base + 2 * stride] = aa - bb
        stride *= 2
        del a
    return y * INV_SQRT_128


def rotate_d256(x: torch.Tensor) -> torch.Tensor:
    """x: [heads, T, 256] -> 已按 beellama 口径旋转的 [heads, T, 256]"""
    h, T, dim = x.shape
    assert dim == D
    v = x.reshape(h, T, HEAD_SLICES, RECORD_DIM)
    v = wht128(v)                                   # 逐片 128 点 WHT
    a = v[..., 0, :]
    b = v[..., 1, :]
    o0 = (a + b) * INV_SQRT_2
    o1 = (a - b) * INV_SQRT_2
    return torch.stack([o0, o1], dim=2).reshape(h, T, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=128, choices=[64, 128])
    ap.add_argument("--out", default=None)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--layers", default="35,63")
    args = ap.parse_args()
    G = args.group
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "data", "real_g%d.bin" % G)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    layers = [int(s) for s in args.layers.split(",")]

    tiles = []          # (name, rows, cols, token_is_row, flatfloat32)
    for lay in layers:
        for tag, fname in (("k", SRC_FILES[lay][0]), ("v", SRC_FILES[lay][1])):
            p = os.path.join(args.src, fname)
            if not os.path.exists(p):
                print("缺文件，跳过:", p, file=sys.stderr)
                continue
            x = torch.load(p, map_location="cpu").float()          # [1, T, heads*256]
            T = x.shape[1]
            heads = x.shape[-1] // D
            x = x.reshape(1, T, heads, D).permute(0, 2, 1, 3).contiguous()[0]  # [heads, T, D]
            xr = rotate_d256(x)                                    # [heads, T, 256]
            # G128 与 G64 都只取前 128 个 token，保证可比
            ntok = min(T, 128)
            for h in range(heads):
                for g in range(ntok // G):
                    t0 = g * G
                    for s in range(HEAD_SLICES):
                        dimlo = s * RECORD_DIM
                        blk = xr[h, t0:t0 + G, dimlo:dimlo + RECORD_DIM]      # [G tok, 128 dim]
                        if tag == "k":
                            tile = blk.transpose(0, 1).contiguous()          # [128 dim, G tok]
                            rows, cols, tok_is_row = RECORD_DIM, G, 0
                        else:
                            tile = blk.contiguous()                          # [G tok, 128 dim]
                            rows, cols, tok_is_row = G, RECORD_DIM, 1
                        name = "L%d_%s_h%d_s%d_g%d" % (lay, tag, h, s, g)
                        tiles.append((name, rows, cols, tok_is_row, tile.reshape(-1).tolist()))

    with open(out, "wb") as f:
        f.write(struct.pack("<i", len(tiles)))
        for name, rows, cols, tok_is_row, data in tiles:
            nb = name.encode("utf-8")
            f.write(struct.pack("<i", len(nb)))
            f.write(nb)
            f.write(struct.pack("<iii", rows, cols, tok_is_row))
            f.write(struct.pack("<%df" % len(data), *data))
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print("写入 %s: %d tiles, G=%d, sha256=%s" % (out, len(tiles), G, h))
    print("tile 例:", tiles[0][0], tiles[0][1], "x", tiles[0][2], "token_is_row=%d" % tiles[0][3])


if __name__ == "__main__":
    main()
