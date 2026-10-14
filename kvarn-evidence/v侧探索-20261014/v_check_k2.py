import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_codecs as V  # noqa: E402

T = r"C:\Users\xunying\AppData\Local\hermes\cache\scratch\kvarn-tiles"


def cos(a, b):
    a = a.reshape(-1, 256); b = b.reshape(-1, 256)
    return ((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)).mean().item()


for lay in (35, 63):
    k = torch.load(os.path.join(T, "layer%d_k_proj.pt" % lay), map_location="cpu").float()[0]
    k = k.reshape(256, 4, 256).permute(1, 0, 2)
    kr = V.hadamard_256(k)
    b, (H2, nt) = V._blocks(kr)
    kb = b.transpose(-1, -2).contiguous()
    bal, sc, sr = V._sinkhorn(kb, 8)
    q, s, z = V._rtn(bal, 4, None)
    rec = (s.float() * q + z.float()) * sc * sr
    ref = V.hadamard_256(V._unblocks(rec.transpose(-1, -2).contiguous(), H2, nt))
    mine = V.k_quant(k, 4, variant="rtn")[0]
    d = (ref - mine).abs().max().item()
    rel = (ref - mine).norm().item() / ref.norm().item()
    ls = V.k_quant(k, 4, variant="ls", quantile=0.01, refine=1)[0]
    ls2 = V.k_quant(k, 4, variant="ls", quantile=0.01, refine=2)[0]
    print("layer%d rtn等价 max|d|=%.2e rel=%.2e | Kcos rtn=%.6f ls+clip1=%.6f ls+clip2=%.6f (增益 %+.6f / %+.6f)"
          % (lay, d, rel, cos(k, ref), cos(k, ls), cos(k, ls2), cos(k, ls) - cos(k, ref), cos(k, ls2) - cos(k, ref)))
