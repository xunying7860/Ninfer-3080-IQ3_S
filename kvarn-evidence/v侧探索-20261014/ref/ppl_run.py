#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 0 正式 Gate 的 PPL 臂：同一份真实长文档、同一批 token，比四种 KV 量化下的平均 NLL。

四臂：
  baseline    不量化（这份 checkpoint 自己的真值）
  rk4v4       现生产口径
  kvarn_k4v2  上游口径（V 以 token 为行）
  kvarnK+V4c  kvarn 的 K + "V 通道为行"4bit（上一轮 attention 层唯一过 0.999 的组合）

做法：在 16 个 self_attn 层上给 k_proj/v_proj 挂 forward hook，把输出换成 fake-quant 后的版本
（口径与 compare_real.py 一致：都喂 pre-RoPE 的投影输出，两种 codec 都自带旋转），
再用 lm_head（一次性反量化到 GPU，fp16 约 2.5 GB）算 next-token logprob，取平均 NLL 与 PPL。

判据（来自 `阶段0-执行计划.md`）：候选 PPL ≤ rk4v4 × (1 + 0.1%)。
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/data/kvarn")
from embed_from_ckpt import build_inputs_embeds  # noqa: E402
from kv_codecs import kvarn_fakequant, rk4v4_fakequant  # noqa: E402
from v_diag import sinkhorn_v_channel  # noqa: E402

SRC = "/data/models/Qwen3.8-27B-3Bit-GSQ"
MODEL = os.environ.get("GSQ_MODEL", "/data/kvarn/model-shim")
DOC = os.environ.get("PPL_DOC", "/data/kvarn/ppl_doc.txt")
MAX_TOK = int(os.environ.get("PPL_TOK", "3000"))
D = 256
N_KV = 4

ARM = "baseline"


def load_model():
    from transformers import AutoConfig, AutoTokenizer  # noqa: F401
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration as C
    import transformers.modeling_utils as mu
    # 见 references/qwen35-hf-load-x99.md：auto 会挑错类；0.15 会把 packed 全判 UNEXPECTED；
    # _initialize_missing_keys 会给压缩占位参数随机初始化 ⇒ 掐掉，之后扫残留 meta 落零并计数
    mu.PreTrainedModel._initialize_missing_keys = lambda self, *a, **k: None
    C._init_weights = lambda self, m: None
    m = C.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float16,
                          device_map={"": 0}, low_cpu_mem_usage=True)
    m.eval()
    for owner_name in ("model", None):
        owner = m.model if owner_name else m
        for nm, _ in list(owner.named_children()):
            if any(k in nm.lower() for k in ("visual", "vision", "mtp")):
                setattr(owner, nm, torch.nn.Identity())
    n_meta = 0
    for nm, p in list(m.named_parameters()):
        if getattr(p.device, "type", "cpu") == "meta":
            n_meta += 1
    print("残留 meta:", n_meta, "(应为 0)", flush=True)

    from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor as PQC
    import torch.nn.functional as F

    def mk(mod):
        scheme = getattr(mod, "quantization_scheme", None)
        if scheme is None or not hasattr(mod, "weight_packed"):
            return
        if getattr(scheme, "input_activations", None) is not None or \
           getattr(scheme, "output_activations", None) is not None:
            return
        emb = isinstance(mod, torch.nn.Embedding)

        def fwd(x, _m=mod, _s=scheme, _e=emb):
            sd = {"weight_packed": _m.weight_packed.data, "weight_scale": _m.weight_scale.data}
            for extra in ("weight_zero_point", "weight_shape"):
                if hasattr(_m, extra):
                    sd[extra] = getattr(_m, extra).data
            w = PQC.decompress(sd, _s)["weight"]
            b = getattr(_m, "bias", None)
            b = b.data if b is not None else None
            return F.embedding(x, w, padding_idx=getattr(_m, "padding_idx", None)) if _e else F.linear(x, w, b)
        mod.forward = fwd
    n = 0
    for nm, md in m.named_modules():
        if isinstance(md, (torch.nn.Linear, torch.nn.Embedding)):
            mk(md)
            n += 1
    print("现场反量化 forward 覆盖模块:", n, flush=True)
    layers = m.model.language_model.layers
    return m, layers


def lm_head_weight(m):
    """lm_head 4bit/group64/symmetric：一次性反量化成 fp16 放 GPU（约 2.5 GB），PPL 要用整行词表。"""
    idx = json.load(open(SRC + "/model.safetensors.index.json"))["weight_map"]
    name = [k for k in idx if k.endswith("lm_head.weight_packed")][0]
    from safetensors import safe_open
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    with safe_open(SRC + "/" + idx[name], framework="pt") as f:
        packed = f.get_tensor(name)
        scale = f.get_tensor(name[: -len("weight_packed")] + "weight_scale")
        shape = [int(x) for x in f.get_tensor(name[: -len("weight_packed")] + "weight_shape").reshape(-1)]
    w = unpack_from_int32(packed, 4, torch.Size(shape))
    w = w.to(torch.float16) * scale.to(torch.float16).repeat_interleave(64, dim=-1)[:, : shape[1]]
    print("lm_head 反量化:", tuple(w.shape), w.dtype, flush=True)
    return w.to("cuda:0")


def quant_hook(part):
    def f(_mod, _inp, out):
        T = out.shape[1]
        # 一律搬 CPU 做 fake-quant：kv_codecs 里 rk4v4 的 Lloyd-Max 码本是 CPU 上的常量，
        # 与 GPU 张量混算会 RuntimeError: found at least two devices, cuda:0 and cpu!
        x = out.reshape(1, T, N_KV, D).permute(0, 2, 1, 3).contiguous().squeeze(0).float().cpu()  # [4,T,256]
        if ARM == "baseline":
            return out
        if ARM == "rk4v4":
            k_hat, v_hat, _ = rk4v4_fakequant(x, x)
        elif ARM == "kvarn_k4v2":
            k_hat, v_hat, _ = kvarn_fakequant(x, x, tile_tokens=64, bits_k=4, bits_v=2)
        elif ARM == "rkV4c":
            # 零容量代价的升级：K 保持现生产口径，只把 V 换成"通道为行 + Sinkhorn"的 4bit
            # （上一轮 V 逐向量 cosine 0.995266 -> 0.998630，字节数一模一样）
            k_hat, _v0, _ = rk4v4_fakequant(x, x)
            v_hat = sinkhorn_v_channel(x, 4)
        elif ARM == "kvarnK+rkV":
            # kvarn 的 K（4bit+Sinkhorn，146 B）+ 现生产的 V（G32 signed int4，76 B）= 222 B
            # 这是真正的目标组合：K 侧更准、V 侧保持现役质量 ⇒ 同质量省 20.7% 容量
            k_hat, _v, _ = kvarn_fakequant(x, x, tile_tokens=64, bits_k=4, bits_v=2)
            _k2, v_hat, _ = rk4v4_fakequant(x, x)
        elif ARM == "kvarnK+V4c":
            k_hat, _v, _ = kvarn_fakequant(x, x, tile_tokens=64, bits_k=4, bits_v=4)
            v_hat = sinkhorn_v_channel(x, 4)
        else:
            raise SystemExit("未知臂 " + ARM)
        sel = k_hat if part == "k" else v_hat
        return sel.to(device=out.device, dtype=out.dtype).permute(1, 0, 2).reshape(1, T, N_KV * D)
    return f


def run_arm(m, layers, ids, W):
    global ARM
    handles = []
    for L in layers:
        attn = getattr(L, "self_attn", None)
        if attn is None:
            continue
        for part, nm in (("k", "k_proj"), ("v", "v_proj")):
            h = getattr(attn, nm).register_forward_hook(quant_hook(part))
            handles.append(h)
    emb = build_inputs_embeds(ids)
    with torch.no_grad():
        hs = m.model.language_model(inputs_embeds=emb).last_hidden_state[0]      # [T, 5120]
        # 逐块算 logits，省显存；只需真实 token 的 logit 与整词表的 logsumexp
        nll, n = 0.0, 0
        for s in range(0, hs.shape[0] - 1, 256):
            e = min(s + 256, hs.shape[0] - 1)
            lg = hs[s:e].to(torch.float16) @ W.t()                               # [b, vocab]
            tgt = torch.tensor(ids[s + 1:e + 1], device=lg.device)
            tok_lg = lg.gather(1, tgt[:, None]).squeeze(1).float()
            lse = torch.logsumexp(lg.float(), dim=-1)
            nll += (lse - tok_lg).sum().item()
            n += tgt.numel()
    for h in handles:
        h.remove()
    mean_nll = nll / n
    import math
    return mean_nll, math.exp(mean_nll)


def main():
    import re
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
    text = open(DOC, encoding="utf-8", errors="ignore").read()
    text = re.sub(r"\s+", " ", text)[:20000]
    ids = tok(text, add_special_tokens=False)["input_ids"][:MAX_TOK]
    # 对齐到 64 的整数倍：kvarn 的 tile=64 里有 assert T % tile == 0（实测 3000 % 64 = 40 直接 AssertionError）
    ids = ids[:(len(ids) // 64) * 64]
    print("PPL 语料: %d 字符 → %d token（取前 %d）" % (len(text), len(ids), MAX_TOK), flush=True)
    print("语料开头:", text[:60], flush=True)

    m, layers = load_model()
    n_self = sum(1 for L in layers if hasattr(L, "self_attn"))
    print("self_attn 层数(挂量化 hook):", n_self, flush=True)
    W = lm_head_weight(m)
    if torch.cuda.is_available():
        print("GPU0 已用 MiB:", torch.cuda.memory_allocated(0) // 2**20, flush=True)

    out = {}
    for arm in ("baseline", "rk4v4", "rkV4c", "kvarn_k4v2", "kvarnK+rkV", "kvarnK+V4c"):
        global ARM
        ARM = arm
        try:
            nll, ppl = run_arm(m, layers, ids, W)
            out[arm] = (nll, ppl)
            print("%-14s mean NLL = %.6f   PPL = %.4f" % (arm, nll, ppl), flush=True)
        except Exception as e:
            print("%-14s 失败: %s %s" % (arm, type(e).__name__, str(e)[:200]), flush=True)
    if "rk4v4" in out:
        base = out["rk4v4"][1]
        print("\n=== 判据：候选 PPL ≤ rk4v4 × 1.001 ===")
        for a, (nll, ppl) in out.items():
            d = 100.0 * (ppl / base - 1.0)
            print("  %-14s PPL=%.4f  vs rk4v4 %+.3f%%  ⇒ %s"
                  % (a, ppl, d, "基准" if a == "rk4v4" else ("过" if d <= 0.1 else "不过")))
    print("PPL_RUN_DONE")


if __name__ == "__main__":
    main()
