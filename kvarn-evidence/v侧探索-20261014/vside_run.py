#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V 侧改进臂的 PPL oracle（x99，GPU0 短窗口）。

口径与 /data/kvarn/g128/stage0_run.py 完全一致（同一 load_model / lm_head / post-RoPE /
同语料同位置），只把 V 侧的编解码器换成 v_codecs.py 的变体；K 侧固定 g128 k4（与 K4V4/K4V3 锚点同侧）。

臂表（bytes 为 record 口径 32*(Kb+Vb)+24；括号内为诚实 side info 口径）：
  baseline_fp16 / rk4v4_post / K4V4_g128 / K4V3_g128 / V2only_g128   ← 锚点复现
  K4V3chan_ls1      V 通道为行 3bit + 加权 LS 尺度 + 1 轮重取整       248 (244)
  K4V3chan_lsq01r1  同上 + 1% 分位数裁剪                              248 (244)
  K4V3chan_ls2      同上 refine=2                                     248 (244)
  K4V3chan_m4333    4 head 位宽 [4,3,3,3] 通道为行 + LS1              256
  K4V3chan_m4433    4 head 位宽 [4,4,3,3] 通道为行 + LS1              264
  K4V3tok_lsq01r1   V token 为行 3bit + LS + 裁剪（免费旋钮对照）      248 (248)
  K4V2chan_ls4      V 通道为行 2bit + LS refine=4                     216 (212)

安全：进程内看门狗——一旦发现 x99 上有 ninfer 引擎进程起来，立即自杀释放显存。
"""
import math
import os
import re
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, "/data/kvarn")
sys.path.insert(0, "/data/kvarn/g128")
sys.path.insert(0, "/data/kvarn/vside-20261014")

from ppl_run import load_model, lm_head_weight, build_inputs_embeds, D, N_KV  # noqa: E402
import g128_codecs as G  # noqa: E402
import v_codecs as V  # noqa: E402

ARMS = {
    "baseline_fp16":  dict(k=None, v=None, post=False),
    "rk4v4_post":     dict(k="rk", v="rk", post=True),
    "K4V4_g128":      dict(k="k4", v="v4tok", post=True),
    "K4V3_g128":      dict(k="k4", v="v3tok", post=True),
    "V2only_g128":    dict(k=None, v="v2tok", post=True),
    "K4V3chan_ls1":   dict(k="k4", v=("chan", 3, dict(variant="ls", refine=1)), post=True),
    "K4V3chan_lsq01r1": dict(k="k4", v=("chan", 3, dict(variant="ls", quantile=0.01, refine=1)), post=True),
    "K4V3chan_ls2":   dict(k="k4", v=("chan", 3, dict(variant="ls", refine=2)), post=True),
    "K4V3chan_m4333": dict(k="k4", v=("mix", [4, 3, 3, 3], dict(mode="channel", variant="ls", refine=1)), post=True),
    "K4V3chan_m4433": dict(k="k4", v=("mix", [4, 4, 3, 3], dict(mode="channel", variant="ls", refine=1)), post=True),
    "K4V3tok_lsq01r1": dict(k="k4", v=("tok", 3, dict(variant="ls", quantile=0.01, refine=1)), post=True),
    "K4V2chan_ls4":   dict(k="k4", v=("chan", 2, dict(variant="ls", refine=4)), post=True),
    "K4V3chan_lsq005r2": dict(k="k4", v=("chan", 3, dict(variant="ls", quantile=0.005, refine=2)), post=True),
    "K4V3chan_lsq01r2": dict(k="k4", v=("chan", 3, dict(variant="ls", quantile=0.01, refine=2)), post=True),
    "K4Vprom_k16":    dict(k="k4", v=("prom", 16, dict(quantile=0.01, refine=1)), post=True),
    "K4Vprom_k8":     dict(k="k4", v=("prom", 8, dict(quantile=0.01, refine=1)), post=True),
    "K4Vprom_k32":    dict(k="k4", v=("prom", 32, dict(quantile=0.01, refine=1)), post=True),
    "K4Vprom_k64":    dict(k="k4", v=("prom", 64, dict(quantile=0.01, refine=1)), post=True),
    "K4Vprom_k128":   dict(k="k4", v=("prom", 128, dict(quantile=0.01, refine=1)), post=True),
}
CAP_LAYERS = []
ARM = "baseline_fp16"
RS = {"cos": None, "sin": None}
COS = {}
CAP = {}
_DBG = set()
WATCH = os.environ.get("VSIDE_WATCH", "1") == "1"


def _gpu0_foreign():
    """返回正在用 GPU0 且不是我自己的进程（uuid→pid）。"""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10).stdout
        uuid0 = None
        for ln in q.strip().splitlines():
            parts = [s.strip() for s in ln.split(",")]
            if len(parts) == 2 and parts[0] == "0":
                uuid0 = parts[1]
        if uuid0 is None:
            return []
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10).stdout
        mine = {os.getpid(), os.getppid()}
        out = []
        for ln in a.strip().splitlines():
            parts = [s.strip() for s in ln.split(",")]
            if len(parts) == 2 and parts[0] == uuid0 and parts[1].isdigit() and int(parts[1]) not in mine:
                out.append(parts[1])
        return out
    except Exception:
        return []


def watchdog():
    """发现引擎进程（或 GPU0 上有别人的进程）即自杀，把 GPU0 让回去。"""
    PAT = re.compile(r"ld-linux-x86-64|ninfer-serve|ninfer-perplexity")
    while True:
        time.sleep(3)
        try:
            ps = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=10).stdout
            hits = [ln for ln in ps.splitlines()
                    if PAT.search(ln) and not re.search(r"vside|pgrep|nvidia-smi|ps -eo|watchdog", ln)]
            foreign = _gpu0_foreign()
        except Exception:
            continue
        if hits or foreign:
            print("\n[watchdog] GPU0 被占用，立即退出释放显存：\n%s\n%s"
                  % ("\n".join(hits[:5]), "foreign pids on GPU0: %s" % foreign), flush=True)
            os._exit(9)


def _rh(t):
    h = t.shape[-1] // 2
    return torch.cat((-t[..., h:], t[..., :h]), dim=-1)


def rope_on(x, cos, sin):
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    return torch.cat([xr * cos + _rh(xr) * sin, xp], dim=-1)


def rope_off(x, cos, sin):
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    return torch.cat([xr * cos - _rh(xr) * sin, xp], dim=-1)


def vec_cos(a, b):
    a2 = a.reshape(-1, D)
    b2 = b.reshape(-1, D)
    return ((a2 * b2).sum(-1) / (a2.norm(dim=-1) * b2.norm(dim=-1) + 1e-12)).mean().item()


def quant_v(spec, x):
    if isinstance(spec, tuple):
        kind = spec[0]
        if kind == "chan":
            return V.v_quant(x, bits=spec[1], mode="channel", **spec[2])[0]
        if kind == "tok":
            return V.v_quant(x, bits=spec[1], mode="token", **spec[2])[0]
        if kind == "mix":
            return V.v_quant_mixed(x, spec[1], **spec[2])[0]
        if kind == "prom":
            return V.v_quant_promote(x, 3, 4, spec[1], **spec[2])[0]
    if spec == "v2tok":
        return G.g128_quant_v(x, 2, iters=8, mode="token")
    if spec == "v3tok":
        return G.g128_quant_v(x, 3, iters=8, mode="token")
    if spec == "v4tok":
        return G.g128_quant_v(x, 4, iters=8, mode="token")
    raise SystemExit("未知 V spec " + str(spec))


def quantize(part, x):
    spec = ARMS[ARM]["k" if part == "k" else "v"]
    if spec is None:
        return x
    if spec == "rk":
        kh, vh, _ = __import__("kv_codecs").rk4v4_fakequant(x, x)
        return kh if part == "k" else vh
    if spec == "k4":
        return G.g128_quant_k(x, 4, iters=8)
    return quant_v(spec, x)


def make_hook(part, layer_idx):
    def f(_m, _i, out):
        T = out.shape[1]
        x = out.reshape(1, T, N_KV, D).permute(0, 2, 1, 3).contiguous().squeeze(0).float().cpu()
        post = ARMS[ARM]["post"]
        c = s = None
        if post and part == "k" and RS["cos"] is not None:
            c = RS["cos"][:T].unsqueeze(0)
            s = RS["sin"][:T].unsqueeze(0)
            x = rope_on(x, c, s)
        q = quantize(part, x)
        if ARM not in _DBG:
            _DBG.add(ARM)
            print("   [dbg] 臂=%s part=%s T=%d" % (ARM, part, T), flush=True)
        if part == "k":
            COS.setdefault(ARM, {}).setdefault("k", []).append(vec_cos(x, q))
        else:
            COS.setdefault(ARM, {}).setdefault("v", []).append(vec_cos(x, q))
        if layer_idx in CAP_LAYERS:
            rec = CAP.setdefault(ARM, {}).setdefault(layer_idx, {"k": None, "v": None, "q": None,
                                                                "khat": None, "vhat": None})
            rec[part] = x.clone()
            rec[part + "hat"] = q.clone()
        if post and part == "k" and c is not None:
            q = rope_off(q, c, s)
        return q.to(device=out.device, dtype=out.dtype).permute(1, 0, 2).reshape(1, T, N_KV * D)
    return f


def attn_cos(ref, cur, n_q=512, chunk=128):
    q0, k0, v0 = ref["q"], ref["khat"], ref["vhat"]
    q1, k1, v1 = cur["q"], cur["khat"], cur["vhat"]
    T = q0.shape[1]
    st = max(0, T - n_q)
    rep = q0.shape[0] // k0.shape[0]
    k0 = k0.repeat_interleave(rep, 0); v0 = v0.repeat_interleave(rep, 0)
    k1 = k1.repeat_interleave(rep, 0); v1 = v1.repeat_interleave(rep, 0)
    sc = 1.0 / math.sqrt(D)
    keys = torch.arange(T)
    sims = []
    for s in range(st, T, chunk):
        e = min(s + chunk, T)
        mask = torch.arange(s, e)[:, None] >= keys[None, :]
        a = ((q0[:, s:e] @ k0.transpose(-1, -2)) * sc).masked_fill(~mask, float("-inf")).softmax(-1) @ v0
        b = ((q1[:, s:e] @ k1.transpose(-1, -2)) * sc).masked_fill(~mask, float("-inf")).softmax(-1) @ v1
        sims.append(torch.nn.functional.cosine_similarity(a.reshape(-1, D), b.reshape(-1, D), dim=-1))
    return torch.cat(sims).mean().item()


def make_qhook(idx):
    def f(_m, _i, out):
        T = out.shape[1]
        x = out.reshape(1, T, -1, D).permute(0, 2, 1, 3).contiguous().squeeze(0).float().cpu()
        CAP.setdefault(ARM, {}).setdefault(idx, {"k": None, "v": None, "khat": None, "vhat": None})
        CAP[ARM][idx]["q"] = x
        return out
    return f


def run_arm(m, layers, ids, W):
    handles = []
    for idx, L in enumerate(layers):
        at = getattr(L, "self_attn", None)
        if at is None:
            continue
        for part, nm in (("k", "k_proj"), ("v", "v_proj")):
            handles.append(getattr(at, nm).register_forward_hook(make_hook(part, idx)))
        if idx in CAP_LAYERS:
            handles.append(at.q_proj.register_forward_hook(make_qhook(idx)))
    emb = build_inputs_embeds(ids)
    pos = torch.arange(emb.shape[1], device=emb.device)
    pos3 = pos.view(1, 1, -1).expand(3, 1, -1).contiguous()
    t0 = time.time()
    with torch.no_grad():
        hs = m.model.language_model(inputs_embeds=emb, position_ids=pos3).last_hidden_state[0]
        nll, n = 0.0, 0
        for st in range(0, hs.shape[0] - 1, 256):
            e = min(st + 256, hs.shape[0] - 1)
            lg = hs[st:e].to(torch.float16) @ W.t()
            tgt = torch.tensor(ids[st + 1:e + 1], device=lg.device)
            nll += (torch.logsumexp(lg.float(), -1) - lg.gather(1, tgt[:, None]).squeeze(1).float()).sum().item()
            n += tgt.numel()
    for h in handles:
        h.remove()
    return nll / n, math.exp(nll / n), time.time() - t0


def main():
    global ARM, CAP_LAYERS
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/data/models/Qwen3.8-27B-3Bit-GSQ", trust_remote_code=True)
    DOCS = os.environ.get("PPL_DOC", "/data/kvarn/ppl_doc.txt")
    TOKS = os.environ.get("VSIDE_TOKS") or os.environ.get("PPL_TOK", "2944")
    CH = int(os.environ.get("PPL_CHARS", "20000"))
    RESFILE = os.environ.get("VSIDE_RESULTS", "/data/kvarn/vside-20261014/results.tsv")
    doc_list = [p.strip() for p in DOCS.split(";") if p.strip()]
    tok_list = [int(x) for x in TOKS.split(",")]

    if WATCH:
        threading.Thread(target=watchdog, daemon=True).start()
        print("[watchdog] 已启用（GPU0 被占用即退出）", flush=True)

    m, layers = load_model()
    W = lm_head_weight(m)
    self_idx = [i for i, L in enumerate(layers) if getattr(L, "self_attn", None) is not None]
    CAP_LAYERS.append(self_idx[-1])
    print("self_attn 层数=%d，attention cosine 取末层 idx=%d" % (len(self_idx), self_idx[-1]), flush=True)

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding as _RE
    tc2 = getattr(m.config, "text_config", m.config)
    rot = _RE(tc2).to(device="cuda:0", dtype=torch.float32)

    done = set()
    if os.path.exists(RESFILE):
        for ln in open(RESFILE, encoding="utf-8"):
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 3:
                done.add((p[0], p[1]))
    print("已完成臂（续跑用）：%s" % sorted(done), flush=True)

    res = {}
    for ci, doc in enumerate(doc_list):
        MAX_TOK = tok_list[ci] if ci < len(tok_list) else tok_list[-1]
        parts = [open(p.strip(), encoding="utf-8", errors="ignore").read() for p in doc.split(",") if p.strip()]
        text = re.sub(r"\s+", " ", "\n".join(parts))[:CH]
        ids = tok(text, add_special_tokens=False)["input_ids"][:MAX_TOK]
        ids = ids[:(len(ids) // 128) * 128]
        tag = os.path.basename(doc)
        print("\n########## 语料 %s → %d token ##########" % (tag, len(ids)), flush=True)

        p = torch.arange(len(ids), device="cuda:0").view(1, 1, -1).expand(3, 1, -1).contiguous()
        with torch.no_grad():
            o = rot(torch.zeros(1, len(ids), 1, device="cuda:0", dtype=torch.float16), p)
        RS["cos"], RS["sin"] = o[0][0].float().cpu(), o[1][0].float().cpu()
        print("离线 RoPE: cos %s 均值 %.4f" % (tuple(RS["cos"].shape), RS["cos"].mean().item()), flush=True)

        only = os.environ.get("VSIDE_ARMS_%d" % ci)
        if only is None:
            only = os.environ.get("VSIDE_ARMS_A" if ci == 0 else "VSIDE_ARMS_B")
        arms = only.split(",") if only else list(ARMS)
        out, ref, acos = {}, None, {}
        COS.clear(); CAP.clear(); _DBG.clear()
        for arm in arms:
            if (tag, arm) in done:
                print("%-18s 已在 results.tsv，跳过" % arm, flush=True)
                continue
            ARM = arm
            t0 = time.time()
            try:
                nll, ppl, ft = run_arm(m, layers, ids, W)
            except Exception as e:
                print("%-18s 前向失败: %s %s" % (arm, type(e).__name__, str(e)[:200]), flush=True)
                continue
            out[arm] = (nll, ppl)
            if arm == "baseline_fp16":
                ref = CAP.get(arm)
            ck = cv = float("nan")
            if arm in COS:
                ck = sum(COS[arm]["k"]) / len(COS[arm]["k"]) if COS[arm].get("k") else float("nan")
                cv = sum(COS[arm]["v"]) / len(COS[arm]["v"]) if COS[arm].get("v") else float("nan")
            ac = float("nan")
            if arm != "baseline_fp16" and ref and CAP.get(arm) and CAP_LAYERS[0] in ref:
                ac = attn_cos(ref[CAP_LAYERS[0]], CAP[arm][CAP_LAYERS[0]])
            line = ("%-18s mean NLL=%.6f PPL=%.4f 用时%.0fs | K cos=%.6f V cos=%.6f | attn cos=%.6f"
                    % (arm, nll, ppl, time.time() - t0, ck, cv, ac))
            print(line, flush=True)
            with open(RESFILE, "a", encoding="utf-8") as f:
                f.write("%s\t%s\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n" % (tag, arm, nll, ppl, ck, cv, ac))
            if ARM != "baseline_fp16":
                CAP.pop(arm, None)

        base = out["rk4v4_post"][1] if "rk4v4_post" in out else None
        if base:
            print("\n=== [%s] 判据：Δ = PPL(候选)/PPL(rk4v4) − 1 ≤ +0.10%% ===" % tag, flush=True)
            for a in arms:
                if a in out:
                    d = 100.0 * (out[a][1] / base - 1.0)
                    print("  %-18s PPL=%.4f  vs rk4v4 %+.4f%%  ⇒ %s"
                          % (a, out[a][1], d, "基准" if a == "rk4v4_post" else ("过" if d <= 0.1 else "不过")), flush=True)
        res[tag] = out
    print("VSIDE_DONE", flush=True)


if __name__ == "__main__":
    main()
