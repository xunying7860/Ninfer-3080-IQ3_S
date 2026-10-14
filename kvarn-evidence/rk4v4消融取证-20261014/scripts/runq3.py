#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""runq3.py <port> <engine_log> <out.json> <arm_tag> <ids_csv> [runs_override]

只跑指定 id。id 取值 = prompts.json 里的 F1/F2/C1/C2/Z1/Z2/L1 + 追加的 Q1/Q2/R3 + 特殊项 G4。
G4 = 逐字复刻 gate_test.py 的 ctrl 套件四轮会话（system nonce=gate-prod、1200 行文档、
     reasoning_effort=none + enable_thinking=False、逐轮把 assistant 结果并回历史），
     用来在不跑整套门禁的前提下快速反复量 R4（第 4 轮）那句的稳定性。

环境变量 RUNQ_EFFORT=<none|minimal|low|medium|high|xhigh|max>
  ⇒ 在 payload 里显式带 "reasoning_effort": <值>（不再发 chat_template_kwargs.enable_thinking），
    用于把「思考档」对齐到生产件的 xhigh。G4 项永远不发（保持门禁原样）。

统一口径：一致性看 combined = reasoning + content，取前 200 字逐字比较。
"""
import json, os, re, sys, time, urllib.request

PORT = int(sys.argv[1]); LOG = sys.argv[2]; OUT = sys.argv[3]; ARM = sys.argv[4]
IDS = [x for x in sys.argv[5].split(",") if x]
RUNS = int(sys.argv[6]) if len(sys.argv) > 6 else None
EFFORT = os.environ.get("RUNQ_EFFORT") or None
BASE = "http://127.0.0.1:%d" % PORT
REQ_RE = re.compile(r"req#(\d+) done")
START_RE = re.compile(r"req#(\d+) started")
TTFT_RE = re.compile(r"TTFT\s+((?:\d+m\s+)?[\d.,]+\s*(?:ms|s))")
CACHE_RE = re.compile(r"cache ([\d,]+) \(([\d.]+)%(?:, ([^)]+))?\)")
PROMPT_RE = re.compile(r"prompt ([\d,]+)")
OUT_RE = re.compile(r"output ([\d,]+)")
PRE_RE = re.compile(r"prefill ([\d.]+) tok/s")
DEC_RE = re.compile(r"decode ([\d.]+) tok/s")
MTP_RE = re.compile(r"mtp accepted ([^|]+)")
MIX_RE = re.compile(r"mixed speculation accepted ([^|]+)")


def log_lines():
    try:
        with open(LOG, errors="replace") as fh:
            return fh.readlines()
    except Exception:
        return []


def last_matching(after, rx):
    hit = None
    for ln in log_lines()[after:]:
        if rx.search(ln):
            hit = ln.strip()
    return hit


def parse_done(line):
    if not line:
        return {}
    def num(rx):
        m = rx.search(line)
        return m.group(1).replace(",", "") if m else None
    m = TTFT_RE.search(line); ttft_ms = None
    if m:
        txt = m.group(1).strip(); minutes = 0
        if re.match(r"^\d+m\s", txt):
            minutes = int(re.match(r"(\d+)m", txt).group(1)); txt = re.sub(r"^\d+m\s+", "", txt)
        vm = re.match(r"([\d.,]+)\s*(ms|s)", txt)
        if vm:
            v = float(vm.group(1).replace(",", ""))
            ttft_ms = int(round(v * 1000)) if vm.group(2) == "s" else int(round(v))
        if ttft_ms is not None:
            ttft_ms = minutes * 60000 + ttft_ms
    cm = CACHE_RE.search(line)
    return {"ttft_ms": ttft_ms, "cache_tokens": num(CACHE_RE), "cache_pct": (cm.group(2) if cm else None),
            "cache_kind": (cm.group(3).strip() if cm and cm.group(3) else None),
            "prompt_tokens": num(PROMPT_RE), "output_tokens": num(OUT_RE),
            "prefill_tps": num(PRE_RE), "decode_tps": num(DEC_RE),
            "mtp": (MTP_RE.search(line).group(1).strip() if MTP_RE.search(line) else None),
            "mixed": (MIX_RE.search(line).group(1).strip() if MIX_RE.search(line) else None),
            "raw": line}


def post(path, payload, timeout=2400):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def model_id():
    for _ in range(10):
        try:
            with urllib.request.urlopen(BASE + "/v1/models", timeout=6) as x:
                return json.load(x)["data"][0]["id"]
        except Exception:
            time.sleep(2)
    raise RuntimeError("cannot read model id")


def build_doc(n):
    lines = []
    for i in range(n):
        if i == 1000:
            lines.append("Line %06d: MARK-7731-KXQ at 1000" % i)
        elif i == 1500:
            lines.append("Line %06d: LATE-9082-QQ at 1500" % i)
        else:
            lines.append("Line %06d: routine maintenance notes, inventory records and log entries follow." % i)
    return "\n".join(lines)


def build_gate_doc(n):
    mk = n // 2; early = 7; late = n - 7
    lines = []
    for i in range(n):
        if i == mk:
            lines.append("Line %06d: MARK-7731-KXQ at %d" % (i, mk))
        elif i == early:
            lines.append("Line %06d: EARLY-4417-ZZ early note" % i)
        elif i == late:
            lines.append("Line %06d: LATE-9082-QQ late note" % i)
        else:
            lines.append("Line %06d: routine maintenance notes, inventory records and log entries follow." % i)
    return "\n".join(lines), mk


def ask(mid, messages, max_tokens, effort=None, thinking_off=False):
    body = {"model": mid, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if thinking_off:
        body["reasoning_effort"] = "none"
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif effort:
        body["reasoning_effort"] = effort
    before = len(log_lines()); t0 = time.time()
    d = post("/v1/chat/completions", body)
    el = time.time() - t0; time.sleep(0.4)
    ch = (d.get("choices") or [{}])[0]; msg = ch.get("message") or {}
    content = msg.get("content") or ""; reasoning = msg.get("reasoning_content") or ""
    return {"content": content, "reasoning": reasoning, "combined": (reasoning + content),
            "finish": ch.get("finish_reason"), "usage": d.get("usage"), "client_s": round(el, 2),
            "engine_started": last_matching(before, START_RE),
            "engine_raw": last_matching(before, REQ_RE),
            "engine": parse_done(last_matching(before, REQ_RE))}


LONG = [
    {"id": "Q1", "kind": "长答-技术解释", "runs": 2, "max_tokens": 1024,
     "text": "请用中文详细解释 Transformer 自注意力（self-attention）的计算过程：Q、K、V 各自的作用，缩放点积注意力的公式，以及为什么要除以 sqrt(d_k)。要求条理清晰、分点说明，不少于 600 字。"},
    {"id": "Q2", "kind": "长答-技术解释", "runs": 2, "max_tokens": 1024,
     "text": "请用中文详细解释操作系统里虚拟内存的作用与实现机制，覆盖页表、TLB、缺页中断与常见页面置换算法，并说明它为什么能让程序使用超过物理内存的地址空间。要求条理清晰、分点说明，不少于 600 字。"},
    {"id": "R3", "kind": "重复性(连跑3次)", "runs": 3, "max_tokens": 160,
     "text": "用 Python 写一个函数 add(a, b) 返回两数之和。只输出代码，不要解释。"},
]


def main():
    mid = model_id()
    prompts = {p["id"]: p for p in json.load(open("/data/kvarn/g128-rk4v4-test/prompts.json", encoding="utf-8")) + LONG}
    rec = {"arm": ARM, "port": PORT, "model": mid, "log": LOG, "effort_env": EFFORT, "ids": IDS, "results": []}
    print("### arm=%s port=%d model=%s effort_env=%s ids=%s" % (ARM, PORT, mid, EFFORT, IDS), flush=True)
    for i in IDS:
        if i == "G4":
            n = int(os.environ.get("GATE_DOC_LINES", "1200"))
            doc, mk = build_gate_doc(n)
            rounds = [(doc + "\n\n问题：这份文档一共有多少行？只回答数字。", 16),
                      ("文档第 %06d 行写的是什么？把那一行原样抄出来。" % mk, 96),
                      ("第 %06d 行和第 %06d 行各自提到什么？把两处合在一句话里回答。" % (7, n - 7), 128),
                      ("用不超过 40 个字总结这份文档，必须包含第 %06d 行里的暗号。" % mk, 128)]
            nruns = RUNS or 3
            for run in range(1, nruns + 1):
                msgs = [{"role": "system", "content":
                         "You are a careful assistant. Answer briefly and exactly. Session tag: gate-prod"}]
                for idx, (text, mx) in enumerate(rounds, start=1):
                    msgs.append({"role": "user", "content": text})
                    r = ask(mid, msgs, mx, thinking_off=True)
                    rec["results"].append({"id": "G4", "kind": "gate-ctrl-R%d" % idx, "run": run, "round": idx,
                                           "max_tokens": mx, "prompt_chars": len(text), **r})
                    print("  G4 run%d R%d finish=%s out=%s cache=%s(%s%%) mtp=%s mixed=%s | %r" %
                          (run, idx, r["finish"], r["engine"].get("output_tokens"), r["engine"].get("cache_tokens"),
                           r["engine"].get("cache_pct"), r["engine"].get("mtp"), r["engine"].get("mixed"),
                           r["content"][:120]), flush=True)
                    msgs.append({"role": "assistant", "content": r["content"]})
                with open(OUT, "w", encoding="utf-8") as fh:
                    json.dump(rec, fh, ensure_ascii=False, indent=1)
            continue
        p = prompts.get(i)
        if p is None:
            print("  !! 未知 id %s" % i); continue
        text = p["text"]
        if p.get("doc_lines"):
            text = build_doc(p["doc_lines"]) + "\n\n" + p["text"]
        nruns = RUNS or p.get("runs", 2)
        for run in range(1, nruns + 1):
            r = ask(mid, [{"role": "user", "content": text}], p["max_tokens"], effort=EFFORT)
            e = r["engine"]
            rec["results"].append({"id": i, "kind": p["kind"], "run": run, "max_tokens": p["max_tokens"],
                                   "prompt_chars": len(text), **r})
            print("  %s run%d client=%.1fs ttft=%s cache=%s(%s%%,%s) prompt=%s out=%s finish=%s combined=%dch started=%s" %
                  (i, run, r["client_s"], e.get("ttft_ms"), e.get("cache_tokens"), e.get("cache_pct"),
                   e.get("cache_kind"), e.get("prompt_tokens"), e.get("output_tokens"), r["finish"],
                   len(r["combined"]), (r["engine_started"] or "")[-40:]), flush=True)
            with open(OUT, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=1)
    print("### written %s (%d results)" % (OUT, len(rec["results"])), flush=True)


if __name__ == "__main__":
    main()
