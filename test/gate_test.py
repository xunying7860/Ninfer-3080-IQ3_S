#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务 B 一票否决门：重启前后同一会话**逐 token 一致**。

设计（写在文件里免得以后再猜）：
  * 题面（needle 文本）不含 nonce；nonce 只出现在 system 标签里
    ⇒ 不同 nonce 的两次运行，**期望答案逐字相同**，可以直接做严格比对。
  * 每个 suite 定义「轮次 + 在第几轮之后重启」；比对时把两个 suite 的输出逐轮 diff。
  * 判活/判冷只看引擎自己的报数：TTFT / cache % / prompt tokens（从 ninfer-serve.log 的
    `req#N done` 行取），客户端耗时只作旁证。
  * 带媒体的请求（视觉）在 60f18c3 的磁盘恢复里**明确不支持**（disk_tier.cpp:350
    `prompt.has_media()` 直接 return nullopt）⇒ 视觉组只验「一致性」，速度另算。

用法：
  python3 gate_test.py run --suite ctrl      --out ev_ctrl.json
  python3 gate_test.py run --suite rst       --out ev_rst.json
  python3 gate_test.py compare ev_ctrl.json ev_rst.json
"""
import argparse, base64, json, os, re, struct, subprocess, sys, threading, time, zlib, urllib.request, urllib.error

PORT = int(os.environ.get("GATE_PORT", "18082"))
HOST = os.environ.get("GATE_HOST", "127.0.0.1")
LOG = os.environ.get("GATE_LOG", "/data/ninfer/logs/ninfer-serve.log")
RESTART_CMD = os.environ.get("GATE_RESTART_CMD", "sudo -n systemctl restart ninfer.service")
READY_TIMEOUT = int(os.environ.get("GATE_READY_TIMEOUT", "420"))
BASE_URL = "http://%s:%d" % (HOST, PORT)

# ── 题面 ────────────────────────────────────────────────────────────────────
MARK = "MARK-7731-KXQ"
EARLY = "EARLY-4417-ZZ"
LATE = "LATE-9082-QQ"
# 行数可由 GATE_DOC_LINES 放大做「线上真实深度」的测试（1200 行 ≈ 25.2k token，
# 8000 行 ≈ 168k token；实测 25,249 token / 1200 行）。needle 位置随行数自适应。
DOC_LINES = int(os.environ.get("GATE_DOC_LINES", "1200"))
MARK_LINE = DOC_LINES // 2
EARLY_LINE = 7
LATE_LINE = DOC_LINES - 7


def build_doc(n=None):
    n = n or DOC_LINES
    lines = []
    for i in range(n):
        if i == MARK_LINE:
            lines.append("Line %06d: %s at %d" % (i, MARK, MARK_LINE))
        elif i == EARLY_LINE:
            lines.append("Line %06d: %s early note" % (i, EARLY))
        elif i == LATE_LINE:
            lines.append("Line %06d: %s late note" % (i, LATE))
        else:
            lines.append("Line %06d: routine maintenance notes, inventory records and log entries follow." % i)
    return "\n".join(lines)


DOC = build_doc()

ROUNDS = [
    # (role-user text, max_tokens)
    (DOC + "\n\n问题：这份文档一共有多少行？只回答数字。", 16),
    ("文档第 %06d 行写的是什么？把那一行原样抄出来。" % MARK_LINE, 96),
    ("第 %06d 行和第 %06d 行各自提到什么？把两处合在一句话里回答。" % (EARLY_LINE, LATE_LINE), 128),
    ("用不超过 40 个字总结这份文档，必须包含第 %06d 行里的暗号。" % MARK_LINE, 128),
]

# 真正的工具调用轮（之前 tools 组只是「带着 tools 参数问文档题」，模型答文本，没有 <tool_call>，
# 不足以验「工具调用的会话重启后一致」这条验收项）
TOOL_ROUNDS = [
    ("北京明天的天气怎么样？必须调用工具查询，不要自己编答案。", 128),
    ("刚才工具返回的温度是多少摄氏度？只回答数字。", 32),
    ("那我要不要带伞？一句话回答。", 64),
]

SUITES = {
    # name: (restart 在第几轮【之后】，1-based；system 附加；标签)
    "ctrl":        {"restart_after": [],         "tag": "no-restart"},
    "rst":         {"restart_after": [1],        "tag": "restart-after-r1"},
    "rst2":        {"restart_after": [2, 3],     "tag": "restart-twice"},
    "tools":       {"restart_after": [1],        "tag": "tools-after-r1", "tools": True},
    "toolsctrl":   {"restart_after": [],         "tag": "tools-no-restart", "tools": True},
    "vis":         {"restart_after": [1],        "tag": "vision-after-r1", "vision": True},
    "visctrl":     {"restart_after": [],         "tag": "vision-no-restart", "vision": True},
    "apitool":     {"restart_after": [1],        "tag": "tool-call-after-r1", "tool_native": True, "tools_rounds": [1]},
    "apitoolctrl": {"restart_after": [],         "tag": "tool-call-no-restart", "tool_native": True, "tools_rounds": [1]},
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的当前天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名"}},
                       "required": ["city"]},
    },
}]

# 1x1 红色 PNG（stdlib 手搓，保证两次运行字节完全一致）
def tiny_png():
    raw = b""
    for y in range(64):
        raw += b"\x00" + (b"\xc8\x20\x20" * 64 if y < 40 else b"\x20\x40\xc8" * 64)
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


VIS_IMG = tiny_png()
VIS_ROUNDS = [
    ([{"type": "image_url", "image_url": {"url": VIS_IMG}},
      {"type": "text", "text": "记住这张图：上半部分和下半部分各是什么颜色？只回答『上:X 下:Y』。"}], 32),
    ("上半部分是什么颜色？只回答颜色。", 32),
    ("下半部分是什么颜色？只回答颜色。", 32),
]

REQ_RE = re.compile(r"req#(\d+) done")
# 引擎对 >1s 的值写成 "TTFT 18.4s"，>60s 写成 "TTFT 2m 53.0s"，<1s 写成 "TTFT 169 ms"；
# cache 也可能没有 ", kind" 后缀（冷启那发就是 `cache 0 (0.0%)`）⇒ 正则要能吃三种形态，
# 否则判据会误报 None（2026-10-12 踩过一次）。
TTFT_RE = re.compile(r"TTFT\s+((?:\d+m\s+)?[\d.,]+\s*(?:ms|s))")
CACHE_RE = re.compile(r"cache ([\d,]+) \(([\d.]+)%(?:, ([^)]+))?\)")
PROMPT_RE = re.compile(r"prompt ([\d,]+)")
OUT_RE = re.compile(r"output ([\d,]+)")
PRE_RE = re.compile(r"prefill ([\d.]+) tok/s")
DEC_RE = re.compile(r"decode ([\d.]+) tok/s")
MTP_RE = re.compile(r"mtp accepted ([\d/]+ \([\d.]+%\))")


def log_lines():
    try:
        with open(LOG, errors="replace") as fh:
            return fh.readlines()
    except Exception:
        return []


def last_req_done(after_index):
    """取 after_index 之后最新的一条 `req#N done` 行（判据来源：引擎自己的报数）。"""
    hit = None
    for ln in log_lines()[after_index:]:
        if REQ_RE.search(ln):
            hit = ln.strip()
    return hit


def parse_done(line):
    if not line:
        return {}
    def num(rx):
        m = rx.search(line)
        return m.group(1).replace(",", "") if m else None
    m = TTFT_RE.search(line)
    ttft_ms = None
    if m:
        # "2m 53.0s" / "18.4s" / "650 ms" 三种形态统一成毫秒。分钟只在「数字+m+空格」时才算，
        # 否则 "650 ms" 会被当成分钟 ⇒ re.match 返回 None 直接崩（2026-10-12 踩过）。
        txt = m.group(1).strip()
        minutes = 0
        if re.match(r"^\d+m\s", txt):
            minutes = int(re.match(r"(\d+)m", txt).group(1))
            txt = re.sub(r"^\d+m\s+", "", txt)
        vm = re.match(r"([\d.,]+)\s*(ms|s)", txt)
        if vm:
            v = float(vm.group(1).replace(",", ""))
            ttft_ms = int(round(v * 1000)) if vm.group(2) == "s" else int(round(v))
        ttft_ms = (minutes * 60000 + ttft_ms) if ttft_ms is not None else None
    cm = CACHE_RE.search(line)
    return {"ttft_ms": ttft_ms, "cache_tokens": num(CACHE_RE),
            "cache_pct": (cm.group(2) if cm else None),
            "cache_kind": (cm.group(3).strip() if cm and cm.group(3) else None),
            "prompt_tokens": num(PROMPT_RE), "output_tokens": num(OUT_RE),
            "prefill_tps": num(PRE_RE), "decode_tps": num(DEC_RE),
            "mtp": (MTP_RE.search(line).group(1) if MTP_RE.search(line) else None),
            "line": line}


def slot_action(slot, action, payload, retries=40):
    """POST /slots/{id}?action=save|restore|erase —— 上游 slot 持久化只有这条通道。
    409 slot_busy 是常态（会话/事务占用），按退避重试。
    ★ 超时要给足：168k 深度一次 save 写 3.13 GiB 实测 22.7 s，第二次 save 超过 180 s
      （2026-10-12 实测客户端 180s 读超时），故 900 s。"""
    for _ in range(retries):
        try:
            return post("/slots/%d?action=%s" % (slot, action), payload, timeout=900)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:240]
            if e.code in (409, 503):
                time.sleep(2)
                continue
            return {"http_error": e.code, "body": body}
    return {"http_error": "busy-timeout"}


def post(path, payload, timeout=2400):
    req = urllib.request.Request(BASE_URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_ready(timeout=READY_TIMEOUT):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(BASE_URL + "/v1/models", timeout=4) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("engine not ready within %ds" % timeout)


def model_id():
    for _ in range(5):
        try:
            with urllib.request.urlopen(BASE_URL + "/v1/models", timeout=5) as x:
                return json.load(x)["data"][0]["id"]
        except Exception:
            time.sleep(2)
    raise RuntimeError("cannot read model id")


def ask(messages, max_tokens, tools=None, temperature=0.0):
    """发一条 greedy 请求，返回 (content, tool_calls, finish, usage, 引擎日志判据行, 客户端耗时)。"""
    body = {"model": model_id(), "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    before = len(log_lines())
    t0 = time.time()
    d = post("/v1/chat/completions", body)
    el = time.time() - t0
    choice = (d.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    time.sleep(0.5)
    done = parse_done(last_req_done(before))
    return {"content": msg.get("content") or "",
            "id_slot": d.get("id_slot"), "session_digest": d.get("session_digest"),
            "reasoning": msg.get("reasoning_content") or "",
            "tool_calls": json.dumps(msg.get("tool_calls"), ensure_ascii=False, sort_keys=True)
                            if msg.get("tool_calls") else None,
            "finish": choice.get("finish_reason"),
            "usage": d.get("usage"), "client_s": round(el, 2), "engine": done}


def restart(on_ready=None):
    t0 = time.time()
    print("  >>> restart: %s" % RESTART_CMD, flush=True)
    subprocess.run(RESTART_CMD, shell=True, check=True)
    time.sleep(1.5)
    took = wait_ready()
    print("  >>> ready in %.0f s (restart+boot)" % (time.time() - t0), flush=True)
    extra = {}
    if on_ready is not None:
        extra = on_ready() or {}
    return {"restart_wall_s": round(time.time() - t0, 1), "ready_after_restart_s": round(took, 1), **extra}


def run_suite(name, out_path):
    cfg = SUITES[name]
    # ★ 定 nonce：不同 nonce 会让 system 提示词的 token 数差 1（实测：ctrl 25,302 vs rst 25,301），
    #   于是"同一会话"其实并不同 token 流，开放题（R4 总结）就会合法地分叉 —— 门禁必须用**完全相同**
    #   的 token 流，所以 nonce 由 GATE_NONCE 固定（每个候选件一个），冷启唯一性改由
    #   「先清空磁盘层 + 新进程」保证。
    nonce = os.environ.get("GATE_NONCE") or "gate-%d-%s" % (int(time.time()), "%04x" % (os.getpid() & 0xffff))
    sysmsg = "You are a careful assistant. Answer briefly and exactly. Session tag: " + nonce
    record = {"suite": name, "tag": cfg.get("tag"), "nonce": nonce, "port": PORT,
              "model": None, "rounds": [], "restarts": []}
    # 上游 slot 持久化（机制 B）只能走 REST：save 每轮调一次，restore 在重启就绪后调一次。
    # GATE_SLOT_SAVE=1 打开 save；GATE_SLOT_RESTORE=1 打开重启后的 restore。文件名规则见 docs/serving.md。
    slot_state = {"slot": None, "digest": None, "file": None}
    if os.environ.get("GATE_SLOT_SAVE"):
        slot_state["file"] = re.sub(r"[^A-Za-z0-9._-]", "-", "%s-%s-%d" % (name, nonce, DOC_LINES)).strip(".-")[:120]
    wait_ready()
    record["model"] = model_id()
    print("### suite=%s (%s) model=%s port=%d" % (name, cfg.get("tag"), record["model"], PORT), flush=True)

    if cfg.get("vision"):
        rounds = VIS_ROUNDS
    elif cfg.get("tool_native"):
        rounds = TOOL_ROUNDS
    else:
        rounds = [(t, m) for (t, m) in ROUNDS]

    messages = [{"role": "system", "content": sysmsg}]
    for idx, (text, mx) in enumerate(rounds, start=1):
        messages.append({"role": "user", "content": text})
        tools_rounds = cfg.get("tools_rounds", [1] if cfg.get("tools") else [])
        tools = TOOLS if idx in tools_rounds else None
        r = ask(messages, mx, tools=tools)
        record["rounds"].append({"round": idx, **r})
        print("  R%d client=%ss ttft=%s ms cache=%s (%s%%, %s) prompt=%s out=%s finish=%s%s"
              % (idx, r["client_s"], r["engine"].get("ttft_ms"), r["engine"].get("cache_tokens"),
                 r["engine"].get("cache_pct"), r["engine"].get("cache_kind"),
                 r["engine"].get("prompt_tokens"), r["engine"].get("output_tokens"), r["finish"],
                 (" tools=" + r["tool_calls"][:60]) if r["tool_calls"] else ""), flush=True)
        # 把引擎侧的 assistant 结果并回会话历史（工具轮要补 tool_calls + tool 结果）
        asst = {"role": "assistant", "content": r["content"]}
        if r["tool_calls"]:
            asst["tool_calls"] = json.loads(r["tool_calls"])
            asst["content"] = r["content"] or None
            messages.append(asst)
            messages.append({"role": "tool", "tool_call_id": json.loads(r["tool_calls"])[0]["id"],
                             "content": "晴，23 摄氏度，微风"})
            record["rounds"][-1]["tool_result_appended"] = True
        else:
            messages.append(asst)
        if slot_state["file"] is not None and r.get("id_slot") is not None:
            slot_state["slot"] = r["id_slot"]
            slot_state["digest"] = r.get("session_digest")
            payload = {"filename": slot_state["file"]}
            if slot_state["digest"]:
                payload["if_digest"] = slot_state["digest"]
            sv = slot_action(r["id_slot"], "save", payload)
            record["rounds"][-1]["slot_save"] = sv
            print("  R%d slot_save slot=%s -> %s" % (idx, r["id_slot"], json.dumps(sv, ensure_ascii=False)[:180]), flush=True)
        if idx in cfg["restart_after"]:
            def _restore():
                if slot_state["file"] is None or slot_state["slot"] is None or not os.environ.get("GATE_SLOT_RESTORE"):
                    return None
                rs = slot_action(slot_state["slot"], "restore", {"filename": slot_state["file"]})
                print("  >>> slot_restore file=%s -> %s" % (slot_state["file"], json.dumps(rs, ensure_ascii=False)[:220]), flush=True)
                return {"slot_restore": rs}
            record["restarts"].append({"after_round": idx, **restart(on_ready=_restore)})
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=1)
    print("### evidence written: %s" % out_path, flush=True)
    return record


def norm_tools(s):
    """工具调用里的 `id`（call_<随机>）是每次请求新生成的，不参与会话语义
    （2026-10-12 实测：ctrl 与 rst 的 tool_calls 只有 id 不同，function/arguments 逐字相同）。
    比对时把 id 去掉，否则「工具调用会话重启后一致」这条永远判 FAIL。"""
    if not s:
        return s
    try:
        obj = json.loads(s)
    except Exception:
        return s
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in ("id", "index")}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    return json.dumps(strip(obj), ensure_ascii=False, sort_keys=True)


def cmp_files(a_path, b_path):
    a, b = (json.load(open(p, encoding="utf-8")) for p in (a_path, b_path))
    print("A=%s (%s, nonce=%s)" % (a_path, a["suite"], a["nonce"]))
    print("B=%s (%s, nonce=%s)" % (b_path, b["suite"], b["nonce"]))
    ok_all = True
    for i in range(max(len(a["rounds"]), len(b["rounds"]))):
        ra = a["rounds"][i] if i < len(a["rounds"]) else None
        rb = b["rounds"][i] if i < len(b["rounds"]) else None
        if ra is None or rb is None:
            print("  R%d 轮数不一致 ⇒ FAIL" % (i + 1)); ok_all = False; continue
        same_c = ra["content"] == rb["content"]
        same_t = norm_tools(ra["tool_calls"]) == norm_tools(rb["tool_calls"])
        same_r = ra["reasoning"] == rb["reasoning"]
        ok = same_c and same_t and same_r and bool(rb["content"] or rb["tool_calls"])
        ok_all &= ok
        print("  R%d %s  content_equal=%s tools_equal(去 id)=%s reasoning_equal=%s  A_out=%s B_out=%s"
              % (i + 1, "OK" if ok else "MISMATCH", same_c, same_t, same_r,
                 (ra["content"][:40] or ("<tool_call>" if ra["tool_calls"] else "")),
                 (rb["content"][:40] or ("<tool_call>" if rb["tool_calls"] else ""))))
        if not same_c:
            print("     A: %r" % ra["content"][:200]); print("     B: %r" % rb["content"][:200])
        if not same_t:
            print("     A tools: %s" % (ra["tool_calls"] or "")[:200]); print("     B tools: %s" % (rb["tool_calls"] or "")[:200])
    print("### VERDICT: %s" % ("PASS(逐 token 一致)" if ok_all else "FAIL(逐 token 不一致)"))
    return 0 if ok_all else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--suite", required=True, choices=sorted(SUITES)); r.add_argument("--out", required=True)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    s = sub.add_parser("ping")
    a = ap.parse_args()
    if a.cmd == "run":
        run_suite(a.suite, a.out); return 0
    if a.cmd == "compare":
        return cmp_files(a.a, a.b)
    print("ready in %.1f s; model=%s" % (wait_ready(), model_id())); return 0


if __name__ == "__main__":
    sys.exit(main())
