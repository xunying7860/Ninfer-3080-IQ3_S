#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""slot-sidecar.py —— 上游 slot 持久化（机制 B）缺的那个「调用方」。

为什么要它（2026-10-12 实测 + master docs/serving.md 原文）：引擎自己**不会**在请求结束时存档，
也不会重启前存档；只有 `POST /slots/{id}?action=save|restore|erase` 这条路。`--auto-save-evicted`
只负责把「此前存过的」会话在被驱逐前回写。⇒ 不写这个 sidecar（或 provider 插件），机制 B 永远不生效。

做什么：
  1) 稳态循环：GET /slots → 对每个 retained 且不在处理中的 slot，若 session_digest 与上次存的
     不同，就 POST action=save（文件名 slot-<id>，用 if_digest 防「存到一半被人改了」）。
  2) 重启侦测：GET /slots 失败或 slot 列表空/该 slot 没有 retained 会话 ⇒ 认为引擎是新起来的 ⇒
     从状态文件里按 slot id 找最近绑定的文件名，POST action=restore 把它搬回来。
  3) 全部动作写日志与状态文件；409 slot_busy / slot_session_mismatch 退避重试，501（这条线没开
     slot 持久化）就当不支持、不刷屏。

只在引擎口上动 REST，不动生产配置、不动二进制。
用法：
  python3 slot-sidecar.py --base http://127.0.0.1:18082 --interval 20 \
      --state /data/ninfer/test/slot-sidecar-state.json --log /data/ninfer/test/slot-sidecar.log [--once]
"""
import argparse, json, os, re, sys, time, urllib.error, urllib.request

BUSY_CODES = (409, 503)
FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def log(path, msg):
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def req_json(url, payload=None, timeout=900, method=None):
    """返回 (status, parsed_or_text)。payload=None ⇒ GET。"""
    data = None if payload is None else json.dumps(payload).encode()
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                               method=method or ("POST" if data else "GET"))
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            body = resp.read().decode(errors="replace")
            try:
                return resp.status, json.loads(body)
            except Exception:
                return resp.status, body[:300]
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        return e.code, body
    except Exception as e:
        return 0, "%s: %s" % (type(e).__name__, e)


def load_state(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_state(path, st):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def slot_filename(slot_id):
    name = "slot-%d" % slot_id
    assert FILENAME_RE.match(name)
    return name


def do_recover(args, st, slots=None):
    """把 state 里绑定的快照搬回引擎里那些「空着的」 slot（= 引擎刚重启过的形状）。
    引擎自己不会 restore，这条是所有恢复路径的唯一入口（守护循环与 --recover 都走这里）。
    slots=None 时自己 GET 一次；已有 retained 会话的 slot 不动（别把新会话覆盖掉）。"""
    if slots is None:
        code, slots = req_json(args.base + "/slots", timeout=15)
        if code != 200 or not isinstance(slots, list):
            log(args.log, "recover: GET /slots -> %s %s" % (code, str(slots)[:160]))
            return 0
    # 有请求在途就不要动：2026-10-12 实测守护在引擎刚起来时硬恢复，与在途请求抢 state 容量，
    # restore_ms 从 1.98 s 涨到 19.82 s，反而把下一条请求让成了全量重算。
    if any(s.get("is_processing") for s in slots):
        log(args.log, "recover 推迟：有 slot 正在处理请求（不与在途请求抢恢复）")
        return 0
    plans = []
    for slot in slots:
        sid = slot.get("id")
        if sid is None:
            continue
        key = str(sid)
        rec = st["slots"].setdefault(key, {})
        bound = rec.get("filename")
        if not bound:
            continue
        if slot.get("retained"):
            log(args.log, "recover 跳过 slot=%s：引擎里已经有 retained 会话（不覆盖）" % sid)
            continue
        plans.append((sid, key, bound, rec))
    # 最近存过的先恢复：引擎的 device state slots 只有一个大位，装不下多条大会话
    # （实测第二条恢复直接 400 slot_restore_failed「does not fit the free state capacity」）
    plans.sort(key=lambda t: (t[3].get("saved_at") or ""), reverse=True)
    n = 0
    for sid, key, bound, rec in plans:
        for attempt in range(args.max_retry):
            c2, b2 = req_json("%s/slots/%d?action=restore" % (args.base, sid), {"filename": bound})
            if c2 == 200:
                rec["last_restored_from"] = bound
                rec["restored_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                d2 = None
                if isinstance(b2, dict):
                    d2 = b2.get("session_digest") or b2.get("digest")
                if d2:
                    # 恢复出来的会话 digest 与存的时候一致才算搬对；顺手记下来避免立刻又存一遍
                    rec["last_seen_digest"] = d2
                    rec["restored_digest_matches_saved"] = (d2 == rec.get("last_saved_digest"))
                log(args.log, "restore slot=%s file=%s -> %s" % (sid, bound, str(b2)[:220]))
                save_state(args.state, st)
                n += 1
                break
            if c2 in BUSY_CODES:
                time.sleep(2); continue
            if c2 == 400 and "does not fit the free state capacity" in str(b2):
                log(args.log, "restore slot=%s 装不下（已知容量约束：device state slots 只有一个大位）⇒ 跳过：%s"
                    % (sid, str(b2)[:150]))
                break
            log(args.log, "restore slot=%s 失败 %s %s" % (sid, c2, str(b2)[:160]))
            break
    st["engine_up"] = True
    save_state(args.state, st)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18082")
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--state", default="/data/ninfer/test/slot-sidecar-state.json")
    ap.add_argument("--log", default="/data/ninfer/test/slot-sidecar.log")
    ap.add_argument("--once", action="store_true", help="只跑一轮（调试/验收用）")
    ap.add_argument("--max-retry", type=int, default=6)
    ap.add_argument("--recover", action="store_true",
                    help="不做稳态轮询，只把 state 里绑定的快照 restore 进空 slot（引擎重启后的恢复动作）")
    args = ap.parse_args()

    st = load_state(args.state)
    st.setdefault("slots", {})

    if args.recover:
        n = do_recover(args, st)
        log(args.log, "--recover 完成，恢复 %d 个 slot" % n)
        return 0
    engine_up = st.get("engine_up", True)   # 状态里记着上次是否在跑 ⇒ 重启能被下一个人看见并恢复
    disabled_seen = False
    bad_n = 0                               # 连续 GET /slots 失败的次数（用来抑制刷屏）

    while True:
        code, slots = req_json(args.base + "/slots", timeout=15)
        if code == 0:
            if engine_up:
                log(args.log, "引擎不可达（%s）⇒ 下一轮起来时按状态文件 restore" % slots)
            engine_up = False
            if args.once:
                return 0
            time.sleep(args.interval)
            continue

        if code == 501 or (isinstance(slots, str) and "slot_persistence_disabled" in str(slots)):
            if not disabled_seen:
                log(args.log, "这条线的 slot 持久化没开（501 slot_persistence_disabled）⇒ 需要 "
                              "--slot-save-path；sidecar 不动作")
                disabled_seen = True
            if args.once:
                return 0
            time.sleep(args.interval)
            continue

        if code != 200 or not isinstance(slots, list):
            # 引擎正在起（503 model_loading）或端口还没听时会连撞几十次：只报第一次和每 10 次一条，
            # 否则日志被启动过程刷满、真正的事件反而看不见（2026-10-12 mastF 实测一次启动刷了 25 行）
            bad_n += 1
            if bad_n == 1 or bad_n % 10 == 0:
                log(args.log, "GET /slots -> %s %s（连续第 %d 次）" % (code, str(slots)[:140], bad_n))
            if args.once:
                return 1
            time.sleep(args.interval)
            continue
        bad_n = 0

        if not engine_up:
            log(args.log, "引擎刚起来（/slots 恢复响应）⇒ 按 state 绑定 restore")
            do_recover(args, st, slots)
            engine_up = True

        for slot in slots:
            sid = slot.get("id")
            if sid is None:
                continue
            key = str(sid)
            rec = st["slots"].setdefault(key, {})
            if slot.get("is_processing"):
                continue                       # 有请求在途，别抢
            if not slot.get("retained"):
                continue
            digest = slot.get("session_digest")
            fname = slot_filename(sid)
            if digest and rec.get("last_seen_digest") == digest and rec.get("filename") == fname:
                continue                       # 没有新东西可存
            payload = {"filename": fname}
            if digest:
                payload["if_digest"] = digest
            ok = False
            for attempt in range(args.max_retry):
                c2, b2 = req_json("%s/slots/%d?action=save" % (args.base, sid), payload)
                if c2 == 200 and isinstance(b2, dict):
                    rec.update({"filename": fname, "last_saved_digest": digest,
                                "last_seen_digest": digest, "n_saved": b2.get("n_saved"),
                                "n_written": b2.get("n_written"),
                                "save_ms": (b2.get("timings") or {}).get("save_ms"),
                                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
                    save_state(args.state, st)
                    log(args.log, "save slot=%s file=%s tokens=%s bytes=%s ms=%s"
                        % (sid, fname, b2.get("n_saved"), b2.get("n_written"),
                           (b2.get("timings") or {}).get("save_ms")))
                    ok = True
                    break
                if c2 in BUSY_CODES:
                    time.sleep(2); continue
                log(args.log, "save slot=%s 失败 %s %s" % (sid, c2, str(b2)[:160]))
                break
            if not ok:
                # 存失败也别把 digest 记成「已存」，下轮重试；但记一下最后看到的 digest 以免日志刷屏
                rec["last_seen_digest"] = digest
                save_state(args.state, st)
        engine_up = True
        st["engine_up"] = engine_up
        save_state(args.state, st)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
