#!/usr/bin/env python3
# ★ strata「重启不丢上下文」补丁（纯包装层 serve/server.py，不重编引擎）
# 依据（源码实测）：
#   · 引擎支持 stdin 命令 `SAVE <path>` → SAVED <tokens> <bytes> <ms> ；`RESTORE <path>` → RESTORED ...
#   · 包装层早有完整实现 StrataEngine.session_file("save"/"restore", path)（含进度/超时/FIFO）
#   · 但没有任何地方在【停机】时调用它 —— 这就是"优雅重启丢常驻会话"的根因（与 ninfer 磁盘层同一边界）
# 做法：① close() 里发 QUIT 之前 SAVE ② 启动引擎后若有会话文件则 RESTORE ③ 配置 session_autosave 控制
# 幂等、改前备份、失败绝不阻断停机。
import os
import shutil
import sys
import time

SP = "/data/workspace/strata01403/serve/server.py"
CFG = "/data/workspace/strata01403/strata-iq3_xxs.1card-nvme.json"
STAMP = time.strftime("%Y%m%d-%H%M%S")

s = open(SP, encoding="utf-8").read()
if "_autosave_on_close" in s:
    print("补丁已在，跳过改写")
else:
    orig = s

    # ---------- ① 助手函数：会话文件路径 ----------
    anchor1 = "def slot_save_dir("
    i1 = s.index(anchor1)
    helper = '''def session_autosave_path(cfg: dict) -> str | None:
    """★ 停机落盘 / 启动恢复所用的会话文件路径。配置 "session_autosave"：文件名（相对 cwd）或 false/"" 关闭。
    缺省开启（cwd 下 strata-session-autosave.bin）——用户要求"重启不丢上下文"且配置固化、无需每次重配。"""
    v = cfg.get("session_autosave", "strata-session-autosave.bin")
    if v is False or v is None or v == "":
        return None
    if not isinstance(v, str):
        raise ValueError("session_autosave must be a file name or false")
    if any(c in v for c in "\\r\\n\\0"):
        raise ValueError("session_autosave must not contain control characters")
    return v if os.path.isabs(v) else os.path.abspath(os.path.join(cfg.get("cwd") or ".", v))


'''
    s = s[:i1] + helper + s[i1:]

    # ---------- ② StrataEngine 里加方法 + 类属性 ----------
    anchor2 = "    def close(self):\n        \"\"\"End the engine process: QUIT first"
    i2 = s.index(anchor2)
    method = '''    autosave_path: str | None = None     # ★ 停机前把常驻会话 SAVE 到这个文件（None = 关闭）

    def _autosave_on_close(self) -> None:
        """★ 停机前把当前常驻会话落盘，供下次启动 RESTORE —— 这是"优雅重启不丢上下文"的关键一步。
        任何失败都只打印一行、绝不阻断停机（宁可丢缓存，不可停不下来）。"""
        path = self.autosave_path
        if not path or self.proc is None:
            return
        try:
            if self.proc.poll() is not None or not self.alive():
                return
            r = self.session_file("save", path)
            print(f"[strata] session autosaved before shutdown: {r['tokens']} tokens, "
                  f"{r['bytes'] >> 20} MiB, {r['ms'] / 1000:.1f} s -> {path}", flush=True)
        except Exception as e:                      # SessionRefused / EngineDied / 任何意外
            print(f"[strata] session autosave failed ({type(e).__name__}: {e}); stopping anyway", flush=True)

'''
    s = s[:i2] + method + s[i2:]

    # close() 顶部调用（在 self.proc is None 之外、try 之前）
    anchor3 = ("    def close(self):\n        \"\"\"End the engine process: QUIT first")
    j = s.index(anchor3)
    k = s.index("        if self.proc is None:\n            return\n", j) + len("        if self.proc is None:\n            return\n")
    s = s[:k] + "        self._autosave_on_close()          # ★ 先落盘再做 QUIT\n" + s[k:]

    # ---------- ③ 启动后 RESTORE ----------
    anchor4 = "        engine.silence_s = silence                      # an attribute of its own: restart() keeps it\n"
    i4 = s.index(anchor4) + len(anchor4)
    restore = '''        # ★ 启动时恢复上次停机落盘的会话（重启不丢上下文）；失败只打印一行、按空会话启动
        try:
            autosave = session_autosave_path(cfg)
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        engine.autosave_path = autosave
        if autosave and os.path.exists(autosave):
            try:
                r = engine.session_file("restore", autosave)
                print(f"[strata] session restored at startup: {r['tokens']} tokens, {r['bytes'] >> 20} MiB, "
                      f"{r['ms'] / 1000:.1f} s from {autosave}", flush=True)
            except Exception as e:
                print(f"[strata] session restore failed ({type(e).__name__}: {e}); starting with an empty session",
                      flush=True)
        elif autosave:
            print(f"[strata] session autosave on: {autosave} (nothing to restore yet)", flush=True)
'''
    s = s[:i4] + restore + s[i4:]

    shutil.copy2(SP, SP + ".bak-%s-pre-autosave" % STAMP)
    open(SP, "w", encoding="utf-8").write(s)
    print("已写补丁：%s（备份 %s.bak-%s-pre-autosave）" % (SP, SP, STAMP))
    print("  新增行数：%d" % (s.count("\n") - orig.count("\n")))

# ---------- 语法检查 ----------
import py_compile
try:
    py_compile.compile(SP, cfile="/tmp/_chk.pyc", doraise=True)
    print("语法检查：OK")
except py_compile.PyCompileError as e:
    print("语法检查失败：", e)
    sys.exit(1)

# ---------- ④ 配置：显式写上 session_autosave（固化）----------
import json
cfg = json.load(open(CFG, encoding="utf-8"))
if "session_autosave" not in cfg:
    shutil.copy2(CFG, CFG + ".bak-%s-pre-autosave" % STAMP)
    cfg["session_autosave"] = "strata-session-autosave.bin"
    json.dump(cfg, open(CFG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("配置已加 session_autosave = strata-session-autosave.bin")
else:
    print("配置已有 session_autosave =", cfg["session_autosave"])
