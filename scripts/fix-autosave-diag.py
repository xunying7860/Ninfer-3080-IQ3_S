#!/usr/bin/env python3
# 修复上一轮 heredoc 引号弄坏的诊断行（语法错误）—— 用干净的三引号替换块
import py_compile
import sys

P = "/data/workspace/strata01403/serve/server.py"
s = open(P, encoding="utf-8").read()

BAD_START = "        path = self.autosave_path\n"
BAD_END = "                return\n"
i = s.index(BAD_START)
# 找这段的结束（下一个 "        r = self.session_file(\"save\", path)" 之前）
j = s.index('            r = self.session_file("save", path)', i)

GOOD = '''        path = self.autosave_path
        if not path or self.proc is None:
            print("[strata] autosave skipped: path=%r proc=%r" % (path, self.proc), flush=True)
            return
        try:
            if self.proc.poll() is not None or not self.alive():
                print("[strata] autosave skipped: engine already gone rc=%r ended=%r"
                      % (self.proc.poll(), getattr(self, "ended", None)), flush=True)
                return
'''
s = s[:i] + GOOD + s[j:]
open(P, "w", encoding="utf-8").write(s)
try:
    py_compile.compile(P, cfile="/tmp/_chk2.pyc", doraise=True)
    print("语法检查：OK")
except py_compile.PyCompileError as e:
    print("语法检查仍失败：", e)
    sys.exit(1)
print("已修复诊断行")
