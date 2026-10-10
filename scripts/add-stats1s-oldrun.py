#!/usr/bin/env python3
# 给旧引擎 run.sh 加 --log-stats-interval-ms 1000（幂等；改前备份）
# 依据：桌面面板的 tg 取自**日志 throughput 行的 decode 速率**（引擎唯一的连续速率信号）；
#       默认 5000ms ⇒ tg 5 秒才刷新一次，看起来和 tg_3s 同值 ⇒ 收到 1000ms。
# ★ 行尾必须带反斜杠，且续行中间绝不能写注释（shell 不把非词首 # 当注释）。
import shutil
import sys
import time

P = "/data/ninfer/run.sh"
ANCHOR = '  --chat-template "$ROOT/chat_template.jinja" --default-reasoning-effort max --default-max-tokens 131072 \\\n'
NEW = ('  --chat-template "$ROOT/chat_template.jinja" --default-reasoning-effort max --default-max-tokens 131072 \\\n'
       '  --log-stats-interval-ms 1000 \\\n')

s = open(P, encoding="utf-8").read()
if "--log-stats-interval-ms" in s:
    print("已经设置，跳过")
    sys.exit(0)
n = s.count(ANCHOR)
if n != 1:
    print("锚点命中 %d 次（期望 1）—— 不写盘" % n)
    sys.exit(1)
shutil.copy2(P, P + ".bak-%s-pre-stats1s" % time.strftime("%Y%m%d-%H%M%S"))
open(P, "w", encoding="utf-8").write(s.replace(ANCHOR, NEW))
print("已加：--log-stats-interval-ms 1000")
