#!/usr/bin/env bash
# systemd ExecStop 钩子（2026-10-12 实测语义后写的）：停机【前】让 slot-sidecar 存一次会话。
# 为什么放 ExecStop：这台 systemd 没有 ExecStopPre（systemd-analyze verify: Unknown key name 'ExecStopPre' … ignoring）。
# 实测两条语义（临时单元 ordtest/ordtest2，测完已撤）：
#   1) ExecStop 在 SIGTERM 之前跑完（差 −1.004 s）⇒ 放这里才能"先存再停"
#   2) TimeoutStopSec 把 ExecStop 的执行时间算进去（TimeoutStopSec=4 时 8 s 的 ExecStop 被中途砍）
#      ⇒ 168K 会话一次 save 实测 117.1 s（快照 3,283,941,800 B），必须把 TimeoutStopSec 抬到 300
# 规矩：这个钩子**永远 exit 0** —— 存不上也不能把停服务卡住或让重启失败。
set -u
T=/data/ninfer/test
LOG=/data/ninfer/logs/slot-sidecar.log
mkdir -p /data/ninfer/logs 2>/dev/null
echo "[stop-hook $(date '+%F %T')] 开始：让 sidecar 在 SIGTERM 之前存一次" >> "$LOG" 2>/dev/null
# 停机前的现场快照（哪条会话还活着、digest 是什么）——每次生产停机都留一行可核对的证据
curl -s -m 5 http://127.0.0.1:18082/slots 2>/dev/null | head -c 300 >> "$LOG" 2>/dev/null
echo "" >> "$LOG" 2>/dev/null
# 厂商件（没有 slot 持久化）走的是"这条线没开 ⇒ 不动作"那条分支，正常退出
timeout 240 python3 "$T/slot-sidecar.py" --base http://127.0.0.1:18082 --once --max-retry 40 \
    --state "$T/slot-sidecar-state.json" --log "$LOG" >>"$LOG" 2>&1
rc=$?
echo "[stop-hook $(date '+%F %T')] 结束 rc=$rc（rc 非 0 也只记日志，不拦停机）" >> "$LOG" 2>/dev/null
exit 0
