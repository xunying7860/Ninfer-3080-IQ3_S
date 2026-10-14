#!/usr/bin/env bash
# 候选件重启（模拟生产的 SIGTERM 优雅停机路径：先 TERM，等退出，再拉起）
set -u
ROOT=/data/ninfer
PIDF=$ROOT/test/cand.pid
LOG=$ROOT/logs/cand-serve.log
if [ -f "$PIDF" ]; then
  pid=$(cat "$PIDF")
  if kill -0 "$pid" 2>/dev/null; then
    echo "[restart] $(date '+%F %T') SIGTERM -> pid=$pid"
    kill -TERM "$pid"
    for i in $(seq 1 120); do
      kill -0 "$pid" 2>/dev/null || { echo "[restart] exited after ${i}s"; break; }
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "[restart] still alive after 120s, SIGKILL"
      kill -KILL "$pid"
    fi
  fi
fi
echo "[restart] $(date '+%F %T') relaunch"
setsid nohup bash -c "exec bash $ROOT/test/run-cand.sh" </dev/null >>"$LOG" 2>&1 &
echo $! > "$PIDF"
echo "[restart] new pid=$(cat $PIDF)"
