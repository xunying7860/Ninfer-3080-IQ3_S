#!/bin/bash
# armctl3.sh <NAME> startfresh|start|stop|restart
# 通用版：NAME 对应 run/<NAME>.env（内含 ARM_BIN/ARM_PORT/ARM_PROFILE/ARM_EXTRA/ARM_EFFORT/ARM_CG_MIB）。
# 一次只起一个实例；普通进程，绝不碰 systemd；停用 PID，绝不 pkill -f。
set -u
ROOT=/data/kvarn/g128-rk4v4-test
NAME=${1:?用法: armctl3.sh <NAME> startfresh|start|stop|restart}; MODE=${2:?}
ENVF=$ROOT/run/$NAME.env
[ -f "$ENVF" ] || { echo "[armctl3] 缺 $ENVF"; exit 2; }
set -a; . "$ENVF"; set +a
PIDF=$ROOT/run/pid-$NAME
CUR=$ROOT/run/log-$NAME.path
mkdir -p "$ROOT/logs" "$ROOT/run"

stop_arm() {
  if [ ! -f "$PIDF" ]; then echo "[armctl3] no pidfile for $NAME"; return 0; fi
  pid=$(cat "$PIDF")
  if ! kill -0 "$pid" 2>/dev/null; then echo "[armctl3] pid $pid not alive"; return 0; fi
  echo "[armctl3] $(date '+%F %T') SIGTERM -> pid=$pid"
  kill -TERM "$pid"
  for i in $(seq 1 150); do
    if ! kill -0 "$pid" 2>/dev/null; then echo "[armctl3] pid=$pid exited after ${i}s"; return 0; fi
    sleep 1
  done
  echo "[armctl3] 150s 未退出 ⇒ SIGKILL"
  kill -KILL "$pid"; sleep 2
}
newlog() { TS=$(date +%Y%m%d-%H%M%S); echo "$ROOT/logs/$NAME-$TS.log" > "$CUR"; echo "[armctl3] new log: $(cat "$CUR")"; }
launch_arm() {
  log=$(cat "$CUR")
  echo "[armctl3] $(date '+%F %T') launch name=$NAME bin=$ARM_BIN port=$ARM_PORT profile=$ARM_PROFILE extra='${ARM_EXTRA:-}' effort=${ARM_EFFORT:-high} cg=${ARM_CG_MIB:-1024} log=$log" >> "$log"
  setsid nohup bash "$ROOT/armgen.sh" >>"$log" 2>&1 </dev/null &
  echo $! > "$PIDF"
  echo "[armctl3] launched pid=$(cat "$PIDF") log=$log"
}
case "$MODE" in
  startfresh) stop_arm; newlog; launch_arm ;;
  start)      [ -f "$CUR" ] || newlog; launch_arm ;;
  restart)    stop_arm; [ -f "$CUR" ] || newlog; launch_arm ;;
  stop)       stop_arm ;;
  *) echo "用法: armctl3.sh <NAME> startfresh|start|stop|restart"; exit 2 ;;
esac
