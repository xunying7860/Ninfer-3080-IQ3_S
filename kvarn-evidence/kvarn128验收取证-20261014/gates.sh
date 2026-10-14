#!/bin/bash
# gates.sh <NAME> [<round-tag>] —— 只读调用 /data/ninfer/test/gate_test.py（不改它）；重启走本目录 armctl.sh。
set -u
ROOT=/data/kvarn/g128-验收-20261014
NAME=${1:?用法: gates.sh <NAME> [tag]}
TAG=${2:-r1}
ENVF=$ROOT/run/$NAME.env
[ -f "$ENVF" ] || { echo "[gates] 缺 $ENVF"; exit 2; }
set -a; . "$ENVF"; set +a
PORT=$ARM_PORT
LOG=$(cat "$ROOT/run/log-$NAME.path")
mkdir -p "$ROOT/ev" "$ROOT/gatelogs"
cd "$ROOT" || exit 9
export GATE_PORT="$PORT" GATE_HOST=127.0.0.1 GATE_LOG="$LOG"
export GATE_RESTART_CMD="bash $ROOT/run/armctl.sh $NAME restart"
export GATE_DOC_LINES="${GATE_DOC_LINES:-1200}"
export GATE_NONCE="${GATE_NONCE:-gate-prod}"
SUITES=${GATE_SUITES:-"ctrl rst rst2 toolsctrl tools apitoolctrl apitool visctrl vis"}
echo "=== [gates] name=$NAME tag=$TAG port=$PORT log=$LOG nonce=$GATE_NONCE doc_lines=$GATE_DOC_LINES suites='$SUITES' ==="
for s in $SUITES; do
  echo "=== [gates] $NAME/$TAG $s start $(date '+%T') ==="
  python3 /data/ninfer/test/gate_test.py run --suite "$s" --out "$ROOT/ev/$NAME-$TAG-$s.json" >"$ROOT/gatelogs/$NAME-$TAG-$s.log" 2>&1
  echo "=== [gates] $NAME/$TAG $s exit=$? end $(date '+%T') ==="
done
echo "=== [gates] compares $NAME/$TAG $(date '+%T') ==="
for pair in "ctrl rst" "rst rst2" "toolsctrl tools" "apitoolctrl apitool" "visctrl vis"; do
  set -- $pair
  [ -f "$ROOT/ev/$NAME-$TAG-$1.json" ] && [ -f "$ROOT/ev/$NAME-$TAG-$2.json" ] || continue
  python3 /data/ninfer/test/gate_test.py compare "$ROOT/ev/$NAME-$TAG-$1.json" "$ROOT/ev/$NAME-$TAG-$2.json" >"$ROOT/gatelogs/$NAME-$TAG-cmp-$1-$2.log" 2>&1
  rc=$?
  echo "--- $NAME/$TAG compare $1 vs $2 exit=$rc ---"
  grep -E "VERDICT|MISMATCH|轮数不一致|^     A:|^     B:" "$ROOT/gatelogs/$NAME-$TAG-cmp-$1-$2.log" || echo "(no verdict line)"
done
echo "=== [gates] ALL DONE $NAME/$TAG $(date '+%T') ==="
