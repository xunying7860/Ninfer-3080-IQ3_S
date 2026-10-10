#!/bin/bash
# GPU0 归属切换器（x99）
# 用法:
#   switch-gpu0.sh fusion   → 新引擎（ninfer-fusion.service，18084）占用 GPU0，让出生产 ninfer
#   switch-gpu0.sh ninfer   → 生产 ninfer（ninfer.service + watchdog，18082）占用 GPU0，停新引擎
#   switch-gpu0.sh status   → 看两侧服务与 GPU 归属
#
# 为什么必须先停 watchdog：ninfer-watchdog 会真请求探活并在失败时 `systemctl restart ninfer`
# ⇒ 不先停它，新引擎一起来就被生产 ninfer 抢走 GPU0（两边互相顶）。
set -uo pipefail

FUSION=ninfer-fusion.service
PROD=ninfer.service
WD=ninfer-watchdog.service

case "${1:-status}" in
  fusion)
    echo "== 停生产（先 watchdog）+ 禁止自启 =="
    sudo systemctl stop "$WD" 2>/dev/null
    sudo systemctl stop "$PROD" 2>/dev/null
    sudo systemctl disable "$PROD" "$WD" 2>/dev/null
    sleep 3
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    echo "== 起新引擎 =="
    sudo systemctl enable "$FUSION" >/dev/null 2>&1
    sudo systemctl start "$FUSION"
    echo "已发起，等就绪（约 2 min，判据：18084 的 /v1/models 返回 200）"
    ;;
  ninfer)
    echo "== 停新引擎 + 禁止自启 =="
    sudo systemctl stop "$FUSION" 2>/dev/null
    sudo systemctl disable "$FUSION" 2>/dev/null
    sleep 3
    echo "== 恢复生产 =="
    sudo systemctl enable "$PROD" "$WD" >/dev/null 2>&1
    sudo systemctl start "$PROD"
    echo "生产已发起（约 2 min 到 listening on http://0.0.0.0:18082）"
    ;;
  *) : ;;
esac

echo "== 状态 =="
for s in "$FUSION" "$PROD" "$WD" strata; do printf "%-24s active=%s enabled=%s\n" "$s" "$(systemctl is-active $s)" "$(systemctl is-enabled $s 2>/dev/null)"; done
echo "== 端口 =="
ss -ltn | grep -E "18082|18084" || echo "(18082/18084 均未监听)"
echo "== GPU =="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
