#!/usr/bin/env bash
# 一次性探子（2026-10-12）：只为抓 master 件的 /metrics 与 /health 两份指纹。
# 为什么要它：看门狗 engine-watchdog.py 的判据只认三个指标名
#   llamacpp:prompt_tokens_total（缺了就 counters=None ⇒ 计数停滞永不成立 ⇒ 对活锁失明）
#   llamacpp:tokens_predicted_total、llamacpp:kv_cache_usage_ratio，外加 /health 里的 "status":"ok"
# ⇒ 换引擎线之前必须知道 master 线还给不给这三个名字。
# 动作：停生产 → 起 master 候选（allowance 192、磁盘层关、独立 profile）→ 抓两份 → 停候选 → 恢复生产。
# 不入 systemd、不改任何配置、不删任何东西。日志 /data/ninfer/test/logs/probe-master.log
set -u
cd /data/ninfer/test || exit 9
OUT=/data/ninfer/test/probe-master
mkdir -p "$OUT" logs
say() { echo "[probe $(date '+%T')] $*" | tee -a logs/probe-master.log; }
export CAND_BIN=/data/ninfer/bundle/bin/ninfer-serve.master
export CAND_NO_DISK=1 CAND_ALLOWANCE=192
export CAND_DISK=/data/nvme/ninfer-disktier-probe
export CAND_EXTRA="--device-profile-path /data/ninfer/profiles-master-e.json"

say "停生产 ninfer"
sudo -n systemctl stop ninfer.service
sleep 4
say "起 master 候选"
bash cand.sh restart 2>&1 | tee -a logs/probe-master.log | tail -2

code=$(curl -s -m 10 -o "$OUT/health.json" -w '%{http_code}' http://127.0.0.1:18082/health)
say "GET /health -> http=$code（已存 $OUT/health.json）"
mcode=$(curl -s -m 10 -o "$OUT/metrics.txt" -w '%{http_code}' http://127.0.0.1:18082/metrics)
say "GET /metrics -> http=$mcode 行数=$(wc -l < "$OUT/metrics.txt")"
grep -aE "^llamacpp:(prompt_tokens_total|tokens_predicted_total|kv_cache_usage_ratio)[[:space:]]" "$OUT/metrics.txt" \
  | cut -c1-90 | tee "$OUT/watchdog-keys.txt"
say "看门狗要的三条指标在 master 上的命中数=$(wc -l < "$OUT/watchdog-keys.txt")/3"

say "抓完 ⇒ 停候选、恢复生产"
bash cand.sh stop 2>&1 | tee -a logs/probe-master.log | tail -2
sudo -n systemctl start ninfer.service
for i in $(seq 1 300); do
  [ "$(curl -s -o /dev/null -m 3 -w '%{http_code}' http://127.0.0.1:18082/v1/models)" = 200 ] && break
  sleep 2
done
say "生产已回 active=$(systemctl is-active ninfer) models=$(curl -s -o /dev/null -m 5 -w '%{http_code}' http://127.0.0.1:18082/v1/models)"
say "PROBE_DONE"
