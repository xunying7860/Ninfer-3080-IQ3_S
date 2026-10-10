#!/usr/bin/env bash
# ★ 诊断轮：给磁盘层的关机落盘加"编译安全"的打印，编一次、上卡、跑单会话判据、抓诊断、自动回滚
# 目的：一次拿到硬证据 —— ① flush_disk_tier 关机时到底被调用没 ② 各角色槽位计数
#       ③ spill_owner_to_disk 里 sequence.kv 是否为空 ④ plan 出的条目数（=实写量）
# 只用**已验证存在**的字段：continuation_slots[i].role、sequence.kv、spill.items（都从上游源码里读到过）
set -u
SC="C:/Users/xunying/AppData/Local/hermes/cache/scratch"
TREE="$SC/ninfer-upstream/full"
OUT="$SC/ninfer-diag"; LOG="$OUT/diag-pipeline.log"; X99="xunying@169.254.146.59"
mkdir -p "$OUT"; say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== 1) 源码加诊断（幂等、编译安全）==="
python - "$TREE/src/models/qwen3_5/program/storage/disk_tier.cpp" <<'PY'
import sys
p = sys.argv[1]; s = open(p, encoding="utf-8").read()
if "[diag]" in s:
    print("诊断已在"); raise SystemExit(0)

# ① flush_disk_tier：调用标记 + 角色计数
old_flush = """void ProgramImpl::flush_disk_tier() noexcept {
    if (!disk_kv) { return; }"""
new_flush = """void ProgramImpl::flush_disk_tier() noexcept {
    // ★ 诊断（临时）：确认关机时本函数是否被调用、以及各角色槽位计数
    {
        unsigned f = 0, rm = 0, a = 0, c = 0;
        for (std::uint32_t i = 0; i < continuation_capacity; ++i) {
            switch (continuation_slots[i].role) {
            case ContinuationSlotRole::Free: ++f; break;
            case ContinuationSlotRole::ReservedMaterialization: ++rm; break;
            case ContinuationSlotRole::Active: ++a; break;
            case ContinuationSlotRole::Catalogued: ++c; break;
            }
        }
        std::fprintf(stderr, "[diag] flush_disk_tier called: disk_kv=%d capacity=%u free=%u reserved=%u active=%u catalogued=%u\\n",
                     disk_kv ? 1 : 0, (unsigned) continuation_capacity, f, rm, a, c);
        std::fflush(stderr);
    }
    if (!disk_kv) { return; }"""
assert old_flush in s, "flush 锚点不匹配"
s = s.replace(old_flush, new_flush, 1)

# 每个被落盘的槽：角色 + kv 是否存在 + plan 条目数
old_body = """    const auto deadline = std::chrono::steady_clock::now() + kShutdownSpillBudget;
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role != ContinuationSlotRole::Catalogued &&
            continuation_slots[index].role != ContinuationSlotRole::Active) { continue; }
        if (std::chrono::steady_clock::now() > deadline) { break; }
        spill_owner_to_disk(continuation_states[index], deadline);"""
new_body = """    const auto deadline = std::chrono::steady_clock::now() + kShutdownSpillBudget;
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role != ContinuationSlotRole::Catalogued &&
            continuation_slots[index].role != ContinuationSlotRole::Active) { continue; }
        if (std::chrono::steady_clock::now() > deadline) { break; }
        std::fprintf(stderr, "[diag] spilling slot %u role=%d kv=%d\\n", (unsigned) index,
                     (int) continuation_slots[index].role, continuation_states[index].kv ? 1 : 0);
        std::fflush(stderr);
        spill_owner_to_disk(continuation_states[index], deadline);"""
if old_body in s:
    s = s.replace(old_body, new_body, 1)
else:
    print("注意：flush 循环体没匹配上（可能上一轮的 Active 改动形态不同），只加函数级诊断")

# ② spill_owner_to_disk：空 kv 的早退与 plan 条目数
old_sp = """void ProgramImpl::spill_owner_to_disk(const SequenceState& sequence,
                                      std::chrono::steady_clock::time_point deadline) noexcept {
    if (!disk_kv || !sequence.kv) { return; }
    try {
        DiskOwnerSpill spill = plan_owner_spill(sequence);"""
new_sp = """void ProgramImpl::spill_owner_to_disk(const SequenceState& sequence,
                                      std::chrono::steady_clock::time_point deadline) noexcept {
    if (!disk_kv || !sequence.kv) {
        std::fprintf(stderr, "[diag] spill_owner_to_disk early-return: disk_kv=%d kv=%d\\n",
                     disk_kv ? 1 : 0, sequence.kv ? 1 : 0);
        std::fflush(stderr);
        return;
    }
    try {
        DiskOwnerSpill spill = plan_owner_spill(sequence);
        std::fprintf(stderr, "[diag] spill plan items=%zu stopped=%d\\n", spill.items.size(), spill.stopped ? 1 : 0);
        std::fflush(stderr);"""
assert old_sp in s, "spill 锚点不匹配"
s = s.replace(old_sp, new_sp, 1)

open(p, "w", encoding="utf-8").write(s)
print("诊断已加")
PY
[ $? -eq 0 ] || { say "加诊断失败，中止"; exit 1; }
grep -c "\[diag\]" "$TREE/src/models/qwen3_5/program/storage/disk_tier.cpp" | tee -a "$LOG"

say "=== 2) Docker 构建（CUDA 并行度按上游 Dockerfile；ccache 只给 C/CXX）==="
IMG="ninfer-diag-$(date +%Y%m%d-%H%M%S)"
docker build --progress=plain -t "$IMG" -f "$TREE/Dockerfile" "$TREE" >>"$LOG" 2>&1 || { say "构建失败（完整输出在日志）"; exit 1; }
say "镜像已建：$IMG"

say "=== 3) 取件（上游 Dockerfile 装到 /usr/local/bin）==="
CID=$(docker create "$IMG"); OK=0
for P in /usr/local/bin/ninfer-serve /out/ninfer-serve; do
  docker cp "$CID:$P" "$OUT/ninfer-serve.new" 2>/dev/null && { say "取件成功：$P"; OK=1; break; }
done
[ "$OK" = "1" ] || { say "取件失败"; docker rm "$CID" >/dev/null; exit 1; }
docker rm "$CID" >/dev/null
ls -lh "$OUT/ninfer-serve.new" | tee -a "$LOG"; sha256sum "$OUT/ninfer-serve.new" | tee -a "$LOG"

say "=== 4) 上卡（备份+换件+重启）==="
STAMP=$(date +%Y%m%d-%H%M%S)
ssh -o BatchMode=yes "$X99" "cp -f /data/ninfer/bundle/bin/ninfer-serve /data/ninfer/bundle/bin/ninfer-serve.bak-$STAMP-diag && echo backed-up" | tee -a "$LOG"
scp -o BatchMode=yes "$OUT/ninfer-serve.new" "$X99":/data/ninfer/bundle/bin/ninfer-serve.new >/dev/null
ssh -o BatchMode=yes "$X99" 'set -e
  D=/data/ninfer/bundle/bin/ninfer-serve; mv -f $D.new $D; chmod +x $D
  : > /data/ninfer/logs/ninfer-serve.log
  T0=$(date +%s); sudo -n systemctl restart ninfer.service
  for i in $(seq 1 240); do c=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:18082/v1/models 2>/dev/null || true); [ "$c" = 200 ] && break; sleep 1; done
  echo "重启到 models=200：$(( $(date +%s)-T0 )) 秒 code=$c"' | tee -a "$LOG"

say "=== 5) 单会话判据（不制造第二个会话）==="
scp -o BatchMode=yes "$OUT/../ninfer-build-patched/deploy/single-restart-test.py" "$X99":/tmp/srt.py >/dev/null 2>&1 || \
  scp -o BatchMode=yes "$SC/ninfer-build-patched/single-restart-test.py" "$X99":/tmp/srt.py >/dev/null 2>&1 || true
ssh -o BatchMode=yes "$X99" 'ls -l /tmp/srt.py 2>/dev/null || echo "缺测试脚本"' | tee -a "$LOG"
ssh -o BatchMode=yes "$X99" 'setsid nohup python3 -u /tmp/srt.py > /tmp/srt.log 2>&1 </dev/null & sleep 3; echo launched' | tee -a "$LOG"
for i in $(seq 1 30); do sleep 15; ssh -o BatchMode=yes "$X99" 'grep -q VERDICT /tmp/srt.log 2>/dev/null' && break; done
ssh -o BatchMode=yes "$X99" 'cat /tmp/srt.log' | tee -a "$LOG"

say "=== 6) 关机期的诊断行（关键证据）==="
ssh -o BatchMode=yes "$X99" 'grep -a "\[diag\]" /data/ninfer/logs/ninfer-serve.log | tail -25' | tee -a "$LOG"
say "=== 7) 仅当判据未过才回滚 ==="
V=$(ssh -o BatchMode=yes "$X99" 'grep -o "VERDICT: [A-Z]*" /tmp/srt.log | tail -1')
say "判定：$V"
if [ "$V" != "VERDICT: PASS" ]; then
  ssh -o BatchMode=yes "$X99" "set -e
    D=/data/ninfer/bundle/bin/ninfer-serve
    B=\$(ls -t /data/ninfer/bundle/bin/ninfer-serve.bak-*-diag | head -1)
    cp -f \"\$B\" \$D; chmod +x \$D; sudo -n systemctl restart ninfer.service
    for i in \$(seq 1 240); do c=\$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:18082/v1/models 2>/dev/null || true); [ \"\$c\" = 200 ] && break; sleep 1; done
    echo \"未过 ⇒ 已回滚 \$B models=\$c\"" | tee -a "$LOG"
else
  say "✅ 通过 ⇒ 保留新件（备份：/data/ninfer/bundle/bin/ninfer-serve.bak-*-diag）"
fi
say "=== 完成（诊断证据见上）==="
