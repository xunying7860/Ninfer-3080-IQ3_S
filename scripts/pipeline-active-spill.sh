#!/usr/bin/env bash
# ★ 一条龙：给 ninfer 加"单会话也能跨重启"（改 flush_disk_tier 让 Active 也落盘）→ 自编 sm_86 → 上卡 → 判据 → 失败自动回滚
# 判据：**单个会话**（不制造第二个会话）→ 重启引擎 → 回同一题面 ⇒ 应 ~1 秒内恢复（冷 prefill 是 ~27 秒）
set -u
SC="C:/Users/xunying/AppData/Local/hermes/cache/scratch"
TREE="$SC/ninfer-upstream/full"
OUT="$SC/ninfer-build-patched"
LOG="$OUT/pipeline.log"
X99="xunying@169.254.146.59"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

mkdir -p "$OUT"
say "=== 0) 打补丁（只改一处判定）==="
F="$TREE/src/models/qwen3_5/program/storage/disk_tier.cpp"
cp -f "$F" "$OUT/disk_tier.cpp.orig"
python - "$F" <<'PY'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = ("        if (continuation_slots[index].role != ContinuationSlotRole::Catalogued) { continue; }\n")
new = ("        // ★ 我方补丁（2026-10-10）：关机时连\"正在用(Active)\"的那条也落盘。\n"
       "        // 原生只落 Catalogued（被挤出/已编目的），于是\"从头到尾只有一个会话\"时优雅重启\n"
       "        // 后它不在磁盘上，必须整段重算 prefill —— 用户明确不要这种体验。\n"
       "        // 本函数只在 ~ProgramImpl 里被调用（关机时刻），Active 的续段状态此刻是稳定可写的；\n"
       "        // 落盘键仍是提示词摘要，重启后同一题面即可从磁盘命中。\n"
       "        if (continuation_slots[index].role != ContinuationSlotRole::Catalogued &&\n"
       "            continuation_slots[index].role != ContinuationSlotRole::Active) { continue; }\n")
if "我方补丁（2026-10-10）" in s:
    print("补丁已在"); raise SystemExit(0)
if old not in s:
    print("锚点不匹配，未改"); raise SystemExit(2)
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("已打补丁")
PY
[ $? -eq 0 ] || { say "补丁失败，中止"; exit 1; }
grep -n -A7 "我方补丁（2026-10-10）" "$F" | head -12 | tee -a "$LOG"

say "=== 1) Docker 构建（与当初编 fork 同基座同 Dockerfile；CUDA 编译并行度 8）==="
IMG="ninfer-patched-$(date +%Y%m%d-%H%M%S)"
docker build --progress=plain -t "$IMG" -f "$TREE/Dockerfile" "$TREE" >>"$LOG" 2>&1 || { say "docker build 失败，中止（完整输出见本日志）"; exit 1; }
say "镜像已建：$IMG"

say "=== 2) 取出二进制 ==="
CID=$(docker create "$IMG")
docker cp "$CID":/src/build/apps/ninfer-serve "$OUT/ninfer-serve.new" 2>>"$LOG" || \
  { # 回退：在镜像里找
    docker cp "$CID":/src/build-linux/apps/ninfer-serve "$OUT/ninfer-serve.new" 2>>"$LOG" || \
    { docker run --rm "$IMG" bash -lc 'find / -name ninfer-serve -type f 2>/dev/null' > "$OUT/found.txt" 2>&1; say "没取到二进制，见 found.txt"; docker rm "$CID" >/dev/null; exit 1; }; }
docker rm "$CID" >/dev/null
ls -lh "$OUT/ninfer-serve.new" | tee -a "$LOG"

say "=== 3) 校验：补丁串在、架构 sm_86、sha256 ==="
PATCHED=$(strings -a "$OUT/ninfer-serve.new" | grep -c "我方补丁" || true)
say "二进制里的补丁串计数=$PATCHED（源码里那条中文注释不会进二进制，所以 0 也正常）"
file "$OUT/ninfer-serve.new" 2>/dev/null | tee -a "$LOG" || true
sha256sum "$OUT/ninfer-serve.new" | tee -a "$LOG"
# 真正的判据：行为（下面第 5 步）

say "=== 4) 上卡（备份旧件 + 只换二进制；同基座 ⇒ 现有 ldd 闭包仍适用）==="
ssh -o BatchMode=yes "$X99" 'set -e
  D=/data/ninfer/bundle/bin/ninfer-serve
  cp -f $D $D.bak-'"$(date +%Y%m%d-%H%M%S)"'-pre-activepatch
  ls -l $D.bak-*pre-activepatch | tail -1'  | tee -a "$LOG"
scp -o BatchMode=yes "$OUT/ninfer-serve.new" "$X99":/data/ninfer/bundle/bin/ninfer-serve.new | tee -a "$LOG"
ssh -o BatchMode=yes "$X99" 'set -e
  D=/data/ninfer/bundle/bin/ninfer-serve
  mv -f $D.new $D
  chmod +x $D
  md5sum $D
  T0=$(date +%s); sudo -n systemctl restart ninfer.service
  for i in $(seq 1 200); do c=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:18082/v1/models 2>/dev/null || true); [ "$c" = 200 ] && break; sleep 1; done
  echo "重启到 models=200：$(( $(date +%s)-T0 )) 秒 code=$c"' | tee -a "$LOG"

say "=== 5) 判据：单会话 → 重启 → 回同一题面（不制造第二个会话！）==="
scp -o BatchMode=yes "$SC/../scratch/ninfer-single-restart-test.py" "$X99":/tmp/ 2>/dev/null || \
python - <<'PY' > "$OUT/single-restart-test.py"
print('''#!/usr/bin/env python3
import json,re,subprocess,time,urllib.request
LOG="/data/ninfer/logs/ninfer-serve.log"
SYS="You are a careful assistant. Answer briefly and exactly."
def doc(n,m):
    return "\\n".join("Line %06d: %s"%(i, "MARK-%s at %d"%(m,i) if i==n//2 else
        "Line %06d: routine maintenance notes, inventory records and log entries follow."%i) for i in range(n))
def ask(tag,text,timeout=1800):
    body={"model":"qwen3.8-27b","messages":[{"role":"system","content":SYS},
          {"role":"user","content":text+"\\n\\nQuestion: how many lines? Answer the number only."}],
          "max_tokens":12,"temperature":1,"top_k":20,"reasoning_effort":"none",
          "chat_template_kwargs":{"enable_thinking":False}}
    r=urllib.request.Request("http://127.0.0.1:18082/v1/chat/completions",
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t=time.time()
    with urllib.request.urlopen(r,timeout=timeout) as x: d=json.load(x)
    el=time.time()-t; pt=d.get("usage",{}).get("prompt_tokens")
    time.sleep(0.6); line=""
    for ln in reversed(open(LOG,errors="replace").readlines()[-400:]):
        if "req#" in ln and "done" in ln and format(pt,"," ) in ln: line=ln.strip(); break
    print("  [%s] %6.1fs prompt=%s | %s"%(tag,el,pt,line[-110:]),flush=True)
    return el
P=doc(1200,"SINGLE")
print("### 单会话：第 1 发",flush=True); a=ask("发1",P)
print("### 直接重启（**不制造第二个会话**）",flush=True)
t0=time.time(); subprocess.run(["sudo","-n","systemctl","restart","ninfer.service"],check=True)
for _ in range(200):
    time.sleep(1)
    try:
        if urllib.request.urlopen("http://127.0.0.1:18082/v1/models",timeout=3).status==200: break
    except Exception: pass
print("  重启到就绪 %.0f 秒"%(time.time()-t0),flush=True)
print("### 回同一题面",flush=True); b=ask("发2-重启后",P)
print("### 判据：重启后 %.1fs / 首发 %.1fs ⇒ %s"%(b,a,"✅ 修好（来自磁盘）" if b < a/5 else "❌ 未生效（仍在全量 prefill）"),flush=True)''')
PY
scp -o BatchMode=yes "$OUT/single-restart-test.py" "$X99":/tmp/single-restart-test.py | tee -a "$LOG"
ssh -o BatchMode=yes "$X99" 'setsid nohup python3 -u /tmp/single-restart-test.py > /tmp/single-restart-test.log 2>&1 </dev/null & sleep 5; echo launched' | tee -a "$LOG"
for i in $(seq 1 40); do sleep 15; grep -q "判据" /tmp/single-restart-test.log 2>/dev/null && break; ssh -o BatchMode=yes "$X99" 'grep -q "判据" /tmp/single-restart-test.log 2>/dev/null && exit 0 || exit 1' || true; done
ssh -o BatchMode=yes "$X99" 'cat /tmp/single-restart-test.log' | tee -a "$LOG"

say "=== 6) 判定 + 失败自动回滚 ==="
VERDICT=$(ssh -o BatchMode=yes "$X99" 'grep -o "✅ 修好（来自磁盘）\|❌ 未生效（仍在全量 prefill）" /tmp/single-restart-test.log | tail -1')
say "判定：$VERDICT"
if [ "$VERDICT" != "✅ 修好（来自磁盘）" ]; then
  say "未通过 ⇒ 回滚到打补丁前的二进制"
  ssh -o BatchMode=yes "$X99" 'set -e
    D=/data/ninfer/bundle/bin/ninfer-serve
    B=$(ls -t $D.bak-*pre-activepatch | head -1)
    cp -f "$B" $D && chmod +x $D && sudo -n systemctl restart ninfer.service
    echo "已回滚到 $B"'
else
  say "通过 ⇒ 保留新件。备份件仍在 $X99:/data/ninfer/bundle/bin/ninfer-serve.bak-*pre-activepatch"
fi
say "=== 完成 ==="
