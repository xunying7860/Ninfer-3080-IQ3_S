#!/usr/bin/env bash
# 续跑：从已建好的镜像取件 → 上卡 → 判据（单会话直接重启）→ 失败自动回滚
set -u
SC="C:/Users/xunying/AppData/Local/hermes/cache/scratch/ninfer-build-patched"
OUT="$SC/deploy"; LOG="$OUT/deploy.log"; X99="xunying@169.254.146.59"
mkdir -p "$OUT"; say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

IMG=$(docker images --format "{{.Repository}}" | grep -i '^ninfer-patched' | head -1)
[ -n "$IMG" ] || { say "找不到 patched 镜像，中止"; exit 1; }
say "用镜像：$IMG"
CID=$(docker create "$IMG")
OK=0
for P in /usr/local/bin/ninfer-serve /out/ninfer-serve /out/bin/ninfer-serve /src/build/apps/ninfer-serve; do
  if docker cp "$CID:$P" "$OUT/ninfer-serve.new" 2>/dev/null; then say "取件成功：$P"; OK=1; break; fi
done
if [ "$OK" != "1" ]; then
  say "候选路径都不存在，在镜像里全盘找："
  docker run --rm --entrypoint bash "$IMG" -lc 'find / -name "ninfer-serve" -type f 2>/dev/null | head -5' | tee -a "$LOG"
  docker rm "$CID" >/dev/null; exit 1
fi
docker rm "$CID" >/dev/null
ls -lh "$OUT/ninfer-serve.new" | tee -a "$LOG"
file "$OUT/ninfer-serve.new" 2>/dev/null | tee -a "$LOG"
sha256sum "$OUT/ninfer-serve.new" | tee -a "$LOG"
# 校验补丁痕迹：新件的字符数应与旧件不同（改了一处判定，符号表/字符串可能不变 ⇒ 以行为判据为主）
ssh -o BatchMode=yes "$X99" 'D=/data/ninfer/bundle/bin/ninfer-serve; echo "旧件大小/时间:"; ls -l $D | cut -c1-80; md5sum $D' | tee -a "$LOG"

say "=== 上卡：备份旧件 → 只替换二进制 ==="
STAMP=$(date +%Y%m%d-%H%M%S)
ssh -o BatchMode=yes "$X99" "cp -f /data/ninfer/bundle/bin/ninfer-serve /data/ninfer/bundle/bin/ninfer-serve.bak-$STAMP-pre-activepatch && ls -l /data/ninfer/bundle/bin/ninfer-serve.bak-$STAMP-pre-activepatch" | tee -a "$LOG"
scp -o BatchMode=yes "$OUT/ninfer-serve.new" "$X99":/data/ninfer/bundle/bin/ninfer-serve.new | tee -a "$LOG"
ssh -o BatchMode=yes "$X99" 'set -e
  D=/data/ninfer/bundle/bin/ninfer-serve
  mv -f $D.new $D; chmod +x $D; ls -l $D | cut -c1-60
  T0=$(date +%s); sudo -n systemctl restart ninfer.service
  for i in $(seq 1 240); do c=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:18082/v1/models 2>/dev/null || true); [ "$c" = 200 ] && break; sleep 1; done
  echo "重启到 models=200：$(( $(date +%s)-T0 )) 秒 code=$c"' | tee -a "$LOG"

say "=== 判据：单会话 → 直接重启 → 回同一题面（不制造第二个会话）==="
python - "$OUT/single-restart-test.py" <<'PY'
import sys
open(sys.argv[1], "w", encoding="utf-8").write(r'''#!/usr/bin/env python3
import json, subprocess, time, urllib.request
LOG="/data/ninfer/logs/ninfer-serve.log"
SYS="You are a careful assistant. Answer briefly and exactly."
def doc(n,m):
    return "\n".join("Line %06d: %s"%(i, "MARK-%s at %d"%(m,i) if i==n//2 else
        "Line %06d: routine maintenance notes, inventory records and log entries follow."%i) for i in range(n))
def ask(tag,text,timeout=1800):
    body={"model":"qwen3.8-27b","messages":[{"role":"system","content":SYS},
          {"role":"user","content":text+"\n\nQuestion: how many lines? Answer the number only."}],
          "max_tokens":12,"temperature":1,"top_k":20,"reasoning_effort":"none",
          "chat_template_kwargs":{"enable_thinking":False}}
    r=urllib.request.Request("http://127.0.0.1:18082/v1/chat/completions",
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t=time.time()
    with urllib.request.urlopen(r,timeout=timeout) as x: d=json.load(x)
    el=time.time()-t; pt=d.get("usage",{}).get("prompt_tokens")
    time.sleep(0.6); line=""
    for ln in reversed(open(LOG,errors="replace").readlines()[-400:]):
        if "req#" in ln and "done" in ln and format(pt,",") in ln: line=ln.strip(); break
    import re
    m=re.search(r"cache ([\d,]+) \(([\d.]+)%, ([a-z ]+)\)",line)
    print("  [%s] %6.1fs prompt=%s cache=%s (%s%%, %s)"%(tag,el,pt,m.group(1) if m else "?",m.group(2) if m else "?",m.group(3).strip() if m else line[-70:]),flush=True)
    return el
P=doc(1200,"SINGLE")
print("### 单会话第 1 发",flush=True); a=ask("发1",P)
print("### 直接重启（**不制造第二个会话**）",flush=True)
t0=time.time(); subprocess.run(["sudo","-n","systemctl","restart","ninfer.service"],check=True)
for _ in range(240):
    time.sleep(1)
    try:
        if urllib.request.urlopen("http://127.0.0.1:18082/v1/models",timeout=3).status==200: break
    except Exception: pass
print("  重启到就绪 %.0f 秒"%(time.time()-t0),flush=True)
print("### 回同一题面（判据）",flush=True); b=ask("发2-重启后",P)
v="✅ 修好（来自磁盘）" if b < a/5 else "❌ 未生效（仍在全量 prefill）"
print("### 判据：重启后 %.1fs vs 首发 %.1fs ⇒ %s"%(b,a,v),flush=True)
print("### VERDICT: %s"%("PASS" if b<a/5 else "FAIL"),flush=True)''')
PY
scp -o BatchMode=yes "$OUT/single-restart-test.py" "$X99":/tmp/single-restart-test.py >/dev/null
ssh -o BatchMode=yes "$X99" 'setsid nohup python3 -u /tmp/single-restart-test.py > /tmp/single-restart-test.log 2>&1 </dev/null & sleep 3; echo launched' | tee -a "$LOG"
for i in $(seq 1 24); do sleep 15; ssh -o BatchMode=yes "$X99" 'grep -q VERDICT /tmp/single-restart-test.log' && break; done
ssh -o BatchMode=yes "$X99" 'cat /tmp/single-restart-test.log' | tee -a "$LOG"

V=$(ssh -o BatchMode=yes "$X99" 'grep -o "VERDICT: [A-Z]*" /tmp/single-restart-test.log | tail -1')
say "判定：$V"
if [ "$V" != "VERDICT: PASS" ]; then
  say "未通过 ⇒ 回滚"
  ssh -o BatchMode=yes "$X99" "set -e
    D=/data/ninfer/bundle/bin/ninfer-serve
    B=\$(ls -t /data/ninfer/bundle/bin/ninfer-serve.bak-*-pre-activepatch | head -1)
    cp -f \"\$B\" \$D; chmod +x \$D; sudo -n systemctl restart ninfer.service
    for i in \$(seq 1 240); do c=\$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:18082/v1/models 2>/dev/null || true); [ \"\$c\" = 200 ] && break; sleep 1; done
    echo \"已回滚到 \$B（models=\$c）\"" | tee -a "$LOG"
else
  say "通过 ⇒ 保留新件（备份在 $X99:/data/ninfer/bundle/bin/ninfer-serve.bak-*-pre-activepatch）"
fi
say "=== 完成 ==="
