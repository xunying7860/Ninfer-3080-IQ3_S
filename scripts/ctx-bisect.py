#!/usr/bin/env python3
# 2026-10-15 二分找 --max-context/--kv-capacity 最大可启动值（用户指令：调高到起不来为止）
import re, subprocess, time
LOG = '/data/ninfer/logs/ninfer-serve.log'
RUN = '/data/ninfer/run.sh'
RS  = '/data/ninfer/test/ctx-bisect-result.txt'
ANCHOR = re.compile(rb'--max-context \d+ --kv-capacity \d+')
def log(m):
    line = time.strftime('%H:%M:%S ') + m
    with open(RS,'a') as f: f.write(line + '\n')
    print(line, flush=True)
def set_ctx(x):
    s = open(RUN,'rb').read()
    new = ('--max-context %d --kv-capacity %d' % (x,x)).encode()
    s2 = ANCHOR.sub(new, s)
    if s2.count(new) != 1: raise RuntimeError('anchor count %d' % s2.count(new))
    open(RUN,'wb').write(s2)
def all_lines():
    with open(LOG,'rb') as f: return f.read().split(b'\n')
def try_ctx(x, wait_s=240):
    set_ctx(x)
    subprocess.run(['sudo','systemctl','stop','ninfer'], capture_output=True)
    mark = len(all_lines())
    subprocess.run(['sudo','systemctl','start','ninfer'], capture_output=True)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        time.sleep(5)
        chunk = all_lines()[mark:]
        txt = b'\n'.join(chunk)
        if b'engine ready' in txt:
            cap = [l for l in chunk if b'capacity |' in l]
            log('CTX=%d OK | %s' % (x, cap[-1].decode('utf-8','replace') if cap else ''))
            return True
        if b'available for runtime capacity' in txt:
            bad = [l for l in chunk if b'available for runtime capacity' in l]
            log('CTX=%d FAIL | %s' % (x, bad[-1].decode('utf-8','replace') if bad else ''))
            subprocess.run(['sudo','systemctl','stop','ninfer'], capture_output=True); return False
        if b'FATAL' in txt or b'engine failed' in txt:
            bad = [l for l in chunk if b'FATAL' in l or b'engine failed' in l]
            log('CTX=%d FATAL | %s' % (x, bad[-1].decode('utf-8','replace') if bad else ''))
            subprocess.run(['sudo','systemctl','stop','ninfer'], capture_output=True); return False
    log('CTX=%d TIMEOUT' % x)
    subprocess.run(['sudo','systemctl','stop','ninfer'], capture_output=True); return None
lo = 317440
log('BISECT START known_ok=%d' % lo)
up = None
for c in (393216, 524288, 655360, 786432, 917504, 1048576):
    if c <= lo: continue
    r = try_ctx(c)
    if r is True: lo = c
    else: up = c; break
if up is None:
    up = lo * 2
    while True:
        r = try_ctx(up)
        if r is True: lo = up; up = lo * 2
        else: break
while up - lo > 64:
    mid = ((lo + up) // 2 // 64) * 64
    if mid <= lo or mid >= up: break
    r = try_ctx(mid)
    if r is True: lo = mid
    else: up = mid
log('BISECT DONE max_ctx=%d next_fail=%d' % (lo, up))
set_ctx(lo)
subprocess.run(['sudo','systemctl','start','ninfer'], capture_output=True)
log('FINAL ctx=%d written+started' % lo)
