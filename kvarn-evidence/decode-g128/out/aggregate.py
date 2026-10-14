import re,sys
p=sys.argv[1]
hdr=None; rows=[]
for line in open(p,encoding='utf-8'):
    s=line.strip()
    if s.startswith('[') and '|' in s: hdr=s
    m=re.search(r'valid=\s*(\d+)\s+cos_min=([\d.]+).*?反证\(不逆旋转\) cos_min=(-?[\d.]+) cos_mean=(-?[\d.]+) .*?vs原值: 逆旋转后 cos_min=(-?[\d.]+) / 不逆旋转 cos_min=(-?[\d.]+) cos_mean=(-?[\d.]+)',line)
    if m:
        rows.append(dict(hdr=hdr,nv=int(m.group(1)),main=float(m.group(2)),rv=float(m.group(3)),
                         rvm=float(m.group(4)),w=float(m.group(5)),nw=float(m.group(6)),nwm=float(m.group(7))))
print("page 行数 =",len(rows),"（计入判据的 token 总数 =",sum(r['nv'] for r in rows),"）")
print("A) 主判据 device_bf16 vs 期望_bf16:  min=%.9f  max=%.9f ; ==1.0 的页数=%d/%d"%(
    min(r['main'] for r in rows),max(r['main'] for r in rows),sum(1 for r in rows if r['main']==1.0),len(rows)))
print("B) 反证 不逆旋转 vs 期望: cos_min in [%.9f, %.9f] ; max|cos_min|=%.9f ; cos_mean in [%.9f, %.9f]"%(
    min(r['rv'] for r in rows),max(r['rv'] for r in rows),max(abs(r['rv']) for r in rows),
    min(r['rvm'] for r in rows),max(r['rvm'] for r in rows)))
print("C) vs 未旋转原值, 逆旋转后: cos_min in [%.9f, %.9f]"%(min(r['w'] for r in rows),max(r['w'] for r in rows)))
print("D) vs 未旋转原值, 不逆旋转: cos_min in [%.9f, %.9f] ; max|cos_min|=%.9f"%(
    min(r['nw'] for r in rows),max(r['nw'] for r in rows),max(abs(r['nw']) for r in rows)))
print()
byname={}
for r in rows:
    m=re.match(r'\[(\w+)\|(\w)\|([^|]+)\|tok=(\d+)\]',r['hdr'])
    byname.setdefault((m.group(1),m.group(2),m.group(3).strip(),int(m.group(4))),[]).append(r)
for k in sorted(byname):
    v=byname[k]
    print("  %-6s %s %-16s tok=%3d  main_min=%.9f  vs原值_min=%.9f  反证max|cos|=%.9f"%(
        k[0],k[1],k[2],k[3],min(x['main'] for x in v),min(x['w'] for x in v),max(abs(x['rv']) for x in v)))
