#!/bin/sh
# 在 ninfer-fusion-kvmem:linux-sm86 容器内运行：
# 抽出 ninfer 二进制 + 动态库闭包（排除 CUDA，运行用 x99 自带 CUDA 13.3）
# 依据 x99-dual-engine/build/extract-linux-bundle.sh（2026-10-03 同法建出生产 bundle）
set -e
OUT=/out
rm -rf "$OUT"; mkdir -p "$OUT/bin" "$OUT/libs"

BINS=""
for f in /usr/local/bin/ninfer-serve /usr/local/bin/ninfer; do
  if [ -f "$f" ]; then cp -a "$f" "$OUT/bin/"; BINS="$BINS $f"; echo "二进制: $f"; fi
done
if [ -z "$BINS" ]; then echo "找不到 ninfer 二进制，列出 /usr/local/bin:"; ls -l /usr/local/bin; exit 1; fi

# ldd 闭包（含传递依赖），排除 CUDA 运行时与驱动库（用 x99 自带）
: > /tmp/libs.txt
for b in $BINS; do
  ldd "$b" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if ($i ~ /^\//) print $i}' >> /tmp/libs.txt || true
done
sort -u /tmp/libs.txt | grep -v '^/usr/local/cuda' > /tmp/libs2.txt || true
mv /tmp/libs2.txt /tmp/libs.txt

n=0
while read -r lib; do
  [ -e "$lib" ] || continue
  d=$(dirname "$lib"); mkdir -p "$OUT/libs$d"
  cp -aL "$lib" "$OUT/libs$d/" 2>/dev/null && n=$((n+1)) || echo "跳过: $lib"
done < /tmp/libs.txt
echo "库文件数: $n"

# 加载器（ldd 不列）
mkdir -p "$OUT/libs/lib64" "$OUT/libs/lib/x86_64-linux-gnu"
cp -aL /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 "$OUT/libs/lib/x86_64-linux-gnu/" 2>/dev/null || true
cp -aL /lib64/ld-linux-x86-64.so.2 "$OUT/libs/lib64/" 2>/dev/null || true

for b in $BINS; do ldd "$b" > "$OUT/ldd-$(basename $b).txt" 2>&1 || true; done
find "$OUT" -type f | sort > "$OUT/manifest.txt"
du -sh "$OUT"
tar czf /work/bundle-linux-sm86-fusion.tar.gz -C "$OUT" .
ls -l /work/bundle-linux-sm86-fusion.tar.gz
