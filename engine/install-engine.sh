#!/usr/bin/env bash
# 安装 NInfer 引擎 bundle（zstd 分卷，本目录内的 *.tar.zst.part-*）。
# 用法：
#   bash install-engine.sh                 # 解到 /data/ninfer
#   bash install-engine.sh /data/ninfer    # 同上（显式）
#   bash install-engine.sh --check-only    # 只校验分卷与重接后哈希，不动任何文件
# 原则：旧引擎件与旧 libs 一律改名留档，绝不删除；每一步都打数字。
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
NAME=engine-bundle-2cbd88ab
SUMS="$HERE/checksums.txt"
INNER_MD5="2cbd88ab06318c81b695ec64c72b433d"   # bundle/bin/ninfer-serve 应有 md5
CHECK_ONLY=0
ROOT=/data/ninfer

for a in "$@"; do
  case "$a" in
    --check-only) CHECK_ONLY=1 ;;
    *) ROOT="$a" ;;
  esac
done

say(){ printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die(){ printf '[%s] 停止：%s\n' "$(date '+%H:%M:%S')" "$*" >&2; exit 1; }

command -v md5sum >/dev/null || die "缺 md5sum"
[ -f "$SUMS" ] || die "找不到 $SUMS"

ARCH_MD5=$(awk '/^archive md5 /{print $3; exit}' "$SUMS")
[ -n "$ARCH_MD5" ] || die "checksums.txt 里没有 archive md5 行"

# ---- 1. 分卷清单与逐卷 md5 ----
mapfile -t PARTS < <(ls -1 "$HERE"/"$NAME".tar.zst.part-* 2>/dev/null | sort)
[ "${#PARTS[@]}" -ge 1 ] || die "本目录里没有 $NAME.tar.zst.part-*（分卷没拉全？）"
say "分卷 ${#PARTS[@]} 个："
for p in "${PARTS[@]}"; do
  want=$(awk -v n="$(basename "$p")" '$2==n{print $1; exit}' "$SUMS")
  got=$(md5sum "$p" | awk '{print $1}')
  size=$(stat -c %s "$p")
  if [ -n "$want" ] && [ "$want" = "$got" ]; then say "  OK $(basename "$p") $size B $got"
  else say "  ?? $(basename "$p") $size B $got （清单里 $(echo "${want:-无}")）"
       [ -n "$want" ] && die "分卷 $(basename "$p") md5 与 checksums.txt 不符"
  fi
done

# ---- 2. 重接后必须等于 archive md5 ----
say "重接分卷并校验整包 md5 ..."
RE=$(cat "${PARTS[@]}" | md5sum | awk '{print $1}')
if [ "$RE" != "$ARCH_MD5" ]; then die "重接后 md5=$RE ≠ 清单 $ARCH_MD5（分卷顺序或内容有问题）"; fi
say "  OK 整包 md5 $RE"
if [ "$CHECK_ONLY" = 1 ]; then say "--check-only：不解压、不落盘，结束"; exit 0; fi

command -v zstd >/dev/null || die "缺 zstd（Ubuntu: sudo apt install zstd）"

# ---- 3. 旧件留档（不删） ----
TS=$(date +%Y%m%d-%H%M%S)
KEEP="$ROOT/archive-engine-$TS"
if [ -e "$ROOT/bundle/bin/ninfer-serve" ] || [ -d "$ROOT/bundle/libs" ]; then
  mkdir -p "$KEEP"
  [ -e "$ROOT/bundle/bin/ninfer-serve" ] && { mv "$ROOT/bundle/bin/ninfer-serve" "$KEEP/ninfer-serve.old"; say "旧引擎件 → $KEEP/ninfer-serve.old"; }
  [ -d "$ROOT/bundle/libs" ] && { mv "$ROOT/bundle/libs" "$KEEP/libs.old"; say "旧 libs → $KEEP/libs.old"; }
fi
mkdir -p "$ROOT"

# ---- 4. 解压 ----
say "解压 ${#PARTS[@]} 个分卷 → $ROOT/bundle ..."
cat "${PARTS[@]}" | zstd -d -T"$(nproc)" -q -o - | tar -xC "$ROOT"
chmod 755 "$ROOT/bundle/bin/ninfer-serve"
say "解出："; ls -lL "$ROOT/bundle/bin/ninfer-serve" | sed 's/^/  /'
du -sb "$ROOT/bundle" | sed 's/^/  /'

# ---- 5. 解出内容逐文件 md5 核对 ----
say "逐文件 md5 核对（清单在 checksums.txt 的 inner md5 段）..."
cd "$ROOT"
find bundle/bin/ninfer-serve bundle/libs -type f -exec md5sum {} + | sort -k2 > /tmp/engine-extracted-$TS.md5
awk '/^# inner md5 begin/{f=1;next} /^# inner md5 end/{f=0} f' "$SUMS" | sort -k2 > /tmp/engine-expected-$TS.md5
if cmp -s /tmp/engine-extracted-$TS.md5 /tmp/engine-expected-$TS.md5; then
  say "  OK $(wc -l < /tmp/engine-extracted-$TS.md5) 个文件逐 md5 一致"
else
  say "  差异（前 10 行）："; diff /tmp/engine-expected-$TS.md5 /tmp/engine-extracted-$TS.md5 | head -10 | sed 's/^/  /'
  die "解出内容与清单不一致"
fi
got=$(md5sum "$ROOT/bundle/bin/ninfer-serve" | awk '{print $1}')
[ "$got" = "$INNER_MD5" ] || die "引擎件 md5=$got ≠ 期望 $INNER_MD5"
say "  OK 引擎件 md5 $got"

say "loader 与库路径（run.sh 用的两条）："
say "  $ROOT/bundle/libs/lib64/ld-linux-x86-64.so.2"
say "  $ROOT/bundle/libs/lib/x86_64-linux-gnu"
say "模型权重不在包里，需自行放到模型目录后用 run.sh 指定的路径启动。"
