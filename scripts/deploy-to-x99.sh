#!/bin/bash
# 把本机构建出的 Linux sm_86 bundle 投到 x99（SATA 盘：/ = sda）
# 用法：bash deploy-to-x99.sh <bundle.tar.gz> <远端sha256（可选，用于比对）>
set -euo pipefail
X99=${X99:-xunying@169.254.146.59}
TAR=${1:?bundle tar.gz 路径}
TARGET=${TARGET:-/data/ninfer-fusion-kvmem}

echo "== 本机 sha256 =="
LOCAL_SHA=$(sha256sum "$TAR" | cut -d' ' -f1)
LOCAL_SIZE=$(stat -c %s "$TAR")
echo "$LOCAL_SHA  $LOCAL_SIZE  $TAR"

echo "== scp =="
scp -o BatchMode=yes "$TAR" "$X99:/tmp/$(basename "$TAR")"

echo "== 远端解包 + 校验 =="
ssh -o BatchMode=yes "$X99" bash -s <<EOF
set -euo pipefail
T=/tmp/$(basename "$TAR")
echo "远端 sha256:"; sha256sum "\$T"
RS=\$(sha256sum "\$T" | cut -d' ' -f1)
[ "\$RS" = "$LOCAL_SHA" ] || { echo "SHA 不一致，停"; exit 1; }
mkdir -p $TARGET/bundle
tar xzf "\$T" -C $TARGET/bundle
chmod +x $TARGET/bundle/bin/*
echo "--- 落地清单 ---"
find $TARGET/bundle -maxdepth 1 -type d
echo "--- 二进制 ---"
ls -l $TARGET/bundle/bin/
echo "--- 文件数 ---"; find $TARGET/bundle -type f | wc -l
EOF
echo "== 完成 =="
