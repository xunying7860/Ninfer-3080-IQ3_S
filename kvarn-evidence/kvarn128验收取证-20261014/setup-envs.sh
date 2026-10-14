#!/bin/bash
# setup-envs.sh —— 每个臂一份独立 profile 副本（源 = device-profiles-g128.json，md5 7277b862f5c7319fc534c50bbc57a436）
set -u
ROOT=/data/kvarn/g128-验收-20261014
SRC=/data/kvarn/g128-rk4v4-test/device-profiles-g128.json
mkdir -p "$ROOT/profiles"
for a in kvarn128 rk4v4 int8; do
  cp -f "$SRC" "$ROOT/profiles/device-profiles-$a.json"
done
echo "### src: $(md5sum "$SRC")"
md5sum "$ROOT/profiles/"*.json
