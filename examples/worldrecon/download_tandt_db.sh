#!/bin/bash
# download_tandt_db.sh — fetch & extract the Tanks&Temples + Deep Blending
# dataset (standard 3DGS benchmark) into examples/worldrecon/tandt_db/
#
#   bash download_tandt_db.sh
#
#   If the download is slow, enable a proxy first, e.g.:
#     export https_proxy=http://127.0.0.1:7890
set -e
cd "$(dirname "$0")"

URL="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
ZIP=tandt_db.zip
DST=tandt_db

echo "[1/3] download (~1.2GB, resumable)"
curl -L -C - -o "$ZIP" "$URL"

echo "[2/3] verify archive"
unzip -t "$ZIP" > /dev/null && echo "zip OK"

echo "[3/3] extract -> $DST/"
unzip -o -q "$ZIP" -d "$DST"

echo "done. scenes:"
find "$DST" -maxdepth 2 -type d | sort
# Expected: tandt_db/tandt/{train,truck}, tandt_db/db/{drjohnson,playroom},
# each with images/ + sparse/0/{cameras,images,points3D}.bin
