#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="../data"
DEST_DIR="datasets/original"

mkdir -p "$DEST_DIR"

for dir in "$SRC_DIR"/*/; do
  [ -d "$dir" ] || continue
  base=$(basename "$dir")
  # resolve to absolute path
  target=$(realpath "$dir")
  link="$DEST_DIR/$base"
  ln -sfn "$target" "$link"
  echo "Linked: $link → $target"
done
