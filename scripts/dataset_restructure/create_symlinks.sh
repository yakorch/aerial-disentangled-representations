#!/usr/bin/env bash

set -euo pipefail

SRC_DIR="../data"
DEST_DIR="datasets/original"

mkdir -p "$DEST_DIR"

for dir in "$SRC_DIR"/*/; do
  [ -d "$dir" ] || continue

  base=$(basename "$dir")

  ln -sfn "$dir" "$DEST_DIR/$base"
  echo "Linked: $DEST_DIR/$base -> $dir"
done
