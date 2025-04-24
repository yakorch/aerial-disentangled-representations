#!/usr/bin/env bash
set -euo pipefail

# Prune preprocessed BANDON so that for each split (train, val), only images that appear in every t* folder are kept.

DEST_BASE="datasets/preprocessed/BANDON"

for SPLIT in train val; do
  SPLIT_DIR="${DEST_BASE}/${SPLIT}"
  echo "Processing split: ${SPLIT_DIR}"

  # collect all t* subdirectories under the split
  TS_DIRS=()
  for d in "${SPLIT_DIR}"/t*; do
    [ -d "$d" ] || continue
    TS_DIRS+=("$d")
  done

  n_ts=${#TS_DIRS[@]}
  if [ "$n_ts" -eq 0 ]; then
    echo "  No t* directories found in ${SPLIT_DIR}, skipping."
    continue
  fi

  # count how many times each filename appears across all t* dirs
  declare -A counts=()
  for TS_DIR in "${TS_DIRS[@]}"; do
    for f in "$TS_DIR"/*; do
      name=$(basename "$f")
      counts["$name"]=$((counts["$name"] + 1))
    done
  done

  # remove any file whose count != number of t* dirs
  for TS_DIR in "${TS_DIRS[@]}"; do
    for f in "$TS_DIR"/*; do
      name=$(basename "$f")
      if [ "${counts[$name]}" -ne "$n_ts" ]; then
        echo "  Removing unmatched: ${f}"
        rm -f "${f}"
      fi
    done
  done

  echo "  Done split: ${SPLIT}"
  echo
done

echo "Pruning complete for train and val."
