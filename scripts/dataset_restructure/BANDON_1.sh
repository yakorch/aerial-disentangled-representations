#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="datasets/original/BANDON"
DEST_BASE="datasets/preprocessed/BANDON"

# $1 = source imgs dir, $2 = dest split dir
restructure_split() {
  local SRC_IMG_DIR=$1
  local DEST_SPLIT_DIR=$2

  echo "Re-structuring '${SRC_IMG_DIR}' → '${DEST_SPLIT_DIR}'"
  mkdir -p "${DEST_SPLIT_DIR}"

  # collect all unique time-step names (t1, t2, t3, …)
  declare -A seen_ts=()
  for LOC in "${SRC_IMG_DIR}"/*; do
    [ -d "$LOC" ] || continue
    for TS_DIR in "$LOC"/*; do
      seen_ts["$(basename "$TS_DIR")"]=1
    done
  done

  # for each time-step, mkdir and symlink
  for TS in "${!seen_ts[@]}"; do
    echo "  → grouping '${TS}'"
    mkdir -p "${DEST_SPLIT_DIR}/${TS}"

    for LOC in "${SRC_IMG_DIR}"/*; do
      [ -d "$LOC" ] || continue
      LOC_NAME=$(basename "$LOC")

      for IMG in "${LOC}/${TS}"/*; do
        IMG_NAME=$(basename "$IMG")
        REL_PATH=$(python -c \
          "import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" \
          "$IMG" "${DEST_SPLIT_DIR}/${TS}")
        ln -s "$REL_PATH" "${DEST_SPLIT_DIR}/${TS}/${LOC_NAME}_${IMG_NAME}"
      done
    done
  done

  echo "→ done '${DEST_SPLIT_DIR}'"
  echo
}

# 1) train & val
for SPLIT in train val; do
  restructure_split \
    "${SRC_BASE}/${SPLIT}/imgs" \
    "${DEST_BASE}/${SPLIT}"
done

# 2) test & test_ood (inside original/test/)
for SUBDIR in "${SRC_BASE}/test"/*; do
  [ -d "$SUBDIR" ] || continue
  NAME=$(basename "$SUBDIR")   # will be "test" or "test_ood"
  restructure_split \
    "${SUBDIR}/imgs" \
    "${DEST_BASE}/${NAME}"
done

echo "All splits processed!"
