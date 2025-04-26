#!/bin/bash
set -e

src_dir="datasets/original/GVLM_CD"
dest_A="datasets/preprocessed/GVLM_CD/A"
dest_B="datasets/preprocessed/GVLM_CD/B"

mkdir -p "$dest_A" "$dest_B"

for location in "$src_dir"/*/; do
  # basename yields e.g. "A Luoi_Vietnam"
  dir_name=$(basename "$location")
  im1="$location/im1.png"
  im2="$location/im2.png"

  # skip folders without im1
  [[ ! -f "$im1" ]] && continue

  # resolve to absolute paths
  abs1=$(readlink -f "$im1")
  abs2=$(readlink -f "$im2")

  # create absolute symlinks
  ln -sf "$abs1" "$dest_A/${dir_name}.png"
  ln -sf "$abs2" "$dest_B/${dir_name}.png"
done