#!/bin/bash

set -e  # Exit on error

src_dir="datasets/original/GVLM_CD"
dest_dir="datasets/preprocessed/GVLM_CD"
mkdir -p "${dest_dir}"
dest_A="${dest_dir}/A"
dest_B="${dest_dir}/B"

mkdir -p "$dest_A" "$dest_B"

index=1

for location in "$src_dir"/*/; do
    im1="${location}im1.png"
    im2="${location}im2.png"

    # Skip non-location dirs (like A or B if re-run)
    [[ ! -f "$im1" ]] && continue

    # Format index with leading zeros
    idx=$(printf "%04d" "$index")

    rel_im1=$(python -c "import os.path; print(os.path.relpath('$im1', '$dest_A'))")
    rel_im2=$(python -c "import os.path; print(os.path.relpath('$im2', '$dest_B'))")

    ln -s "$rel_im1" "${dest_A}/image_${idx}.png"
    ln -s "$rel_im2" "${dest_B}/image_${idx}.png"

    ((index++))
done
