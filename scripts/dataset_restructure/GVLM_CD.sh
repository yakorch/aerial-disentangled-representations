#!/bin/bash
set -e  # exit on error

src_dir="datasets/original/GVLM_CD"
dest_dir="datasets/preprocessed/GVLM_CD"
dest_A="${dest_dir}/A"
dest_B="${dest_dir}/B"

mkdir -p "$dest_A" "$dest_B"

for location in "$src_dir"/*/; do
    # get just the folder name, e.g. "location42"
    dir_name=$(basename "$location")

    im1="${location}im1.png"
    im2="${location}im2.png"

    # skip if this folder has no im1
    [[ ! -f "$im1" ]] && continue

    # create symlinks named by the directory
    ln -s "$im1" "${dest_A}/${dir_name}.png"
    ln -s "$im2" "${dest_B}/${dir_name}.png"
done
