set -e  # exit on error

# Base directories containing the symlinks
dirs=(
  "datasets/preprocessed/GVLM_CD/A"
  "datasets/preprocessed/GVLM_CD/B"
)

# Image names (without the .png extension)
names=(
  "Askja_Iceland"
  "Kaikoura_New Zealand"
  "Taitung_China"
)

for dir in "${dirs[@]}"; do
  for name in "${names[@]}"; do
    file="$dir/${name}.png"
    if [[ -L "$file" ]]; then
      echo "Unlinking symlink: $file"
      rm "$file"
    elif [[ -e "$file" ]]; then
      echo "Found regular file, not a symlink—skipping: $file"
    else
      echo "Not found: $file"
    fi
  done
done