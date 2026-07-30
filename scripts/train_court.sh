#!/usr/bin/env bash
# Court keypoint pose training from court_train_pack.zip.
#
#   scripts/train_court.sh [path/to/court_train_pack.zip]
#
# Unzips the pack, rewrites dataset.yaml's `path:` to the absolute court/ dir,
# then runs the pose training recipe.
set -euo pipefail

ZIP="${1:-court_train_pack.zip}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [[ ! -f "$ZIP" ]]; then
  echo "error: $ZIP not found (upload the pack to $REPO first)" >&2
  exit 1
fi

unzip -o "$ZIP"

# The pack may unpack court/ at top level or one level down -- locate it.
YAML="$(find . -maxdepth 4 -path ./runs -prune -o -name dataset.yaml -print 2>/dev/null | head -1)"
[[ -n "$YAML" ]] || { echo "error: no dataset.yaml found after unzip" >&2; exit 1; }
COURT_DIR="$(cd "$(dirname "$YAML")" && pwd)"
echo "dataset.yaml: $YAML"
echo "court dir:    $COURT_DIR"

# Rewrite the top-level `path:` key to the absolute court/ dir.
python - "$YAML" "$COURT_DIR" <<'PY'
import re, sys
yaml_path, court_dir = sys.argv[1], sys.argv[2]
src = open(yaml_path).read()
new, n = re.subn(r'(?m)^path:.*$', f'path: {court_dir}', src)
if n == 0:
    new = f'path: {court_dir}\n' + src
    print('no path: key -> prepended')
else:
    print(f'rewrote path: -> {court_dir}')
open(yaml_path, 'w').write(new)
PY

MODEL="$(find . -maxdepth 4 -name 'yolo11s-court.pt' 2>/dev/null | head -1)"
[[ -n "$MODEL" ]] || { echo "error: yolo11s-court.pt not found after unzip" >&2; exit 1; }
echo "model:        $MODEL"

exec yolo pose train model="$MODEL" data="$YAML" \
  epochs=300 imgsz=640 batch=16 optimizer=AdamW lr0=0.0003 mosaic=0 \
  scale=0.3 translate=0.05 patience=50 device=0
