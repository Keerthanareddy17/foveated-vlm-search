#!/usr/bin/env bash
# Step 10: `make record-demo`. Creates .venv-record (Playwright + headless Chromium, local disk) if missing, then
# records results/demo/replay.html to docs/assets/demo.{gif,mp4} with scripts/record_demo.py (needs ffmpeg).
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=.venv-record
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-/content/fvs_local/ms-playwright}"
if ! "$VENV/bin/python" -c "import playwright" 2>/dev/null; then
  uv venv -q --python 3.12 "$VENV"
  uv pip install -q --python "$VENV/bin/python" playwright==1.55.0 pillow==13.0.0
fi
"$VENV/bin/playwright" install chromium >/dev/null
command -v ffmpeg >/dev/null || { echo "ffmpeg not found"; exit 1; }
"$VENV/bin/python" scripts/record_demo.py "$@"
