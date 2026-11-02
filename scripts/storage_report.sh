#!/usr/bin/env bash
# Storage report: project Drive usage vs budget, local cache usage, free disk. Read-only.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO/scripts/env.sh" ] || { echo "scripts/env.sh missing: run 'make setup'" >&2; exit 1; }
source "$REPO/scripts/env.sh"
BUDGET_MB=500   # storage policy (NOTES.md): keep the project's Drive usage under ~500 MB

echo "== Drive project folder (budget ${BUDGET_MB} MB): $PROJECT_DRIVE_DIR"
if [ -d "$PROJECT_DRIVE_DIR" ]; then
  # one du per folder (du skips subfolders it already counted when given several paths)
  find "$PROJECT_DRIVE_DIR" -mindepth 1 -maxdepth 2 -type d | sort | while read -r d; do
    printf '    %-8s %s\n' "$(du -sh "$d" | cut -f1)" "${d#$PROJECT_DRIVE_DIR/}"
  done
  used_mb=$(du -sm "$PROJECT_DRIVE_DIR" | cut -f1)
  files=$(find "$PROJECT_DRIVE_DIR" -type f | wc -l)
  status=$([ "$used_mb" -le "$BUDGET_MB" ] && echo OK || echo "OVER BUDGET")
  echo "    TOTAL ${used_mb} MB in ${files} files -> $status"
else
  echo "    not mounted (run the notebook setup cell)"
fi
echo "    (Deleted files stay in the Drive trash and count against quota until you empty it in the browser.)"
echo
echo "== Local cache (wiped each session): $FVS_LOCAL_DIR"
[ -d "$FVS_LOCAL_DIR" ] && du -sh "$FVS_LOCAL_DIR"/* 2>/dev/null | sed "s#$FVS_LOCAL_DIR/#    #"
echo
echo "== Local disk /content"
df -h /content | awk 'NR==2{print "    "$4" free of "$2" ("$5" used)"}'
