#!/usr/bin/env bash
# Render the HTML sources in this folder to PDFs in docs/ with headless Chrome.
# Usage: bash docs/pdf_src/build.sh
set -euo pipefail
cd "$(dirname "$0")"
CHROME=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
for doc in working viva project_report; do
  "$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=20000 \
    --run-all-compositor-stages-before-draw --print-to-pdf="../$doc.pdf" "file://$PWD/$doc.html" 2>/dev/null
  echo "wrote docs/$doc.pdf"
done
