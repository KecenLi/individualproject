#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

xelatex -interaction=nonstopmode MEETING_FOLLOWUP_REPORT.tex
xelatex -interaction=nonstopmode MEETING_FOLLOWUP_REPORT.tex

# Sync PDF to archive for convenience
mkdir -p archive/2026-02-02_paper_materials
cp -f MEETING_FOLLOWUP_REPORT.pdf archive/2026-02-02_paper_materials/MEETING_FOLLOWUP_REPORT.pdf

echo "PDF generated: $ROOT_DIR/MEETING_FOLLOWUP_REPORT.pdf"
