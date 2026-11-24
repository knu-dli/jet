#!/usr/bin/env bash
# Makes every table and figure of the paper from the finished runs (no GPU or API call).
# Usage: bash run.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
for name in figure2 table2 table3 figure4 figure5 table4 table5 checkpoints; do
    echo "== $name"
    $PY $name.py
done
