#!/usr/bin/env bash
# Budget Tracker baseline on SWE-QA-Pro and SWE-QA, then the judges. Add --adapter <checkpoint> for SFT + Budget Tracker.
# Usage: bash scripts/budget_tracker.sh <run-id> <qwen3|ministral> [run arguments]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONPATH=$PWD/../third_party/SWE-QA-Pro/eval
RUN=$1
if [ "$2" = qwen3 ]; then
    KEY=qwen3-8b-nothink
elif [ "$2" = ministral ]; then
    KEY=ministral-8b
else
    echo "usage: bash scripts/budget_tracker.sh <run-id> <qwen3|ministral> [run arguments]" >&2
    exit 1
fi
shift 2

for BENCHMARK in sweqapro sweqa; do
    $PY -m budget_tracker.run -r "$RUN" -b $BENCHMARK -m $KEY "$@"
done
PY=$PY bash scripts/judge.sh "$RUN" budget_tracker/out
