#!/usr/bin/env bash
# Autorater baseline (GPT-6 Luna as the rater) on SWE-QA-Pro and SWE-QA, then the judges. Add --adapter <checkpoint>
# for SFT + Autorater.
# Usage: bash scripts/autorater.sh <run-id> <qwen3|ministral> [run arguments]
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
    echo "usage: bash scripts/autorater.sh <run-id> <qwen3|ministral> [run arguments]" >&2
    exit 1
fi
shift 2

for BENCHMARK in sweqapro sweqa; do
    $PY -m autorater.run -r "$RUN" -b $BENCHMARK -m $KEY --rater-openai "$@"
done
PY=$PY bash scripts/judge.sh "$RUN" autorater/out
