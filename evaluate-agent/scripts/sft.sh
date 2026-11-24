#!/usr/bin/env bash
# SFT agent (a checkpoint of train/scripts/sft.sh) on SWE-QA-Pro and SWE-QA, then the judges.
# Usage: bash scripts/sft.sh <run-id> <qwen3|ministral> <checkpoint> [run_agent arguments]
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
    echo "usage: bash scripts/sft.sh <run-id> <qwen3|ministral> <checkpoint> [run_agent arguments]" >&2
    exit 1
fi
CHECKPOINT=$3
shift 3

for BENCHMARK in sweqapro sweqa; do
    $PY -m run_agent -r "$RUN" -b $BENCHMARK -m $KEY --adapter "$CHECKPOINT" "$@"
done
PY=$PY bash scripts/judge.sh "$RUN"
