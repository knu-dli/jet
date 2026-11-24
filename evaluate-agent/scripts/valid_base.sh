#!/usr/bin/env bash
# Base agent on the 60 validation questions, judged by GPT-6 Luna, to compare with the checkpoints of scripts/valid.sh.
# Usage: bash scripts/valid_base.sh <run-id> <qwen3|ministral> [run_agent arguments]
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
    echo "usage: bash scripts/valid_base.sh <run-id> <qwen3|ministral> [run_agent arguments]" >&2
    exit 1
fi
shift 2

$PY -m run_agent -r "$RUN" -b valid -m $KEY "$@"
$PY -m score.judge -r "$RUN" -b valid
$PY -m score.summary -b valid -r "$RUN"
