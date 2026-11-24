#!/usr/bin/env bash
# Judges one run: GPT-6 Sol on SWE-QA-Pro and Claude Sonnet 5.5 on SWE-QA (waits for the batch), then prints the scores.
# Usage: bash scripts/judge.sh <run-id> [output dir, default out]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONPATH=$PWD/../third_party/SWE-QA-Pro/eval
RUN=$1
OUT=${2:-out}

$PY -m score.gpt_judge.sweqapro -o "$OUT" -r "$RUN"
# collect 는 batch 가 끝나면 점수를 쓰고 batch id 파일을 지운다. 이미 보낸 batch 가 있으면 그것을 기다린다
BATCH=$OUT/$RUN/sweqa/judge-claude-sonnet-5-5_batch.txt
if [ ! -f "$BATCH" ]; then
    $PY -m score.claude_judge.sweqa submit -o "$OUT" -r "$RUN"
fi
while [ -f "$BATCH" ]; do
    sleep 60
    $PY -m score.claude_judge.sweqa collect -o "$OUT" -r "$RUN"
done
$PY -m score.summary -r "$RUN"
