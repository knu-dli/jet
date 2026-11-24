#!/usr/bin/env bash
# Runs and judges each checkpoint of one training run on the validation questions (run id <run-name>-e<epoch>).
# Usage: bash scripts/valid.sh <run-name> <qwen3|ministral> [checkpoint steps, all by default]
set -euo pipefail
RUN=$1
if [ "$2" = qwen3 ]; then
    KEY=qwen3-8b-nothink
elif [ "$2" = ministral ]; then
    KEY=ministral-8b
else
    echo "usage: bash scripts/valid.sh <run-name> <qwen3|ministral> <checkpoint steps...>" >&2
    exit 1
fi
HERE=$(cd "$(dirname "$0")/.." && pwd)                       # evaluate-agent/
PY=${PY:-python}
export PYTHONPATH=$HERE/../third_party/SWE-QA-Pro/eval

cd "$HERE"
# 학습 run 은 train/JET, train/Truncated_SFT, train/SFT 중 한 곳에 있다
DIR=""
for METHOD in JET Truncated_SFT SFT; do
    if [ -d "../train/$METHOD/out/$RUN" ]; then
        DIR=../train/$METHOD/out/$RUN
    fi
done
if [ -z "$DIR" ]; then
    echo "no training run $RUN under ../train/{JET,Truncated_SFT,SFT}/out" >&2
    exit 1
fi
# checkpoint 가 <run>/train/ 아래에 있는 판
if [ -d "$DIR/train" ]; then
    DIR=$DIR/train
fi
ALL=$(ls -d "$DIR"/checkpoint-* | sed 's/.*-//' | sort -n)
LAST=$(echo "$ALL" | tail -1)
# checkpoint 번호를 주지 않으면 전부
STEPS=("${@:3}")
if [ ${#STEPS[@]} -eq 0 ]; then
    STEPS=($ALL)
fi
for ckpt in "${STEPS[@]}"; do
    epoch=$((ckpt * 4 / LAST))
    $PY -m run_agent -r "$RUN-e$epoch" -b valid -m "$KEY" --adapter "$DIR/checkpoint-$ckpt" ${RUN_AGENT_ARGS:-}
    $PY -m score.judge -r "$RUN-e$epoch" -b valid
done

# checkpoint 마다 validation 점수와 도구 호출
RUNS=()
for ckpt in "${STEPS[@]}"; do
    RUNS+=("$RUN-e$((ckpt * 4 / LAST))")
done
$PY -m score.summary -b valid -r "${RUNS[@]}"
