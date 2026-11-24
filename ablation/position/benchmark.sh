#!/usr/bin/env bash
# Test runs of the chosen ablation checkpoints of Ministral on one benchmark.
# Usage: bash position/benchmark.sh <benchmark> <GPU>:<run-id>:<checkpoint step> ...
set -euo pipefail
BENCHMARK=$1
HERE=$(cd "$(dirname "$0")" && pwd)                  # code/ablation/position
CODE=$(cd "$HERE/../.." && pwd)
OUT=$HERE/out
PY=${PY:-python}
export PYTHONPATH=$CODE/third_party/SWE-QA-Pro/eval
MODEL_KEY=ministral-8b

cd "$CODE/evaluate-agent"
for ITEM in "${@:2}"; do
    GPU=${ITEM%%:*}
    REST=${ITEM#*:}
    RUN=${REST%%:*}
    CHECKPOINT=${REST#*:}
    (
        CUDA_VISIBLE_DEVICES=$GPU $PY -m run_agent -o "$OUT" -r $RUN -b $BENCHMARK -m $MODEL_KEY \
            --adapter "$OUT/$RUN/train/checkpoint-$CHECKPOINT"
        $PY -m score.judge -o "$OUT" -r $RUN -b $BENCHMARK
    ) > "$OUT/$RUN-$BENCHMARK.log" 2>&1 &
    echo "$(date '+%F %T') GPU $GPU: $RUN checkpoint-$CHECKPOINT on $BENCHMARK"
done
wait
RUNS=""
for ITEM in "${@:2}"; do
    REST=${ITEM#*:}
    RUNS="$RUNS ${REST%%:*}"
done
$PY -m score.summary -r $RUNS -b $BENCHMARK
echo "$(date '+%F %T') done"
