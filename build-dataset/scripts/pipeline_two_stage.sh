#!/usr/bin/env bash
# Builds the Expert -> Self data: the Expert JET agent answers the RL questions, and its rollouts and forced answers
# go through the same steps as pipeline.sh.
# Usage: bash scripts/pipeline_two_stage.sh <qwen3|ministral> <Expert JET checkpoint>
set -euo pipefail
source "$(dirname "$0")/common.sh" "$1"

ADAPTER=$(realpath "$2")
RUN=$KEY-expert-to-self
TRAJ=(--trajectories $PWD/../evaluate-agent/out/$RUN/sweqapro-rl/results)
stage "rollouts of the Expert JET agent on the RL questions"
(cd ../evaluate-agent && $PY -m run_agent -r $RUN -b sweqapro-rl -m $KEY --adapter "$ADAPTER" --max-workers 32 "${SAMPLES[@]}")
label_golden_steps
build_training_data
