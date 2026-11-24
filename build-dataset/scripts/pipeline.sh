#!/usr/bin/env bash
# Builds the training data of JET and Truncated SFT. "self" uses the agent's own rollouts on the RL questions,
# "expert" the Expert SFT trajectories.
# Usage: bash scripts/pipeline.sh <qwen3|ministral> <self|expert>   (NUM_SAMPLES=1 for a quick try)
set -euo pipefail
source "$(dirname "$0")/common.sh" "$1"

if [ "$2" = expert ]; then
    RUN=$KEY
else
    RUN=$KEY-self
    TRAJ=(--trajectories $PWD/../evaluate-agent/out/$RUN/sweqapro-rl/results)
    stage "rollouts on the RL questions"
    (cd ../evaluate-agent && $PY -m run_agent -r $RUN -b sweqapro-rl -m $KEY --max-workers 32 "${SAMPLES[@]}")
fi
label_golden_steps
build_training_data
