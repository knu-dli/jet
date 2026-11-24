#!/usr/bin/env bash
# Labels the golden steps of the base agent's SWE-QA-Pro test trajectories (the input of Figure 2a). Run the base
# agent first with evaluate-agent/scripts/base.sh <run-id = qwen3-8b-nothink or ministral-8b>.
# Usage: bash scripts/benchmark_golden_step.sh <qwen3|ministral>
set -euo pipefail
source "$(dirname "$0")/common.sh" "$1"

RUN=$KEY-sweqapro
TRAJ=(--trajectories $PWD/../evaluate-agent/out/$KEY/sweqapro/results)
label_golden_steps
