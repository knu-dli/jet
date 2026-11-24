# evaluate-agent

Runs the agents with the SWE-QA-Pro scaffold on SWE-QA-Pro (260 questions), SWE-QA (720 questions), and the 60
validation questions, runs the Budget Tracker and Autorater baselines, and judges the answers. Every tool call runs in a
bubblewrap sandbox that sees only the repository of the question.

```bash
cd evaluate-agent
export CUDA_VISIBLE_DEVICES=<gpu>                             # GPU index, for example 0
(cd ../third_party/SWE-QA-Pro/eval && bash clone_repos.sh)    # SWE-QA-Pro repositories
(cd ../third_party/SWE-QA-Bench && bash clone_repos.sh)       # SWE-QA repositories

# Each script runs both benchmarks, judges them (GPT-6 Sol, Claude Sonnet 5.5), and prints Score and Calls
bash scripts/base.sh qwen3-base qwen3
bash scripts/budget_tracker.sh qwen3-budget-tracker qwen3
bash scripts/autorater.sh qwen3-autorater qwen3
bash scripts/jet.sh qwen3-self-jet-e1 qwen3 ../train/JET/out/qwen3-self-jet/checkpoint-25
#   sft.sh and truncated_sft.sh take a checkpoint the same way

# Checkpoint selection on the validation questions (GPT-6 Luna): the base agent and every checkpoint of one run
bash scripts/valid_base.sh qwen3-base qwen3
bash scripts/valid.sh qwen3-self-jet qwen3          # all checkpoints, or give the steps
```

The arguments are `<run-id> <qwen3|ministral>`, and the rest goes to the runner, for example `--adapter <checkpoint>`
for SFT + Budget Tracker or `--num-samples 2` for a quick try. Results go to `out/<run-id>/`, and those of the
baselines to `budget_tracker/out/` and `autorater/out/`.

## File structure

```
evaluate-agent/
├── scripts/                     # one script per method, judge.sh, valid_base.sh, valid.sh
├── run_agent.py                 # agent runs and trajectories
├── models.yaml                  # agent settings (vLLM arguments, tool-call parser, step limit 30)
├── tool_sandbox.py  tool_exec.py   # tool calls in a bubblewrap sandbox
├── budget_tracker/              # Budget Tracker baseline
├── autorater/                   # Autorater baseline and its one-shot example
├── score/
│   ├── gpt_judge/  claude_judge/   # benchmark judges
│   ├── judge.py                 # validation judge
│   ├── metrics/                 # Score, Calls, Cost, step limit
│   └── summary.py
└── dataset.py  runner.py  models.py  utils.py
```
