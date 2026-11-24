# golden_step

Labels the golden step of each training trajectory. After every step i, the agent before fine-tuning is forced to
answer 3 times (temperature 0.6) with the forced-answer call of the SWE-QA-Pro scaffold. GPT-6 Luna scores each answer
with the SWE-QA-Pro judge prompt, and the golden step is the earliest step with the highest mean score. The reference
answer is the final answer of an Expert trajectory, or the ground truth answer of the question for a rollout.

`../scripts/pipeline.sh` runs the three steps. To run them one by one, start the agent with vLLM first:

```bash
cd build-dataset
# Qwen3-8B in the Self setting (run id of ../scripts/pipeline.sh; for Expert, use -r qwen3-8b-nothink without --trajectories)
ROLLOUTS=../evaluate-agent/out/qwen3-8b-nothink-self/sweqapro-rl/results
python -m golden_step.answer -r qwen3-8b-nothink-self --trajectories $ROLLOUTS --base-url http://localhost:8091/v1 --model Qwen/Qwen3-8B
python -m golden_step.judge -r qwen3-8b-nothink-self --trajectories $ROLLOUTS
python -m golden_step.label -r qwen3-8b-nothink-self
```

Without `--trajectories` the steps read the Expert trajectories. All three write to the same file per trajectory,
`out/<run-id>/answers/<trajectory id>.json`, and skip the work that is already there.

## Files

```
golden_step/
├── answer.py      # forced answers after every step
├── judge.py       # scores of the forced answers (GPT-6 Luna)
├── label.py       # golden step and the best answer at that step
└── models.py      # the record of one trajectory: steps, answers, scores, golden step
```
