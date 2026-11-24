# rationale

Builds the training data from the golden steps. The rationale generator (Qwen3.8-27B, thinking disabled) reads the
trajectory up to the golden step and the answer at that step, and writes why the information gathered so far is
sufficient to answer (at most 800 tokens). Each trajectory is then truncated at the golden step and ends with the
rationale and the answer.

`../scripts/pipeline.sh` runs both steps. To run them one by one, start the rationale generator with vLLM first:

```bash
cd build-dataset
# Qwen3-8B in the Self setting (run id of ../scripts/pipeline.sh; for Expert, use -r qwen3-8b-nothink without --trajectories)
ROLLOUTS=../evaluate-agent/out/qwen3-8b-nothink-self/sweqapro-rl/results
python -m rationale.generate -r qwen3-8b-nothink-self --trajectories $ROLLOUTS --base-url http://localhost:8091/v1
python -m rationale.assemble -r qwen3-8b-nothink-self --trajectories $ROLLOUTS --model Qwen/Qwen3-8B                  # JET
python -m rationale.assemble -r qwen3-8b-nothink-self --trajectories $ROLLOUTS --model Qwen/Qwen3-8B --no-rationale   # Truncated SFT
```

`-r` is the run id of the golden steps, `--trajectories` the rollouts of the agent (left out for the Expert
trajectories), and `--model` the tokenizer that measures the length of each row (`../train/out/ministral-8b-tokenizer`
for Ministral).

The data goes to `out/<run-id>/datasets/JET/train.jsonl` and `out/<run-id>/datasets/Truncated_SFT/train.jsonl`. For
rollouts, only the training questions of `../questions/out/split.json` are kept, and rows longer than 32,768 tokens
(with the tokenizer of `--model`) are dropped.

## Files

```
rationale/
├── generate.py    # rationales at the golden step
└── assemble.py    # truncated trajectories with the rationale and the answer
```
