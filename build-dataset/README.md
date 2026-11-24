# build-dataset

Builds the training data of JET. The agent is forced to answer after every step of each training trajectory, a judge
scores the answers, and the earliest step with the highest mean score becomes the golden step. The rationale generator
then writes why the information gathered up to that step is sufficient, and the trajectory is truncated there.

The trajectories are either the 1,000 Expert trajectories of
[SWE-QA-Pro-SFT-Trajectories](https://huggingface.co/datasets/TIGER-Lab/SWE-QA-Pro-SFT-Trajectories) or the agent's own
rollouts on the SWE-QA-Pro RL questions (Self). Each script runs every step on one GPU and starts the vLLM servers.

```bash
cd build-dataset
export CUDA_VISIBLE_DEVICES=<gpu>                              # GPU index, for example 0
python -m questions.clone                                      # repositories of the RL questions (Self only)

bash scripts/pipeline.sh qwen3 self     # Self data
bash scripts/pipeline.sh qwen3 expert   # Expert data
bash scripts/pipeline_two_stage.sh qwen3 <Expert JET checkpoint>   # Expert -> Self data
```

Replace `qwen3` with `ministral` for Ministral-3-8B, and set `NUM_SAMPLES=1` to try the whole pipeline on one question
first. The data is written to
`rationale/out/<run-id>/datasets/JET/train.jsonl` and, without the rationale (Truncated SFT),
`rationale/out/<run-id>/datasets/Truncated_SFT/train.jsonl`. The run id is `qwen3-8b-nothink-self` for Self,
`qwen3-8b-nothink` for Expert, and `qwen3-8b-nothink-expert-to-self` for Expert -> Self.

## File structure

```
build-dataset/
├── scripts/
│   ├── pipeline.sh              # Self or Expert data
│   ├── pipeline_two_stage.sh    # Expert -> Self data
│   ├── benchmark_golden_step.sh # golden steps of the base agent on SWE-QA-Pro (Figure 2)
│   └── common.sh                # model settings and the steps the scripts share
├── questions/                   # SWE-QA-Pro RL questions: load, clone repositories, train/validation split
├── golden_step/                 # forced answers, judge (GPT-6 Luna), golden steps
├── rationale/                   # rationales (Qwen3.8-27B) and the training data
├── check_split.py               # checks that training, validation, and test questions do not overlap
└── dataset.py  runner.py  utils.py
```
