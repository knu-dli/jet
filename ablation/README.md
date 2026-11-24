# ablation

The golden step ablation (Table 3). In the Self setting, each training trajectory is truncated at a fixed ratio of its
length instead of the golden step. Everything else is the same as JET.

```bash
cd ablation
export CUDA_VISIBLE_DEVICES=<gpu>                                     # GPU index, for example 0
mkdir -p position/out
PORT=8102 bash position/own.sh 1/2                                    # labels, rationales, training, validation runs
bash position/benchmark.sh sweqapro 0:ministral-8b-own-1of2:100      # test run of the chosen checkpoint
cd ../evaluate-agent && python -m score.gpt_judge.sweqapro -o ../ablation/position/out -r ministral-8b-own-1of2
```

## File structure

```
ablation/position/
├── label.py                     # halting step at a fixed ratio of the trajectory length
├── own.sh  own_qwen.sh          # one ratio end to end, for Ministral and Qwen3
└── benchmark.sh  benchmark_qwen.sh   # test runs of the chosen checkpoints
```
