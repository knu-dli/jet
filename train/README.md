# train

LoRA training of the agents. SFT trains on the full Expert trajectories, Truncated SFT on the truncated trajectories
without the rationale, and JET on the truncated trajectories with the rationale and the halting weight λ = 5. All three
use the same settings: LoRA r 32 and α 64 on the linear layers, learning rate 5e-4, 4 epochs, and an effective batch
of 16 on the GPUs in `CUDA_VISIBLE_DEVICES`.

```bash
cd train
export CUDA_VISIBLE_DEVICES=<gpu>      # GPU indices, for example 0 or 0,1
DATA=../build-dataset/rationale/out/qwen3-8b-nothink-self/datasets
bash scripts/jet.sh qwen3 qwen3-self-jet $DATA/JET
bash scripts/truncated_sft.sh qwen3 qwen3-self-truncated-sft $DATA/Truncated_SFT
bash scripts/sft.sh qwen3 qwen3-sft
```

Replace `qwen3` with `ministral` for Ministral-3-8B (the scripts write its fixed tokenizer on the first run). For
Expert -> Self, add `--init-adapter <Expert JET checkpoint>`. Each epoch saves a checkpoint to
`<SFT|Truncated_SFT|JET>/out/<run name>/`.

## File structure

```
train/
├── scripts/                     # sft.sh, truncated_sft.sh, jet.sh
├── SFT/                         # SFT trainers and the conversion of the Expert trajectories
├── Truncated_SFT/               # Truncated SFT trainers
├── JET/
│   ├── halting.py               # cross-entropy with the halting weight λ on the rationale
│   └── train_qwen.py  train_mistral.py
├── templates/                   # Ministral chat template with {% generation %} markers
├── mistral_tokenizer.py
├── utils.py                     # shared arguments, LoRA, data loading
└── deepspeed_zero2.yaml
```
