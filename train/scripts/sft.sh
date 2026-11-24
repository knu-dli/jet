#!/usr/bin/env bash
# SFT baseline on the full Expert trajectories, with an effective batch of 16 on the GPUs in CUDA_VISIBLE_DEVICES.
# Usage: bash scripts/sft.sh <qwen3|ministral> <run name> [training arguments]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
WHICH=$1
RUN=$2
shift 2
# GPU 수 = CUDA_VISIBLE_DEVICES 의 번호 개수. batch 16 = GPU 수 × gradient accumulation
GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
ACCUM=$((16 / GPUS))

if [ "$WHICH" = qwen3 ]; then
    MODEL_ARGS=(-m SFT.train_qwen --model Qwen/Qwen3-8B)
elif [ "$WHICH" = ministral ]; then
    MODEL_ARGS=(-m SFT.train_mistral --model mistralai/Ministral-3-8B-Instruct-2512-BF16
                --tokenizer out/ministral-8b-tokenizer --chat-template templates/ministral3_training.jinja)
else
    echo "usage: bash scripts/sft.sh <qwen3|ministral> <run name>" >&2
    exit 1
fi
if [ $((16 % GPUS)) -ne 0 ]; then
    echo "the GPU count $GPUS does not divide the batch of 16" >&2
    exit 1
fi
# Ministral 은 transformers·vLLM 이 제대로 읽는 tokenizer 폴더가 필요하다
if [ "$WHICH" = ministral ] && [ ! -d out/ministral-8b-tokenizer ]; then
    $PY -m mistral_tokenizer
fi

if [ ! -f SFT/sftdata/train.jsonl ]; then
    $PY -m SFT.convert_data
fi
$PY -m accelerate.commands.launch --config_file deepspeed_zero2.yaml --num_processes "$GPUS" "${MODEL_ARGS[@]}" \
    -r "$RUN" --dataset-dir SFT/sftdata --learning-rate 5e-4 --gradient-accumulation-steps $ACCUM "$@"
