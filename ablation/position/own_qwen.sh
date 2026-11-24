#!/usr/bin/env bash
# Golden step ablation for Qwen3, one ratio end to end: labels, rationales, training, and validation runs.
# Usage: bash position/own_qwen.sh <ratio, e.g. 1/2>
set -euo pipefail

FRACTION=$1                                          # 예 1/2
PORT=${PORT:-8102}

HERE=$(cd "$(dirname "$0")" && pwd)                  # code/ablation/position
CODE=$(cd "$HERE/../.." && pwd)                      # code
OUT=$HERE/out
PY=${PY:-python}
export PYTHONPATH=$CODE/third_party/SWE-QA-Pro/eval

# 이 판의 이름: 1/2 → qwen3-8b-nothink-own-1of2
SOURCE_RUN=qwen3-8b-nothink-own                      # 우리 방법의 run-id. 강제 답변·점수를 여기서 가져온다
RUN=$SOURCE_RUN-${FRACTION%/*}of${FRACTION#*/}
RUN_DIR=$OUT/$RUN
TRAJECTORIES=$CODE/evaluate-agent/out/$SOURCE_RUN/sweqapro-rl/results

# 모델 (Qwen3 는 TRL 학습 템플릿을 그대로 쓴다, train/JET/train_qwen.py)
MODEL_KEY=qwen3-8b-nothink                           # evaluate-agent/models.yaml 의 키
MODEL=Qwen/Qwen3-8B
GENERATOR=Qwen/Qwen3.8-27B

mkdir -p "$RUN_DIR"
echo "$(date '+%F %T') start $RUN"

# 1) 라벨: 정지 스텝 h = round(ρ·n)
cd "$CODE/ablation"
$PY -m position.label -r $SOURCE_RUN --fraction "$FRACTION" -o "$OUT"

# 2) rationale → 학습 데이터 조립
cd "$CODE/build-dataset"
if [ ! -f "$RUN_DIR/datasets/JET/train.jsonl" ]; then
    vllm serve $GENERATOR --port $PORT --max-model-len 40960 --max-num-seqs 256 > "$RUN_DIR/serve_generator.log" 2>&1 &
    SERVER=$!
    until curl -s localhost:$PORT/v1/models | grep -q id; do     # 서버가 뜰 때까지 (뜨다 죽으면 멈춘다)
        kill -0 $SERVER
        sleep 10
    done
    $PY -m rationale.generate -r $RUN -o "$OUT" --golden-step-dir "$OUT" --trajectories "$TRAJECTORIES" \
        --base-url http://localhost:$PORT/v1 --model $GENERATOR
    kill $SERVER
    sleep 10
    $PY -m rationale.assemble -r $RUN -o "$OUT" --golden-step-dir "$OUT" --trajectories "$TRAJECTORIES" --model $MODEL
fi

# 3) 학습: 원본에서 새 어댑터, λ 5, lr 5e-4, GPU 1장 × 누적 16 = 유효 배치 16
cd "$CODE/train"
if [ ! -d "$RUN_DIR/train/checkpoint-100" ]; then
    accelerate launch --config_file deepspeed_zero2.yaml --num_processes 1 -m JET.train_qwen \
        -r $RUN -o "$RUN_DIR/train" --model $MODEL \
        --dataset-dir "$RUN_DIR/datasets/JET" --learning-rate 5e-4 --halting-weight 5 --gradient-accumulation-steps 16 \
        > "$RUN_DIR/train.log" 2>&1
fi

# 4) validation: epoch 마다 (checkpoint 25, 50, 75, 100) 실주행 + 채점
cd "$CODE/evaluate-agent"
for EPOCH in 1 2 3 4; do
    $PY -m run_agent -o "$OUT" -r $RUN-e$EPOCH -b valid -m $MODEL_KEY --adapter "$RUN_DIR/train/checkpoint-$((EPOCH * 25))"
    $PY -m score.judge -o "$OUT" -r $RUN-e$EPOCH -b valid
done
$PY -m score.summary -r $RUN-e1 $RUN-e2 $RUN-e3 $RUN-e4 -b valid
echo "$(date '+%F %T') done $RUN"
