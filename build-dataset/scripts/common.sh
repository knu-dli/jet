# Shared part of the data scripts: the model settings and the two stages they all run.
# Sourced with the model name: source scripts/common.sh <qwen3|ministral>
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PY:-python}
PORT=${PORT:-8091}
GENERATOR=${GENERATOR:-Qwen/Qwen3.8-27B}
export PYTHONPATH=$PWD/../third_party/SWE-QA-Pro/eval

if [ "$1" = qwen3 ]; then
    KEY=qwen3-8b-nothink
    MODEL=Qwen/Qwen3-8B
    TOKENIZER=$MODEL
    MODEL_ARGS=(--tool-call-parser hermes --default-chat-template-kwargs '{"enable_thinking": false}')
elif [ "$1" = ministral ]; then
    KEY=ministral-8b
    MODEL=mistralai/Ministral-3-8B-Instruct-2512-BF16
    TOKENIZER=$PWD/../train/out/ministral-8b-tokenizer
    MODEL_ARGS=(--tool-call-parser mistral --limit-mm-per-prompt '{"image":0}' --tokenizer-mode hf --tokenizer "$TOKENIZER")
    if [ ! -d "$TOKENIZER" ]; then
        (cd ../train && $PY -m mistral_tokenizer)
    fi
else
    echo "the model is qwen3 or ministral" >&2
    exit 1
fi
ADAPTER=""
TRAJ=()
# NUM_SAMPLES 를 주면 앞의 몇 문항·궤적만 (빠르게 해 보기). 주지 않으면 전체
SAMPLES=()
if [ -n "${NUM_SAMPLES:-}" ]; then
    SAMPLES=(--num-samples $NUM_SAMPLES)
fi

# 중간에 실패해도 띄운 vLLM 서버는 내린다
SERVER=""
trap '[ -n "$SERVER" ] && kill $SERVER 2>/dev/null || true' EXIT

stage() {
    echo "[$(date +%H:%M:%S)] $*"
}

start_server() {   # start_server <log> <vllm serve arguments...>
    stage "starting vLLM ($2), log: $1"
    vllm serve "${@:2}" --port $PORT > "$1" 2>&1 &
    SERVER=$!
    # 서버가 뜰 때까지 기다린다 (뜨다가 죽으면 멈춘다)
    until curl -s localhost:$PORT/v1/models | grep -q id; do
        kill -0 $SERVER
        sleep 10
    done
}

stop_server() {
    kill $SERVER
    SERVER=""
    sleep 10
}

# Forced answers at every step, the judge, and the golden steps of run $RUN (with $ADAPTER on the agent if set)
label_golden_steps() {
    local served=$MODEL
    local lora=()
    if [ -n "$ADAPTER" ]; then
        served=adapter
        local rank
        rank=$($PY -c "import json; print(json.load(open('$ADAPTER/adapter_config.json'))['r'])")
        lora=(--enable-lora --max-lora-rank $rank --lora-modules adapter="$ADAPTER")
    fi
    mkdir -p golden_step/out/$RUN
    start_server golden_step/out/$RUN/serve.log $MODEL --max-model-len 32768 --enable-auto-tool-choice "${MODEL_ARGS[@]}" "${lora[@]}"
    stage "forced answers at every step"
    # 한 궤적의 요청을 8개씩 같이 보내야 vLLM prefix cache 에 맞는다
    $PY -m golden_step.answer -r $RUN "${TRAJ[@]}" --base-url http://localhost:$PORT/v1 --model $served --max-workers 8 --inner-workers 8 "${SAMPLES[@]}"
    stop_server
    stage "judging the forced answers (GPT-6 Luna)"
    $PY -m golden_step.judge -r $RUN "${TRAJ[@]}" "${SAMPLES[@]}"
    stage "golden steps"
    $PY -m golden_step.label -r $RUN
}

# Rationales and the training data of JET (with the rationale) and Truncated SFT (without it)
build_training_data() {
    mkdir -p rationale/out/$RUN
    start_server rationale/out/$RUN/serve_generator.log $GENERATOR --max-model-len 40960 --max-num-seqs 256
    stage "rationales"
    $PY -m rationale.generate -r $RUN "${TRAJ[@]}" --base-url http://localhost:$PORT/v1 --model $GENERATOR
    stop_server
    stage "training data (JET and Truncated SFT)"
    $PY -m rationale.assemble -r $RUN "${TRAJ[@]}" --model "$TOKENIZER"
    $PY -m rationale.assemble -r $RUN "${TRAJ[@]}" --model "$TOKENIZER" --no-rationale
    echo "done: rationale/out/$RUN/datasets/{JET,Truncated_SFT}/train.jsonl"
}
