"""Table 4: hyperparameters, read from the code and from the arguments saved by each run."""

import json
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
OUT = HERE / "out" / "table4.txt"
ANSWER_CODE = CODE_DIR / "build-dataset" / "golden_step" / "answer.py"
RATIONALE_OUT = CODE_DIR / "build-dataset" / "rationale" / "out"
TRAIN_DIR = CODE_DIR / "train"
TRAIN_CODE = TRAIN_DIR / "utils.py"
MODELS_YAML = CODE_DIR / "evaluate-agent" / "models.yaml"
# 표 2 의 학습한 행 14개의 학습 run 폴더 (train_args.json 과 checkpoint-<step> 이 있다)
TRAIN_RUNS = [
    TRAIN_DIR / "SFT" / "out" / "qwen3-8b-nothink-sft-rerun",
    TRAIN_DIR / "SFT" / "out" / "ministral-8b-sft-rerun",
    TRAIN_DIR / "Truncated_SFT" / "out" / "qwen3-8b-nothink-own-plain",
    TRAIN_DIR / "Truncated_SFT" / "out" / "qwen3-8b-nothink-expert-plain",
    TRAIN_DIR / "Truncated_SFT" / "out" / "qwen3-8b-nothink-expert-sis-own-plain",
    TRAIN_DIR / "Truncated_SFT" / "out" / "ministral-8b-own-plain",
    TRAIN_DIR / "Truncated_SFT" / "out" / "ministral-8b-expert-plain",
    TRAIN_DIR / "Truncated_SFT" / "out" / "ministral-8b-expert-sis-own-plain",
    TRAIN_DIR / "JET" / "out" / "qwen3-8b-nothink-own-sis-w5",
    TRAIN_DIR / "JET" / "out" / "qwen3-8b-nothink-expert-sis-w5",
    TRAIN_DIR / "JET" / "out" / "qwen3-8b-nothink-expert-sis-own-sis-w5",
    TRAIN_DIR / "JET" / "out" / "ministral-8b-own-sis-w5",
    TRAIN_DIR / "JET" / "out" / "ministral-8b-expert-sis-w5",
    TRAIN_DIR / "JET" / "out" / "ministral-8b-expert-sis-own-sis-w5",
]
# 평가에 쓴 models.yaml 키
EVAL_MODELS = ["qwen3-8b-nothink", "ministral-8b"]


def read_constant(path: Path, name: str) -> str:
    """파이썬 파일의 `NAME = 값` 줄에서 값을 읽는다."""
    match = re.search(rf"^{name} = (.+)$", path.read_text(), flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"No constant {name} in {path}")
    return match.group(1)


def read_keyword(path: Path, name: str) -> str:
    """파이썬 파일의 `name=값,` 인자에서 값을 읽는다."""
    match = re.search(rf"\b{name}=([^,\n]+),", path.read_text())
    if match is None:
        raise ValueError(f"No keyword {name} in {path}")
    return match.group(1).strip('"')


def shared_value(values: dict[str, object], name: str) -> object:
    """run 마다 읽은 값이 모두 같으면 그 값. 다르면 어느 run 이 다른지 보이며 멈춘다."""
    distinct = set(values.values())
    if len(distinct) != 1:
        raise ValueError(f"{name} differs across runs: {values}")
    return distinct.pop()


def effective_batch_size(run_dir: Path, dataset_dir: str) -> int:
    """학습 행 수 ÷ epoch 당 step 수 (올림 전이라 반올림한다). GPU 수 × gradient accumulation 과 같다."""
    num_rows = 0
    # dataset_dir 는 학습을 돌린 train/ 폴더 기준의 경로다. SFT 기록은 폴더 이름을 바꾸기 전의 sft/ 로 남아 있다
    if dataset_dir.startswith("sft/"):
        dataset_dir = "SFT/" + dataset_dir.removeprefix("sft/")
    # 학습 데이터 폴더도 datasets/teacher·no-teacher 에서 datasets/JET·Truncated_SFT 로 바뀌었다
    dataset_dir = dataset_dir.replace("datasets/no-teacher", "datasets/Truncated_SFT").replace("datasets/teacher", "datasets/JET")
    for line in (TRAIN_DIR / dataset_dir / "train.jsonl").read_text().splitlines():
        if line.strip():
            num_rows += 1
    # checkpoint 는 epoch 마다 하나라서 가장 작은 번호가 epoch 당 step 수다
    steps = []
    for path in run_dir.glob("checkpoint-*"):
        steps.append(int(path.name.split("-")[1]))
    return round(num_rows / min(steps))


def main() -> None:
    rows = []

    # (1) golden step 라벨링
    rows.append(("Intermediate answers per step K", read_constant(ANSWER_CODE, "NUM_ANSWERS"), "golden_step/answer.py"))
    rows.append(("Temperature of intermediate answers", read_constant(ANSWER_CODE, "TEMPERATURE"), "golden_step/answer.py"))
    max_tokens = {}
    for path in sorted(RATIONALE_OUT.glob("*/rationale_args.json")):
        max_tokens[path.parent.name] = json.loads(path.read_text())["max_tokens"]
    rows.append(("Maximum tokens of a rationale", shared_value(max_tokens, "max_tokens"), f"rationale_args.json of {len(max_tokens)} runs"))

    # (2) 학습: run 마다 남긴 train_args.json
    train_args = {}
    batch_sizes = {}
    for run_dir in TRAIN_RUNS:
        args = json.loads((run_dir / "train_args.json").read_text())
        train_args[run_dir.name] = args
        batch_sizes[run_dir.name] = effective_batch_size(run_dir, args["dataset_dir"])
    source = f"train_args.json of {len(TRAIN_RUNS)} runs"
    # halting weight 는 JET run 에만 있다 (SFT 와 Truncated SFT 는 None). 옵션 이름을 바꾸기 전의 기록은 stop_weight 다
    weights = {}
    for name, args in train_args.items():
        weight = args.get("halting_weight", args.get("stop_weight"))
        if weight is not None:
            weights[name] = weight
    rows.append(("Halting weight λ", shared_value(weights, "halting_weight"), f"train_args.json of {len(weights)} JET runs"))
    # (표의 이름, train_args.json 의 키)
    keys = [
        ("LoRA rank", "lora_rank"),
        ("LoRA α", "lora_alpha"),
        ("LoRA dropout", "lora_dropout"),
        ("Learning rate", "learning_rate"),
        ("Warmup ratio", "warmup_ratio"),
        ("Weight decay", "weight_decay"),
        ("Epochs", "num_train_epochs"),
        ("Maximum length", "max_length"),
    ]
    for label, key in keys:
        values = {}
        for name, args in train_args.items():
            values[name] = args[key]
        rows.append((label, shared_value(values, key), source))
    rows.append(("Effective batch size", shared_value(batch_sizes, "batch size"), "training rows / steps per epoch"))
    # 인자가 아니라 코드에 적힌 값. optimizer 는 지정하지 않아 SFTConfig 기본값 (AdamW) 이다
    rows.append(("Learning rate schedule", read_keyword(TRAIN_CODE, "lr_scheduler_type"), "train/utils.py"))
    rows.append(("bf16", read_keyword(TRAIN_CODE, "bf16"), "train/utils.py"))
    optimizer = "AdamW (SFTConfig default)"
    if re.search(r"\boptim=", TRAIN_CODE.read_text()):
        optimizer = read_keyword(TRAIN_CODE, "optim")
    rows.append(("Optimizer", optimizer, "train/utils.py"))

    # (3) 평가: 모델 설정에 없는 값은 defaults 를 따른다
    config = yaml.safe_load(MODELS_YAML.read_text())
    # (표의 이름, models.yaml 의 키)
    keys = [("Temperature", "temperature"), ("Context window", "max_context_length"), ("Step limit", "max_iterations")]
    for label, key in keys:
        values = {}
        for model in EVAL_MODELS:
            values[model] = config["models"][model].get(key, config["defaults"][key])
        rows.append((f"Evaluation: {label}", shared_value(values, key), "evaluate-agent/models.yaml"))

    lines = [f"{'hyperparameter':38s}{'value':28s}source"]
    for label, value, source in rows:
        lines.append(f"{label:38s}{str(value):28s}{source}")
    print("\n".join(lines))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
