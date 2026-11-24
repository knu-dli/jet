"""Checks that the training, validation, and test questions share no question or repository at any stage of the data."""

import importlib.util
import json
import re
from pathlib import Path

from dataset import load_sft_trajectories
from questions.load import load_questions

HERE = Path(__file__).resolve().parent
CODE = HERE.parent


def load_eval_dataset():
    """evaluate-agent/dataset.py 는 이름이 build-dataset/dataset.py 와 같아 경로로 따로 읽는다."""
    spec = importlib.util.spec_from_file_location("eval_dataset", CODE / "evaluate-agent" / "dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def repo_name(path: str) -> str:
    return path.rstrip("/").split("/")[-1].lower()


def report(name: str, shared: set) -> None:
    mark = "OK " if not shared else "!! "
    preview = sorted(shared)[:5]
    print(f"  {mark}{name}: {len(shared)}" + (f"  e.g. {preview}" if shared else ""))


def row_question(row: dict, questions: set[str]) -> str | None:
    """학습 행의 user 글에서 알려진 질문 글을 찾는다 (user 글에 질문이 그대로 들어 있다)."""
    for message in row["messages"]:
        if message["role"] != "user":
            continue
        content = norm(message["content"] if isinstance(message["content"], str) else json.dumps(message["content"]))
        for question in questions:
            if question in content:
                return question
        return None
    return None


def main() -> None:
    eval_dataset = load_eval_dataset()
    split = json.loads((HERE / "questions" / "out" / "split.json").read_text())
    train_ids = set(split["train"])
    valid_ids = set(split["validation"])
    excluded_ids = set(split["excluded"])

    rl = {q.index: q for q in load_questions()}
    rl_text = {i: norm(q.question) for i, q in rl.items()}
    rl_repo = {i: repo_name(q.repo_name) for i, q in rl.items()}
    train_text = {rl_text[i] for i in train_ids}
    valid_text = {rl_text[i] for i in valid_ids}
    train_repo = {rl_repo[i] for i in train_ids}
    valid_repo = {rl_repo[i] for i in valid_ids}

    benchmarks = {"SWE-QA-Pro": eval_dataset.load_sweqapro(), "SWE-QA": eval_dataset.load_sweqa()}
    bench_text = {}
    bench_repo = {}
    for name, items in benchmarks.items():
        bench_text[name] = {norm(item.question) for item in items}
        bench_repo[name] = {repo_name(item.repo) for item in items}
    validation_items = eval_dataset.load_validation()

    print("1. RL split (train 400 / validation 60 / excluded 2)")
    print(f"  sizes: train {len(train_ids)}, validation {len(valid_ids)}, excluded {len(excluded_ids)}, all RL {len(rl)}")
    report("train ∩ validation (ids)", train_ids & valid_ids)
    report("train ∩ validation (question text)", train_text & valid_text)
    report("train ∩ validation (repos)", train_repo & valid_repo)
    report("evaluate-agent 'valid' != split validation", {item.index for item in validation_items} ^ valid_ids)
    for name in benchmarks:
        report(f"train ∩ {name} (question text)", train_text & bench_text[name])
        report(f"validation ∩ {name} (question text)", valid_text & bench_text[name])
        report(f"train ∩ {name} (repos)", train_repo & bench_repo[name])
        report(f"validation ∩ {name} (repos)", valid_repo & bench_repo[name])

    print("2. expert trajectories (1,000 Claude SFT)")
    expert = load_sft_trajectories()
    expert_text = {norm(t.question) for t in expert}
    expert_repo = {repo_name(t.repo_path) for t in expert}
    print(f"  {len(expert)} trajectories, {len(expert_text)} distinct questions, {len(expert_repo)} repos")
    report("expert ∩ validation (question text)", expert_text & valid_text)
    report("expert ∩ validation (repos)", expert_repo & valid_repo)
    report("expert ∩ RL train (question text)", expert_text & train_text)
    for name in benchmarks:
        report(f"expert ∩ {name} (question text)", expert_text & bench_text[name])
        report(f"expert ∩ {name} (repos)", expert_repo & bench_repo[name])

    print("3. assembled training data: which questions each dataset contains")
    known = train_text | valid_text | expert_text
    for name in benchmarks:
        known |= bench_text[name]
    paths = sorted(HERE.glob("rationale/out/*/datasets/*/train.jsonl"))
    paths += sorted((CODE / "ablation").glob("*/out/*/datasets/*/train.jsonl"))
    for path in paths:
        counts = {"train": 0, "validation": 0, "expert": 0, "benchmark": 0, "unknown": 0}
        for line in path.read_text().splitlines():
            question = row_question(json.loads(line), known)
            if question is None:
                counts["unknown"] += 1
            elif question in valid_text:
                counts["validation"] += 1
            elif question in bench_text["SWE-QA-Pro"] or question in bench_text["SWE-QA"]:
                counts["benchmark"] += 1
            elif question in train_text:
                counts["train"] += 1
            else:
                counts["expert"] += 1
        bad = counts["validation"] + counts["benchmark"]
        mark = "OK " if bad == 0 else "!! "
        print(f"  {mark}{path.relative_to(CODE)}: {counts}")

    print("4. Autorater one-shot example")
    example_id = 420
    where = "train" if example_id in train_ids else ("validation" if example_id in valid_ids else "excluded/other")
    print(f"  {'OK ' if where == 'train' else '!! '}question {example_id} is in: {where}")


if __name__ == "__main__":
    main()
