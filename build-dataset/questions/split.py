"""Splits the RL questions into training and validation by repository, and excludes the questions whose
repository is in a test benchmark.
"""

import argparse
import json
import random
from pathlib import Path

from questions.load import Question, load_questions

HERE = Path(__file__).resolve().parent
THIRD_PARTY = HERE.parents[1] / "third_party"
BENCHMARK_REPOS = [THIRD_PARTY / "SWE-QA-Pro" / "eval" / "repos", THIRD_PARTY / "SWE-QA-Bench" / "datas" / "repos"]


def split_questions(questions: list[Question], num_validation: int, seed: int) -> tuple[list[int], list[int]]:
    """저장소 단위로 섞어 앞에서부터 validation 문항을 num_validation 개 채운다. 나머지가 학습용."""
    by_repo: dict[str, list[int]] = {}
    for question in questions:
        by_repo.setdefault(question.repo_name, []).append(question.index)
    repos = sorted(by_repo)
    random.Random(seed).shuffle(repos)
    validation = []
    for repo in repos:
        if len(validation) >= num_validation:
            break
        validation.extend(by_repo[repo])
    validation_set = set(validation)
    train = [q.index for q in questions if q.index not in validation_set]
    return sorted(validation), sorted(train)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the RL questions into train and validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(HERE / "out" / "split.json"),
        help="where to write the split",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(HERE / "out" / "repos-rl"),
        help="cloned repositories; questions whose repository is missing are left out",
    )
    parser.add_argument(
        "--benchmark-repos",
        type=str,
        nargs="+",
        default=[str(path) for path in BENCHMARK_REPOS],
        help="repository folders of the evaluation benchmarks; questions on a repository of the same name are excluded",
    )
    parser.add_argument("--num-validation", type=int, default=60, help="validation questions")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser()
    benchmark_names = set()
    for folder in args.benchmark_repos:
        for path in Path(folder).expanduser().iterdir():
            benchmark_names.add(path.name.lower())
    questions, excluded = [], []
    for question in load_questions():
        if not (repo_root / question.repo_dir).exists():
            continue
        if question.repo_name.split("/")[-1].lower() in benchmark_names:
            excluded.append(question.index)
            continue
        questions.append(question)
    validation, train = split_questions(questions, args.num_validation, args.seed)
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split = {"seed": args.seed, "validation": validation, "train": train, "excluded": sorted(excluded)}
    out_path.write_text(json.dumps(split, indent=2))
    print(
        f"validation {len(validation)}, train {len(train)}, excluded (benchmark repository) {len(excluded)} -> {out_path}"
    )


if __name__ == "__main__":
    main()
