"""Labels the halting step at a fixed ratio of the trajectory length instead of the golden step (Table 3)."""

import argparse
import json
import statistics
from fractions import Fraction
from pathlib import Path


def select_answer(step: dict) -> str | None:
    best_answer = None
    best_total = None
    for answer in step["answers"]:
        if answer["answer"] is None or answer["score"] is None:
            continue
        total = sum(answer["score"].values())
        if best_total is None or total > best_total:
            best_answer = answer["answer"]
            best_total = total
    return best_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label every SFT trajectory with a halting step at a fixed fraction of its length",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "out"),
        help="output dir",
    )
    parser.add_argument(
        "-r",
        "--run-id",
        type=str,
        required=True,
        help="run id of build-dataset/golden_step (answers scored by judge.py)",
    )
    parser.add_argument(
        "--golden-step-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "build-dataset" / "golden_step" / "out"),
        help="output dir of build-dataset/golden_step",
    )
    parser.add_argument(
        "--fraction",
        type=str,
        required=True,
        help="halting step as a fraction of n, e.g. 1/3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fraction = Fraction(args.fraction)
    if not 0 <= fraction <= 1:
        raise ValueError(f"--fraction must be in [0, 1], got {args.fraction}")

    answer_dir = Path(args.golden_step_dir).expanduser() / args.run_id / "answers"
    run_id = f"{args.run_id}-{fraction.numerator}of{fraction.denominator}"
    out_dir = Path(args.output_dir).expanduser() / run_id / "answers"
    out_dir.mkdir(parents=True, exist_ok=True)

    halting_steps = []
    num_truncated, num_without_answer = 0, 0
    for path in sorted(answer_dir.glob("*.json")):
        run = json.loads(path.read_text())
        # steps 는 지점 0..n
        n = len(run["steps"]) - 1
        halting_step = round(float(fraction) * n)
        run["halting_step"] = halting_step
        run["halting_answer"] = select_answer(run["steps"][halting_step])
        if run["halting_answer"] is None:
            num_without_answer += 1
        halting_steps.append(halting_step)
        if halting_step < n:
            num_truncated += 1
        (out_dir / path.name).write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Labeled {len(halting_steps)} trajectories with h = round({args.fraction} n) in {out_dir}")
    if halting_steps:
        print(
            f"mean h = {statistics.mean(halting_steps):.2f}, truncated = {num_truncated}, no answer at h = {num_without_answer}"
        )


if __name__ == "__main__":
    main()
