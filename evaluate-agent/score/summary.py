"""Prints the Score and Calls of the given runs, for example to select a checkpoint on the validation set."""

import argparse
import statistics
from pathlib import Path

from dataset import BENCHMARKS, VALIDATION_SET
from score.metrics.judge_score import load_scores, score_dir
from score.metrics.results import find_run_dir, format_number, load_results
from score.metrics.tool_calls import count_calls


def mean_score_and_calls(run_dir: Path, benchmark: str) -> tuple[float | None, float | None]:
    """한 실행분의 평균 점수와 평균 도구 호출. 채점이나 결과가 없으면 None."""
    scores = load_scores(score_dir(run_dir, benchmark))
    calls = []
    for result in load_results(run_dir / benchmark):
        calls.append(count_calls(result))
    score = statistics.mean(scores) if scores else None
    mean_calls = statistics.mean(calls) if calls else None
    return score, mean_calls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--run-ids", type=str, nargs="+", required=True, help="run ids to show")
    parser.add_argument("-b", "--benchmarks", type=str, nargs="+", default=BENCHMARKS, choices=BENCHMARKS + [VALIDATION_SET], help="benchmarks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"{'':46s}" + "".join(f"{benchmark:^27}  " for benchmark in args.benchmarks))
    print(f"{'run id':46s}" + f"{'Score':>9}{'Calls':>9}{'n':>9}  " * len(args.benchmarks))
    for run_id in args.run_ids:
        run_dir = find_run_dir(run_id)
        if run_dir is None:
            raise ValueError(f"No run dir for {run_id}")
        line = f"{run_id:46s}"
        for benchmark in args.benchmarks:
            num_scored = len(load_scores(score_dir(run_dir, benchmark)))
            num_results = len(load_results(run_dir / benchmark))
            score, calls = mean_score_and_calls(run_dir, benchmark)
            num = f"{num_scored}/{num_results}"
            line += f"{format_number(score, 2):>9}{format_number(calls, 2):>9}{num:>9}  "
        print(line)


if __name__ == "__main__":
    main()
