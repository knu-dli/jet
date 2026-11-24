"""Table 5: mean score of each judge criterion for the rows of Table 2."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
# evaluate-agent 의 score 와 저자 코드 sweqapro 를 import 하려고
sys.path[:0] = [str(CODE_DIR / "evaluate-agent"), str(CODE_DIR / "third_party" / "SWE-QA-Pro" / "eval")]

from score.metrics.judge_score import mean_criteria  # noqa: E402
from score.metrics.results import format_number  # noqa: E402

EVALUATE_DIR = CODE_DIR / "evaluate-agent"
OUT = HERE / "out" / "table5.txt"

# (표의 이름, 실행분 안의 폴더, judge 점수 폴더)
BENCHMARKS = [
    ("SWE-QA-Pro", "sweqapro", "results-gpt-6-sol"),
    ("SWE-QA", "sweqa", "results-claude-sonnet-5-5"),
]
# (머리글, judge 항목)
CRITERIA = [
    ("Corr.", "correctness"),
    ("Comp.", "completeness"),
    ("Rel.", "relevance"),
    ("Clar.", "clarity"),
    ("Reas.", "reasoning"),
]
# 표의 행: (모델, 학습 데이터, 방법, 실행분 폴더). 폴더는 evaluate-agent/ 기준이고 표 2 와 같은 실행분이다.
ROWS = [
    ("Qwen3-8B", "Self", "Base", "out/qwen3-8b-nothink"),
    ("Qwen3-8B", "Self", "Budget Tracker", "budget_tracker/out/qwen3-8b-nothink-bats"),
    ("Qwen3-8B", "Self", "Autorater", "autorater/out/qwen3-8b-nothink-sufficiency"),
    ("Qwen3-8B", "Self", "Trunc. SFT", "out/qwen3-8b-nothink-own-plain-e4"),
    ("Qwen3-8B", "Self", "JET", "out/qwen3-8b-nothink-own-sis-w5-e1"),
    ("Qwen3-8B", "Expert", "SFT", "out/qwen3-8b-nothink-sft-e4"),
    ("Qwen3-8B", "Expert", "Budget Tracker", "budget_tracker/out/qwen3-8b-nothink-sft-e4-bats"),
    ("Qwen3-8B", "Expert", "Autorater", "autorater/out/qwen3-8b-nothink-sft-e4-sufficiency"),
    ("Qwen3-8B", "Expert", "Trunc. SFT", "out/qwen3-8b-nothink-expert-plain-e4"),
    ("Qwen3-8B", "Expert", "JET", "out/qwen3-8b-nothink-expert-sis-w5-e4"),
    ("Qwen3-8B", "Expert → Self", "JET → Trunc. SFT", "out/qwen3-8b-nothink-expert-sis-own-plain-e3"),
    ("Qwen3-8B", "Expert → Self", "JET → JET", "out/qwen3-8b-nothink-expert-sis-own-sis-w5-e1"),
    ("Ministral-3-8B", "Self", "Base", "out/ministral-8b"),
    ("Ministral-3-8B", "Self", "Budget Tracker", "budget_tracker/out/ministral-8b-bats"),
    ("Ministral-3-8B", "Self", "Autorater", "autorater/out/ministral-8b-sufficiency"),
    ("Ministral-3-8B", "Self", "Trunc. SFT", "out/ministral-8b-own-plain-e4"),
    ("Ministral-3-8B", "Self", "JET", "out/ministral-8b-own-sis-w5-e2"),
    ("Ministral-3-8B", "Expert", "SFT", "out/ministral-8b-sft-e2"),
    ("Ministral-3-8B", "Expert", "Budget Tracker", "budget_tracker/out/ministral-8b-sft-e2-bats"),
    ("Ministral-3-8B", "Expert", "Autorater", "autorater/out/ministral-8b-sft-e2-sufficiency"),
    ("Ministral-3-8B", "Expert", "Trunc. SFT", "out/ministral-8b-expert-plain-e4"),
    ("Ministral-3-8B", "Expert", "JET", "out/ministral-8b-expert-sis-w5-e3"),
    ("Ministral-3-8B", "Expert → Self", "JET → Trunc. SFT", "out/ministral-8b-expert-sis-own-plain-e4"),
    ("Ministral-3-8B", "Expert → Self", "JET → JET", "out/ministral-8b-expert-sis-own-sis-w5-e3"),
]


def main() -> None:
    names = [header for header, _ in CRITERIA] + ["Score"]
    width = 7 * len(names)
    lines = [f"{'':50s}" + "".join(f"{benchmark:^{width}}" for benchmark, _, _ in BENCHMARKS)]
    lines.append(f"{'model / data / method':50s}" + "".join(f"{name:>7}" for name in names + names))
    for model, group, method, run_dir in ROWS:
        line = f"{model:16s}{group:15s}{method:19s}"
        for _, folder, score_folder in BENCHMARKS:
            means = mean_criteria(EVALUATE_DIR / run_dir / folder / score_folder)
            # 다섯 항목의 평균을 더하면 표 2 의 Score (문항 총점의 평균) 다
            total = 0.0
            for _, name in CRITERIA:
                line += f"{format_number(means[name], 2):>7}"
                total += means[name]
            line += f"{format_number(total, 2):>7}"
        lines.append(line)
    print("\n".join(lines))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
