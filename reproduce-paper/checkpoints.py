"""Validation score and tool calls of every checkpoint of the trained rows of Table 2 (* marks the chosen one)."""

import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
# evaluate-agent 의 score·models 와 저자 코드 sweqapro 를 import 하려고
sys.path[:0] = [str(CODE_DIR / "evaluate-agent"), str(CODE_DIR / "third_party" / "SWE-QA-Pro" / "eval")]

from score.metrics.judge_score import load_scores, score_dir  # noqa: E402
from score.metrics.results import find_run_dir, format_number, load_results  # noqa: E402
from score.metrics.tool_calls import count_calls  # noqa: E402

OUT = HERE / "out" / "checkpoints.txt"
BENCHMARK = "valid"
EPOCHS = [1, 2, 3, 4]
# 학습 전 agent: (모델, 평가 run id)
BASE_RUNS = [("Qwen3-8B", "qwen3-8b-nothink"), ("Ministral-3-8B", "ministral-8b")]
# 표 2 의 학습한 행: (모델, 학습 데이터, 방법, epoch 을 뗀 평가 run id, 표 2 에 쓴 epoch)
ROWS = [
    ("Qwen3-8B", "Self", "Trunc. SFT", "qwen3-8b-nothink-own-plain", 4),
    ("Qwen3-8B", "Self", "JET", "qwen3-8b-nothink-own-sis-w5", 1),
    ("Qwen3-8B", "Expert", "SFT", "qwen3-8b-nothink-sft", 4),
    ("Qwen3-8B", "Expert", "Trunc. SFT", "qwen3-8b-nothink-expert-plain", 4),
    ("Qwen3-8B", "Expert", "JET", "qwen3-8b-nothink-expert-sis-w5", 4),
    ("Qwen3-8B", "Expert → Self", "JET → Trunc. SFT", "qwen3-8b-nothink-expert-sis-own-plain", 3),
    ("Qwen3-8B", "Expert → Self", "JET → JET", "qwen3-8b-nothink-expert-sis-own-sis-w5", 1),
    ("Ministral-3-8B", "Self", "Trunc. SFT", "ministral-8b-own-plain", 4),
    ("Ministral-3-8B", "Self", "JET", "ministral-8b-own-sis-w5", 2),
    ("Ministral-3-8B", "Expert", "SFT", "ministral-8b-sft", 2),
    ("Ministral-3-8B", "Expert", "Trunc. SFT", "ministral-8b-expert-plain", 4),
    ("Ministral-3-8B", "Expert", "JET", "ministral-8b-expert-sis-w5", 3),
    ("Ministral-3-8B", "Expert → Self", "JET → Trunc. SFT", "ministral-8b-expert-sis-own-plain", 4),
    ("Ministral-3-8B", "Expert → Self", "JET → JET", "ministral-8b-expert-sis-own-sis-w5", 3),
]
# run id 가 "<접두어>-e<epoch>" 꼴이 아닌 평가: Ministral SFT 의 epoch 4 는 epoch 꼬리표 없이 돌렸다 (checkpoint-252)
RUN_IDS = {("ministral-8b-sft", 4): "ministral-8b-sft"}


def mean_score_and_calls(run_id: str) -> tuple[float | None, float | None]:
    """한 run 의 validation 평균 점수와 평균 도구 호출. run 이나 채점이 없으면 None."""
    run_dir = find_run_dir(run_id)
    if run_dir is None:
        return None, None
    # Score: 문항마다 다섯 항목 점수의 합 (score/metrics/judge_score.py)
    scores = load_scores(score_dir(run_dir, BENCHMARK))
    # Calls: 문항마다 궤적의 도구 호출 수 (score/metrics/tool_calls.py)
    calls = []
    for result in load_results(run_dir / BENCHMARK):
        calls.append(count_calls(result))
    if not scores or not calls:
        return None, None
    return statistics.mean(scores), statistics.mean(calls)


def format_cell(score: float | None, calls: float | None, selected: bool) -> str:
    """'점수 / 호출' 한 칸. 표 2 에 쓴 epoch 이면 뒤에 * 를 붙인다."""
    mark = " "
    if selected:
        mark = "*"
    return f"{format_number(score, 2):>7} /{format_number(calls, 2):>6}{mark}"


def main() -> None:
    header = f"{'model / data / method':50s}"
    for epoch in EPOCHS:
        header += f"{'epoch ' + str(epoch):^18}"
    lines = [f"validation ({BENCHMARK}), score / calls, * = epoch used in Table 2", header]
    for model, run_id in BASE_RUNS:
        score, calls = mean_score_and_calls(run_id)
        lines.append(f"{model:16s}{'':15s}{'Base':19s}" + format_cell(score, calls, False))
    for model, group, method, prefix, selected_epoch in ROWS:
        line = f"{model:16s}{group:15s}{method:19s}"
        for epoch in EPOCHS:
            run_id = RUN_IDS.get((prefix, epoch), f"{prefix}-e{epoch}")
            score, calls = mean_score_and_calls(run_id)
            line += format_cell(score, calls, epoch == selected_epoch) + "  "
        lines.append(line)
    print("\n".join(lines))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
