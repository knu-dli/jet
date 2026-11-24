"""Figure 5: validation tool calls and score of Ministral in the Self setting for each halting weight λ."""

import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
# evaluate-agent 의 models·score 와 저자 코드 sweqapro 를 import 하려고
sys.path[:0] = [str(CODE_DIR / "evaluate-agent"), str(CODE_DIR / "third_party" / "SWE-QA-Pro" / "eval")]

from models import Result  # noqa: E402
from score.metrics.results import format_number  # noqa: E402
from score.metrics.tool_calls import count_calls  # noqa: E402

EVAL_OUT = CODE_DIR / "evaluate-agent" / "out"
OUT = HERE / "out" / "figure5.txt"
OUTPUT_DIR = HERE / "out"

# 학습 전 agent 의 validation 결과
BASE = EVAL_OUT / "ministral-8b" / "valid" / "results"
# λ → validation 으로 고른 checkpoint 의 결과 (그림 5 의 점). 폴더 이름 끝의 e<숫자> 가 epoch, w<숫자> 가 λ (λ = 1 은 w 가 없다)
SELECTED = {
    1: EVAL_OUT / "ministral-8b-own-sis-e1" / "valid" / "results",
    5: EVAL_OUT / "ministral-8b-own-sis-w5-e2" / "valid" / "results",
    10: EVAL_OUT / "ministral-8b-own-sis-w10-e3" / "valid" / "results",
    15: EVAL_OUT / "ministral-8b-own-sis-w15-e3" / "valid" / "results",
}
# λ → 네 epoch 의 결과 (네 checkpoint 평균 호출)
ALL_EPOCHS = {
    1: [
        EVAL_OUT / "ministral-8b-own-sis-e1" / "valid" / "results",
        EVAL_OUT / "ministral-8b-own-sis-e2" / "valid" / "results",
        EVAL_OUT / "ministral-8b-own-sis-e3" / "valid" / "results",
        EVAL_OUT / "ministral-8b-own-sis-e4" / "valid" / "results",
    ],
    5: [
        EVAL_OUT / "ministral-8b-own-sis-w5-e1" / "valid" / "results",
        EVAL_OUT / "ministral-8b-own-sis-w5-e2" / "valid" / "results",
        EVAL_OUT / "ministral-8b-own-sis-w5-e3" / "valid" / "results",
        EVAL_OUT / "ministral-8b-own-sis-w5-e4" / "valid" / "results",
    ],
}
RED = "#BE5A46"
GRAY = "#6F7276"
TEXT = "#26282C"
AXIS = "#515356"
GRID = "#EAEBED"
FRAME = "#B7B9BD"
FIGURE_SIZE = (3.3, 2.2)
# 강조하는 λ (원고에서 쓴 값)
CHOSEN_WEIGHT = 5
# 축 범위. 점수 차이가 과장되지 않게 점수 축을 넉넉히 잡는다
CALLS_RANGE = (7, 15)
SCORE_RANGE = (33, 39)


def mean_score_and_calls(results_dir: Path) -> tuple[float, float]:
    """결과 폴더 하나의 (평균 점수, 평균 도구 호출)."""
    scores = []
    calls = []
    for path in sorted(results_dir.glob("*.json")):
        result = Result.model_validate_json(path.read_text())
        # 점수: 다섯 항목의 합 (Result.total). 채점이 없는 문항은 None 이라 뺀다
        if result.total is not None:
            scores.append(result.total)
        # 호출: 궤적의 도구 호출 수. 실패한 문항은 상한으로 센다 (score/metrics/tool_calls.py)
        calls.append(count_calls(result))
    if not calls:
        raise ValueError(f"No results in {results_dir}")
    return statistics.mean(scores), statistics.mean(calls)


def format_line(label: str, results_dir: Path) -> str:
    """'λ=5  <run id>  score 37.78  calls 11.20' 한 줄."""
    score, calls = mean_score_and_calls(results_dir)
    # 폴더는 <run id>/valid/results 이므로 두 칸 위가 run id
    run_id = results_dir.parents[1].name
    return f"{label:>6}  {run_id:30s} score {format_number(score, 2):>6}  calls {format_number(calls, 2):>6}"


def write_counts() -> None:
    # (1) 그림 5 의 점: Base 와 λ 마다 고른 checkpoint
    lines = [format_line("Base", BASE)]
    for weight, results_dir in SELECTED.items():
        lines.append(format_line(f"λ={weight}", results_dir))

    # (2) λ = 1, 5 의 네 checkpoint 평균 호출
    lines.append("")
    lines.append("Mean tool calls over the four checkpoints")
    for weight, results_dirs in ALL_EPOCHS.items():
        calls_per_epoch = []
        for results_dir in results_dirs:
            _, calls = mean_score_and_calls(results_dir)
            calls_per_epoch.append(calls)
        lines.append(f"{'λ=' + str(weight):>6}  {format_number(statistics.mean(calls_per_epoch), 2):>6}")

    print("\n".join(lines))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")


def read_halting_weight(path: Path) -> list[tuple[int, float, float]]:
    """λ 마다 (λ, 점수, 호출). Base 줄과 네 checkpoint 평균 표는 건너뛴다 (λ= 와 score 가 같이 있는 줄만)."""
    points = []
    for line in path.read_text().splitlines():
        match = re.match(r"\s*λ=(\d+)\s+\S+\s+score\s+([\d.]+)\s+calls\s+([\d.]+)", line)
        if match:
            points.append((int(match.group(1)), float(match.group(2)), float(match.group(3))))
    return points


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": FRAME,
            "axes.linewidth": 0.8,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            # 글꼴을 TrueType 으로 넣는다 (Type 3 은 뷰어에서 흐리게 보이고 투고 검사에서 걸릴 수 있다)
            "pdf.fonttype": 42,
        }
    )


def label_points(ax: plt.Axes, weights: list[int], values: list[float], color: str, above: bool) -> None:
    """점마다 값을 적는다. above 면 점 위에, 아니면 점 아래에. 원고에서 쓴 λ 만 굵게."""
    offset = (0, -6)
    va = "top"
    if above:
        offset = (0, 5)
        va = "bottom"
    for weight, value in zip(weights, values):
        fontweight = "normal"
        if weight == CHOSEN_WEIGHT:
            fontweight = "bold"
        ax.annotate(
            f"{value:.1f}",
            (weight, value),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va=va,
            color=color,
            fontweight=fontweight,
        )


def plot_lambda_sweep(points: list[tuple[int, float, float]]) -> None:
    weights = []
    scores = []
    calls = []
    for weight, score, call in points:
        weights.append(weight)
        scores.append(score)
        calls.append(call)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.spines["top"].set_visible(False)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    # 왼쪽 축: 도구 호출
    ax.plot(weights, calls, color=RED, linewidth=1.3, marker="o", markersize=4, zorder=3)
    label_points(ax, weights, calls, RED, above=False)
    ax.set_xticks(weights)
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(*CALLS_RANGE)
    ax.set_xlabel("Halting weight $\\lambda$")
    # 축 이름은 세우지 않고 축 위에 가로로 적는다 (세우면 화살표도 같이 돌아가 방향이 헷갈린다)
    ax.text(0, 1.03, "Tool calls $\\downarrow$", transform=ax.transAxes, ha="left", va="bottom", color=RED)
    ax.tick_params(axis="y", colors=RED)
    # 오른쪽 축: 점수
    right = ax.twinx()
    right.spines["top"].set_visible(False)
    right.tick_params(length=0, colors=GRAY)
    right.plot(weights, scores, color=GRAY, linewidth=1.3, marker="s", markersize=4, linestyle="--", zorder=3)
    label_points(right, weights, scores, GRAY, above=True)
    right.set_ylim(*SCORE_RANGE)
    ax.text(1, 1.03, "Score $\\uparrow$", transform=ax.transAxes, ha="right", va="bottom", color=GRAY)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        path = OUTPUT_DIR / f"figure5.{suffix}"
        # 둘레의 빈 여백을 잘라 LaTeX 에서 그림이 단 가운데에 오게 한다
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"saved {path}")
    plt.close(fig)


def draw() -> None:
    setup_style()
    plot_lambda_sweep(read_halting_weight(OUT))


def main() -> None:
    write_counts()
    draw()


if __name__ == "__main__":
    main()
