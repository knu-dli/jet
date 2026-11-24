"""Figure 4: questions on which the agent reaches the step limit."""

import re
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
from score.metrics.step_limit import reached_step_limit  # noqa: E402

EVAL_OUT = CODE_DIR / "evaluate-agent" / "out"
OUT = HERE / "out" / "figure4.txt"
OUTPUT_DIR = HERE / "out"
# (결과 폴더 이름, 문항 수)
BENCHMARKS = [("sweqapro", 260), ("sweqa", 720)]
# (모델, 학습 데이터, 방법, 평가 run id). 표 2 에 쓴 checkpoint 와 같다 (run id 끝의 e<숫자> 가 epoch)
ROWS = [
    ("Qwen3-8B", "Self", "Base", "qwen3-8b-nothink"),
    ("Qwen3-8B", "Self", "JET", "qwen3-8b-nothink-own-sis-w5-e1"),
    ("Qwen3-8B", "Expert", "SFT", "qwen3-8b-nothink-sft-e4"),
    ("Qwen3-8B", "Expert", "JET", "qwen3-8b-nothink-expert-sis-w5-e4"),
    ("Qwen3-8B", "Expert → Self", "JET → JET", "qwen3-8b-nothink-expert-sis-own-sis-w5-e1"),
    ("Ministral-3-8B", "Self", "Base", "ministral-8b"),
    ("Ministral-3-8B", "Self", "JET", "ministral-8b-own-sis-w5-e2"),
    ("Ministral-3-8B", "Expert", "SFT", "ministral-8b-sft-e2"),
    ("Ministral-3-8B", "Expert", "JET", "ministral-8b-expert-sis-w5-e3"),
    ("Ministral-3-8B", "Expert → Self", "JET → JET", "ministral-8b-expert-sis-own-sis-w5-e3"),
]
RED = "#BE5A46"
GRAY = "#9EA1A5"
TEXT = "#26282C"
AXIS = "#515356"
GRID = "#EAEBED"
FRAME = "#B7B9BD"
# figure4.txt 의 모델 이름 → 패널 제목
MODELS = [("Qwen3-8B", "Qwen3"), ("Ministral-3-8B", "Ministral")]
# 패널 행의 벤치마크 이름, 문항 수 (figure4.txt 의 열 순서)
PANELS = [("SWE-QA-Pro", 260), ("SWE-QA", 720)]
# 묶음 이름 (= figure4.txt 의 data 열), 그 묶음에서 JET 와 비교하는 방법
GROUPS = [("Self", "Base"), ("Expert", "SFT")]
BAR_WIDTH = 0.38
FIGURE_SIZE = (3.3, 3.7)
# 그림 안 모든 글자의 크기 (pt). 원고에 단 폭 그대로 넣으므로 9pt 이상으로 둔다
FONT_SIZE = 10
# 세로축 끝 (%). 가장 높은 막대 위의 숫자와 패널 안 벤치마크 이름이 겹치지 않게 눈금 끝 (50) 보다 높게 둔다
Y_MAX = 57
# figure4.txt 의 한 줄: 모델, data, method, SWE-QA-Pro 문항 수와 %, SWE-QA 문항 수와 %
ROW_PATTERN = r"(\S+)\s+(Self|Expert → Self|Expert)\s+(Base|SFT|JET → JET|JET)\s+(\d+)\s+\d+%\s+(\d+)\s+\d+%"


def count_step_limit(results_dir: Path) -> int:
    """결과 폴더 하나에서 step limit 에 걸린 문항 수."""
    paths = sorted(results_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No results in {results_dir}")
    count = 0
    for path in paths:
        result = Result.model_validate_json(path.read_text())
        if reached_step_limit(result):
            count += 1
    return count


def write_counts() -> None:
    # 머리 두 줄: 벤치마크 이름, 그 아래 limit (문항 수) 와 % (문항 비율)
    header = f"{'':46s}"
    columns = f"{'model / data / method':46s}"
    for benchmark, _ in BENCHMARKS:
        header += f"{benchmark:^16}"
        columns += f"{'limit':>8}{'%':>8}"
    lines = [header, columns]
    for model, group, method, run_id in ROWS:
        line = f"{model:16s}{group:15s}{method:15s}"
        for benchmark, num_questions in BENCHMARKS:
            count = count_step_limit(EVAL_OUT / run_id / benchmark / "results")
            line += f"{count:>8}{count / num_questions:>8.0%}"
        lines.append(line + f"  {run_id}")
    print("\n".join(lines))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")


def read_step_limit(path: Path) -> dict[tuple[str, str, str], list[int]]:
    """(모델, data, method) → [SWE-QA-Pro 문항 수, SWE-QA 문항 수]."""
    counts = {}
    for line in path.read_text().splitlines():
        match = re.match(ROW_PATTERN, line)
        if match:
            counts[(match.group(1), match.group(2), match.group(3))] = [int(match.group(4)), int(match.group(5))]
    if not counts:
        raise ValueError(f"no step limit rows in {path}")
    return counts


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "font.size": FONT_SIZE,
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


def draw_bar(ax: plt.Axes, x: float, count: int, total: int, is_jet: bool) -> None:
    """막대 하나와 그 위의 문항 수. JET 는 빨강·굵은 글씨, 비교 대상은 회색."""
    color = GRAY
    text_color = AXIS
    weight = "normal"
    if is_jet:
        color = RED
        text_color = RED
        weight = "bold"
    percent = 100 * count / total
    ax.bar(x, percent, width=BAR_WIDTH, color=color, linewidth=0)
    ax.text(x, percent + 0.7, str(count), ha="center", va="bottom", fontsize=FONT_SIZE, color=text_color, fontweight=weight)


def plot_step_limit(counts: dict[tuple[str, str, str], list[int]]) -> None:
    fig, axes = plt.subplots(len(PANELS), len(MODELS), figsize=FIGURE_SIZE, sharex=True, sharey=True)
    group_names = []
    for group, _ in GROUPS:
        group_names.append(group)
    for row, (benchmark, total) in enumerate(PANELS):
        for col, (model, title) in enumerate(MODELS):
            ax = axes[row][col]
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
            # 묶음마다 왼쪽 막대는 비교 대상 (Base 나 SFT), 오른쪽 막대는 JET
            for idx, (group, compared) in enumerate(GROUPS):
                draw_bar(ax, idx - BAR_WIDTH / 2, counts[(model, group, compared)][row], total, False)
                draw_bar(ax, idx + BAR_WIDTH / 2, counts[(model, group, "JET")][row], total, True)
            ax.set_xticks(range(len(GROUPS)))
            ax.set_xticklabels(group_names, fontsize=FONT_SIZE)
            ax.set_xlim(-0.55, len(GROUPS) - 0.45)
            ax.set_ylim(0, Y_MAX)
            ax.set_yticks(range(0, Y_MAX, 10))
            ax.tick_params(axis="y", labelsize=FONT_SIZE, pad=1)
            # 벤치마크 이름은 패널 안 오른쪽 위에 (바깥에 세로로 쓰면 두 행의 이름이 겹친다)
            ax.annotate(benchmark, (0.98, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=FONT_SIZE, color=AXIS)
            if row == 0:
                ax.set_title(title, fontsize=FONT_SIZE, pad=3)
    fig.text(0.0, 0.5, "Questions at the step limit (%)", rotation=90, ha="left", va="center", fontsize=FONT_SIZE)
    fig.subplots_adjust(left=0.125, right=1.0, bottom=0.06, top=0.91, wspace=0.05, hspace=0.08)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=GRAY, linewidth=0),
        plt.Rectangle((0, 0), 1, 1, facecolor=RED, linewidth=0),
    ]
    fig.legend(
        handles,
        ["Base (Self) / SFT (Expert)", "JET"],
        fontsize=FONT_SIZE,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0,
        borderpad=0,
        ncol=2,
        handlelength=1.4,
        columnspacing=1.2,
    )
    for suffix in ("pdf", "png"):
        path = OUTPUT_DIR / f"figure4.{suffix}"
        # 둘레의 빈 여백을 잘라 LaTeX 에서 그림이 단 가운데에 오게 한다
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"saved {path}")
    plt.close(fig)


def draw() -> None:
    setup_style()
    plot_step_limit(read_step_limit(OUT))


def main() -> None:
    write_counts()
    draw()


if __name__ == "__main__":
    main()
