"""Figure 2: forced-answer score and the trajectories still running at each step."""

import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
# build-dataset 의 golden_step.models 를 import 하려고
sys.path[:0] = [str(CODE_DIR / "build-dataset"), str(CODE_DIR / "third_party" / "SWE-QA-Pro" / "eval")]

from golden_step.models import Run  # noqa: E402

GOLDEN_STEP_OUT = CODE_DIR / "build-dataset" / "golden_step" / "out"
# (패널, 모델, golden step 파일 폴더)
RUNS = [
    ("(a) Agent", "Ministral", GOLDEN_STEP_OUT / "ministral-8b-sweqapro" / "answers"),
    ("(a) Agent", "Qwen3", GOLDEN_STEP_OUT / "qwen3-8b-nothink-sweqapro" / "answers"),
    ("(b) Expert", "Ministral", GOLDEN_STEP_OUT / "ministral-8b" / "answers"),
    ("(b) Expert", "Qwen3", GOLDEN_STEP_OUT / "qwen3-8b-nothink" / "answers"),
]
# 곡선이 평평해졌다고 보는 기준: 끝 점수와의 차이 (점). 본문에 적을 step 을 고를 때만 쓴다
MARGIN = 0.5
OUT = HERE / "out" / "figure2.txt"
OUTPUT = HERE / "out" / "figure2"
RED = "#BE5A46"
MUSTARD = "#C4921E"
GRAY = "#9EA1A5"
TEXT = "#26282C"
AXIS = "#515356"
GRID = "#EAEBED"
FRAME = "#B7B9BD"
PANELS = ["(a) Agent", "(b) Expert"]
# (figure2.txt 의 모델 이름, 선 색)
MODELS = [("Ministral", RED), ("Qwen3", MUSTARD)]
# 폭은 단 폭. 높이는 아래 패널에 세로축 이름 "Running (%)" 이 한 줄로 들어가게 잡는다
FIGURE_SIZE = (3.3, 3.0)
FONT_SIZE = 10
# 세로 점선을 긋는 step (가로축 눈금 자리) 과 점선의 색
GUIDE_STEPS = [10, 20]
GUIDE = "#B7B9BD"
# figure2.txt 의 한 줄: 패널 | 모델 | step | score | running
ROW_PATTERN = r"(.+) \| (\S+) \| step\s+\d+ \| score ([\d.]+) \| running (\d+)%"


def load_trajectories(answer_dir: Path) -> tuple[list[list[float]], int]:
    """궤적마다 step 0..n 의 점수 목록을 읽는다. 돌려주는 것: (궤적들, 뺀 궤적 수)."""
    trajectories = []
    num_skipped = 0
    for path in sorted(answer_dir.glob("*.json")):
        run = Run.model_validate_json(path.read_text())
        scores = []
        for step in run.steps:
            totals = []
            for answer in step.answers:
                if answer.total is not None:
                    totals.append(answer.total)
            if totals:
                scores.append(sum(totals) / len(totals))
        # 점수가 하나도 없는 step 이 있으면 곡선이 끊기므로 그 궤적은 뺀다
        if len(scores) != len(run.steps):
            num_skipped += 1
            continue
        trajectories.append(scores)
    return trajectories, num_skipped


def write_counts() -> None:
    lines = []
    for panel, model, answer_dir in RUNS:
        trajectories, num_skipped = load_trajectories(answer_dir)
        max_steps = 0
        for scores in trajectories:
            max_steps = max(max_steps, len(scores) - 1)

        # step s 의 값 = 궤적을 s step 까지만 허용했을 때의 평균 점수. 그 전에 끝난 궤적은 끝 점수를 그대로 쓴다
        curve = []
        running = []
        for step in range(max_steps + 1):
            total = 0.0
            num_running = 0
            for scores in trajectories:
                last = len(scores) - 1
                total += scores[min(step, last)]
                if step <= last:
                    num_running += 1
            curve.append(total / len(trajectories))
            running.append(100 * num_running / len(trajectories))

        # 점수가 끝 점수에서 MARGIN 안으로 처음 들어오는 step
        flat = 0
        while abs(curve[flat] - curve[-1]) > MARGIN:
            flat += 1

        lines.append(
            f"{panel} | {model}: {len(trajectories)} trajectories ({num_skipped} skipped), "
            f"within {MARGIN} of the final score from step {flat} ({running[flat]:.0f}% still running)"
        )
        for step in range(max_steps + 1):
            lines.append(f"{panel} | {model} | step {step:2d} | score {curve[step]:.2f} | running {running[step]:.0f}%")

    print("\n".join(lines))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")


def read_score_curve() -> tuple[dict, dict]:
    """(패널, 모델) → step 0..n 의 점수, (패널, 모델) → step 0..n 의 진행 중인 궤적의 비율 (%)."""
    scores = {}
    running = {}
    for line in OUT.read_text().splitlines():
        match = re.match(ROW_PATTERN, line)
        if match:
            key = (match.group(1), match.group(2))
            scores.setdefault(key, []).append(float(match.group(3)))
            running.setdefault(key, []).append(float(match.group(4)))
    return scores, running


def draw() -> None:
    scores, running = read_score_curve()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans"],
            "font.size": FONT_SIZE,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": FRAME,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
        }
    )
    # 위아래는 가로축을, 아래 줄의 두 패널은 세로축 (0~100%) 을 함께 쓴다
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, sharex="col", height_ratios=[1.6, 1])
    axes[1][1].sharey(axes[1][0])

    for col, panel in enumerate(PANELS):
        # 위: 점수 곡선
        for model, color in MODELS:
            axes[0][col].plot(scores[(panel, model)], color=color, linewidth=1.5, label=model)
        axes[0][col].set_title(panel, fontsize=FONT_SIZE, pad=3)

        # 아래: 진행 중인 궤적의 비율
        if panel == "(a) Agent":
            for model, color in MODELS:
                axes[1][col].plot(running[(panel, model)], color=color, linewidth=1.5)
        else:
            # 두 모델의 값이 같다 (점수가 없어 뺀 궤적만 몇 개 달라 1%p 안에서 다르다). 첫 모델의 값을 그린다
            axes[1][col].plot(running[(panel, MODELS[0][0])], color=GRAY, linewidth=1.5)
        axes[1][col].set_xlim(0, len(scores[(panel, MODELS[0][0])]) - 1)
        axes[1][col].set_xticks([0, 10, 20])

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        # 위아래 패널에서 같은 step 을 따라 읽을 수 있게 세로 점선을 곡선 뒤에 긋는다
        for step in GUIDE_STEPS:
            ax.axvline(step, color=GUIDE, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
        ax.tick_params(length=0, labelsize=FONT_SIZE, pad=2)

    axes[1][0].set_ylim(0, 110)
    # 0 은 적지 않는다 (왼쪽 아래 모서리에서 가로축의 0 과 붙어 보인다)
    axes[1][0].set_yticks([50, 100])
    # 세로축을 함께 쓰므로 눈금 숫자는 왼쪽 패널에만
    axes[1][1].tick_params(labelleft=False)
    # 세로축 이름은 축의 가운데에 온다. 가운데 눈금 (30) 도 축의 가운데에 오게 범위를 30 을 중심으로 잡는다
    # (기본 범위에서는 30 이 가운데보다 위에 있어 이름이 아래로 치우쳐 보인다)
    axes[0][0].set_ylim(21.5, 38.5)
    axes[0][0].set_yticks([25, 30, 35])
    axes[0][0].set_ylabel("Score", fontsize=FONT_SIZE)
    axes[1][0].set_ylabel("Running (%)", fontsize=FONT_SIZE)
    fig.align_ylabels(axes[:, 0])
    # 범례는 두 패널이 함께 쓰므로 패널 위 가운데에 한 줄로 둔다 (곡선을 가리지 않는다)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=FONT_SIZE,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.59, 1.0),
        ncol=len(MODELS),
        handlelength=1.4,
        columnspacing=1.5,
        borderaxespad=0,
        borderpad=0,
    )
    # 가로축 이름은 두 패널 아래 가운데에 하나만, 눈금 숫자 바로 아래에
    fig.text(0.59, 0.02, "Step", ha="center", va="bottom", fontsize=FONT_SIZE)
    fig.subplots_adjust(left=0.2, right=0.99, bottom=0.14, top=0.86, wspace=0.24, hspace=0.18)

    # pdf 는 ghostscript 로 글자를 모두 도형으로 바꾼다 (글꼴이 들어가지 않는다)
    fig.savefig(f"{OUTPUT}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{OUTPUT}.raw.pdf", bbox_inches="tight", pad_inches=0.02)
    command = ["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite", "-dNoOutputFonts"]
    subprocess.run(command + [f"-sOutputFile={OUTPUT}.pdf", f"{OUTPUT}.raw.pdf"], check=True)
    Path(f"{OUTPUT}.raw.pdf").unlink()
    print(f"saved {OUTPUT}.pdf")


def main() -> None:
    write_counts()
    draw()


if __name__ == "__main__":
    main()
