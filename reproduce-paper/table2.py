"""Table 2: Score, Calls, and Cost of every method, and the LaTeX of the table (shading and ±% against Base or SFT)."""

import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
# evaluate-agent 의 score·models 와 저자 코드 sweqapro 를 import 하려고
sys.path[:0] = [str(CODE_DIR / "evaluate-agent"), str(CODE_DIR / "third_party" / "SWE-QA-Pro" / "eval")]

from score.metrics.cost import summarize  # noqa: E402
from score.metrics.judge_score import load_scores, score_dir  # noqa: E402
from score.metrics.results import find_run_dir, format_number, load_results  # noqa: E402
from score.metrics.tool_calls import count_calls  # noqa: E402

OUT_DIR = HERE / "out"
BENCHMARKS = ["sweqapro", "sweqa"]
JET = r"\textbf{JET}"
ARROW = r" $\rightarrow$ "
# 표의 행: (모델, 학습 데이터, 방법, 평가 run id). run id 는 validation 으로 고른 checkpoint 다.
ROWS = [
    ("Qwen3-8B", "Self", "Base", "qwen3-8b-nothink"),
    ("Qwen3-8B", "Self", "Budget Tracker", "qwen3-8b-nothink-bats"),
    ("Qwen3-8B", "Self", "Autorater", "qwen3-8b-nothink-sufficiency"),
    ("Qwen3-8B", "Self", "Trunc. SFT", "qwen3-8b-nothink-own-plain-e4"),
    ("Qwen3-8B", "Self", JET, "qwen3-8b-nothink-own-sis-w5-e1"),
    ("Qwen3-8B", "Expert", "SFT", "qwen3-8b-nothink-sft-e4"),
    ("Qwen3-8B", "Expert", "Budget Tracker", "qwen3-8b-nothink-sft-e4-bats"),
    ("Qwen3-8B", "Expert", "Autorater", "qwen3-8b-nothink-sft-e4-sufficiency"),
    ("Qwen3-8B", "Expert", "Trunc. SFT", "qwen3-8b-nothink-expert-plain-e4"),
    ("Qwen3-8B", "Expert", JET, "qwen3-8b-nothink-expert-sis-w5-e4"),
    ("Qwen3-8B", "Expert" + ARROW + "Self", JET + ARROW + "Trunc. SFT", "qwen3-8b-nothink-expert-sis-own-plain-e3"),
    ("Qwen3-8B", "Expert" + ARROW + "Self", JET + ARROW + JET, "qwen3-8b-nothink-expert-sis-own-sis-w5-e1"),
    ("Ministral-3-8B", "Self", "Base", "ministral-8b"),
    ("Ministral-3-8B", "Self", "Budget Tracker", "ministral-8b-bats"),
    ("Ministral-3-8B", "Self", "Autorater", "ministral-8b-sufficiency"),
    ("Ministral-3-8B", "Self", "Trunc. SFT", "ministral-8b-own-plain-e4"),
    ("Ministral-3-8B", "Self", JET, "ministral-8b-own-sis-w5-e2"),
    ("Ministral-3-8B", "Expert", "SFT", "ministral-8b-sft-e2"),
    ("Ministral-3-8B", "Expert", "Budget Tracker", "ministral-8b-sft-e2-bats"),
    ("Ministral-3-8B", "Expert", "Autorater", "ministral-8b-sft-e2-sufficiency"),
    ("Ministral-3-8B", "Expert", "Trunc. SFT", "ministral-8b-expert-plain-e4"),
    ("Ministral-3-8B", "Expert", JET, "ministral-8b-expert-sis-w5-e3"),
    ("Ministral-3-8B", "Expert" + ARROW + "Self", JET + ARROW + "Trunc. SFT", "ministral-8b-expert-sis-own-plain-e4"),
    ("Ministral-3-8B", "Expert" + ARROW + "Self", JET + ARROW + JET, "ministral-8b-expert-sis-own-sis-w5-e3"),
]
# 기준 행의 방법 이름. 이 행이 나오면 그 뒤 행들의 기준이 바뀐다 (Expert → Self 는 바로 위 Expert 의 SFT 가 기준으로 남는다)
REFERENCE_METHODS = ["Base", "SFT"]
# Score 의 변화율에 곱하는 배율과 농도의 상한
SCORE_GAIN = 5
MAX_SHADE = 50
RULE = r"\arrayrulecolor{black!35}\specialrule{0.4pt}{0pt}{0pt}\arrayrulecolor{black}"
HEADER = r"""\begin{table*}[t]
\centering
\small
\setlength{\aboverulesep}{0pt}
\setlength{\belowrulesep}{0pt}
\setlength{\tabcolsep}{8.05pt}
% percent label next to the tool calls, and a blank of the same width (defined here so the table file stands alone)
\def\pct#1#2{\hspace{0.4em}{\footnotesize\textcolor[RGB]{90,90,90}{\makebox[0.8em][c]{#1}\makebox[1.9em][r]{#2\%}}}}
\def\pctblank{\hspace{0.4em}{\footnotesize\makebox[2.7em]{}}}
% shading for the last column: the right overhang of colortbl (\tabcolsep) is set to 0 so the shade ends at the rule
\makeatletter
\newcommand{\cellcolorR}[1]{\gdef\CT@cell@color{\CT@color{#1}\@tempdimc\z@\global\let\CT@cell@color\relax}}
\makeatother
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{@{}llccc!{\color{black!35}\vrule}ccc@{}}
\toprule
\textbf{Training data} & \textbf{Method} & \multicolumn{3}{c!{\color{black!35}\vrule}}{\textbf{SWE-QA-Pro}} & \multicolumn{3}{c}{\textbf{SWE-QA}} \\
 & & Score~$\uparrow$ & Calls~$\downarrow$\pctblank & Cost~(\textcent)$\downarrow$ & Score~$\uparrow$ & Calls~$\downarrow$\pctblank & Cost~(\textcent)$\downarrow$ \\"""
FOOTER = r"""\bottomrule
\end{tabular}
\caption{Main results on SWE-QA-Pro and SWE-QA. Cost is in cents per question. Shading and percentages are relative to Base in Self and to SFT in the other two blocks.}
\label{tab:main-results}
\end{table*}"""


def load_numbers(run_id: str) -> dict[str, dict[str, float]]:
    """benchmark → {score, calls, cost}. 표의 숫자는 모두 여기서 계산한다. 반올림하지 않은 평균이다."""
    # run id 의 폴더를 evaluate-agent/out, budget_tracker/out, autorater/out 에서 차례로 찾는다
    run_dir = find_run_dir(run_id)
    if run_dir is None:
        raise ValueError(f"No run dir for {run_id}")
    numbers = {}
    for benchmark in BENCHMARKS:
        # Score: 채점 폴더의 문항마다 다섯 항목 점수의 합 (score/metrics/judge_score.py). 그 평균이 표의 Score 다
        scores = load_scores(score_dir(run_dir, benchmark))

        # Calls: results/*.json 의 문항마다 궤적의 도구 호출 수 (score/metrics/tool_calls.py). 그 평균이 표의 Calls 다
        # 끝내 답을 못 낸 문항은 step limit 까지 간 것으로 보고 max_iterations - 1 로 센다
        calls = []
        for result in load_results(run_dir / benchmark):
            calls.append(count_calls(result))

        # Cost: 문항마다 agent 호출 전부의 token × 가격, Autorater 는 rater 호출의 token × 가격을 더한다
        # (score/metrics/cost.py). 문항 평균은 USD 라서 100 을 곱해 센트로 바꾼다
        cost = 100 * summarize(run_id, benchmark)["total"]

        numbers[benchmark] = {"score": statistics.mean(scores), "calls": statistics.mean(calls), "cost": cost}
    return numbers


def shade(value: float, reference: float, gain: float, higher_is_better: bool, command: str) -> str:
    """기준 대비 변화율을 농도로 한 칸 색. 변화율이 2% 미만이면 빈 글자."""
    # 예: Calls 18.10 → 12.19 는 100 × (12.19 − 18.10) / 18.10 = −33 이라 농도 33
    percent = round(gain * 100 * (value - reference) / reference)
    if abs(percent) < 2:
        return ""
    # Score 는 오르면 좋고 (higher_is_better), Calls 와 Cost 는 내리면 좋다
    color = "tabbad"
    if (percent > 0) == higher_is_better:
        color = "tabgood"
    return rf"\{command}{{{color}!{min(abs(percent), MAX_SHADE)}}}"


def percent_label(value: float, reference: float) -> str:
    """호출 옆에 적는 ±%. 농도와 달리 50 에서 자르지 않는다."""
    percent = round(100 * (value - reference) / reference)
    sign = "$+$"
    if percent < 0:
        sign = "$-$"
    return rf"\pct{{{sign}}}{{{abs(percent)}}}"


def format_cells(numbers: dict[str, dict[str, float]], reference: dict[str, dict[str, float]] | None) -> list[str]:
    """한 행의 여섯 칸 (벤치마크마다 Score, Calls, Cost). reference 가 None 이면 기준 행이라 색과 ±% 가 없다."""
    cells = []
    for benchmark in BENCHMARKS:
        current = numbers[benchmark]
        # 표에 적는 숫자는 소수 둘째 자리로 반올림한다. 음영과 ±% 는 반올림하지 않은 값으로 계산한다
        score = format_number(current["score"], 2)
        calls = format_number(current["calls"], 2)
        cost = format_number(current["cost"], 2)
        if reference is None:
            cells.extend([score, calls + r"\pctblank", cost])
            continue
        base = reference[benchmark]
        # 표의 마지막 열만 색이 세로 선에서 끝나게 \cellcolorR 을 쓴다
        cost_command = "cellcolor"
        if benchmark == BENCHMARKS[-1]:
            cost_command = "cellcolorR"
        cells.append(shade(current["score"], base["score"], SCORE_GAIN, True, "cellcolor") + score)
        cells.append(shade(current["calls"], base["calls"], 1, False, "cellcolor") + calls + percent_label(current["calls"], base["calls"]))
        cells.append(shade(current["cost"], base["cost"], 1, False, cost_command) + cost)
    return cells


def main() -> None:
    text = [f"{'model / data / method':58s}" + f"{'Score':>8}{'Calls':>8}{'Cost':>7}" * len(BENCHMARKS) + "  run id"]
    latex = [HEADER]
    last_model, last_group, reference = None, None, None
    for model, group, method, run_id in ROWS:
        numbers = load_numbers(run_id)

        # 숫자만 적은 줄 (out/table2.txt)
        line = f"{model:16s}{group.replace(ARROW, ' → '):15s}{method.replace(ARROW, ' → ').replace(JET, 'JET'):27s}"
        for benchmark in BENCHMARKS:
            current = numbers[benchmark]
            line += f"{format_number(current['score'], 2):>8}{format_number(current['calls'], 2):>8}{format_number(current['cost'], 2):>7}"
        text.append(line + f"  {run_id}")

        # LaTeX 의 줄 (out/table2.tex): 모델이 바뀌면 머리 줄, 학습 데이터가 바뀌면 옅은 가로 선
        if model != last_model:
            latex.append(r"\midrule")
            latex.append(rf"\rowcolor{{tabhead}}[0pt][0pt]\multicolumn{{8}}{{@{{}}l@{{}}}}{{\textbf{{{model}}}}} \\")
        elif group != last_group:
            latex.append(RULE)
        first = ""
        if model != last_model or group != last_group:
            first = group
        # Base 와 SFT 는 기준 행이다. 색 없이 적고, 그 뒤 행들이 이 숫자와 비교된다
        if method in REFERENCE_METHODS:
            cells = format_cells(numbers, None)
            reference = numbers
        else:
            cells = format_cells(numbers, reference)
        latex.append(first + " & " + method + " & " + " & ".join(cells) + r" \\")
        last_model, last_group = model, group
    latex.append(FOOTER)

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "table2.txt").write_text("\n".join(text) + "\n")
    (OUT_DIR / "table2.tex").write_text("\n".join(latex) + "\n")
    print("\n".join(text))
    print(f"saved {OUT_DIR / 'table2.txt'} and {OUT_DIR / 'table2.tex'}")


if __name__ == "__main__":
    main()
