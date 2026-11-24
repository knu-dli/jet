"""Table 3: golden step ablation on SWE-QA-Pro in the Self setting (truncation at n/2 and 2n/3)."""

import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
# evaluate-agent 의 models·score 와 저자 코드 sweqapro 를 import 하려고
sys.path[:0] = [str(CODE_DIR / "evaluate-agent"), str(CODE_DIR / "third_party" / "SWE-QA-Pro" / "eval")]

from models import Result  # noqa: E402
from score.metrics.results import format_number  # noqa: E402
from score.metrics.tool_calls import count_calls  # noqa: E402

OUT_DIR = HERE / "out"
# Base 와 JET 는 표 2 와 같은 run, 고정 비율 판은 절제 폴더에 있다
EVAL_OUT = CODE_DIR / "evaluate-agent" / "out"
ABLATION_OUT = CODE_DIR / "ablation" / "position" / "out"
# 표의 행: (모델, 자르는 위치, LaTeX 로 적은 자르는 위치, 평가 run 폴더). 모델마다 첫 행 Base 가 변화율의 기준
ROWS = [
    ("Qwen3-8B", "Base", "Base", EVAL_OUT / "qwen3-8b-nothink"),
    ("Qwen3-8B", "n/2", "$n/2$", ABLATION_OUT / "qwen3-8b-nothink-own-1of2"),
    ("Qwen3-8B", "2n/3", "$2n/3$", ABLATION_OUT / "qwen3-8b-nothink-own-2of3"),
    ("Qwen3-8B", "i* (JET)", r"$i^\star$ (\textbf{JET})", EVAL_OUT / "qwen3-8b-nothink-own-sis-w5-e1"),
    ("Ministral-3-8B", "Base", "Base", EVAL_OUT / "ministral-8b"),
    ("Ministral-3-8B", "n/2", "$n/2$", ABLATION_OUT / "ministral-8b-own-1of2"),
    ("Ministral-3-8B", "2n/3", "$2n/3$", ABLATION_OUT / "ministral-8b-own-2of3"),
    ("Ministral-3-8B", "i* (JET)", r"$i^\star$ (\textbf{JET})", EVAL_OUT / "ministral-8b-own-sis-w5-e2"),
]
# 음영: Score 는 변화가 작아 5배 해서 칠한다. 농도의 상한과 하한
SCORE_GAIN = 5
MAX_SHADE = 50
MIN_SHADE = 2
# LaTeX 의 표 머리와 꼬리 (\pct 는 호출 옆 ±%, \pctblank 는 Base 행의 빈 자리)
TABLE_HEAD = r"""\begin{table}[t]
\centering
\small
\setlength{\aboverulesep}{0pt}
\setlength{\belowrulesep}{0pt}
\setlength{\tabcolsep}{4pt}
% same percent label as table1 (defined here so the table file stands alone)
\def\pct#1#2{\hspace{0.4em}{\footnotesize\textcolor[RGB]{90,90,90}{\makebox[0.8em][c]{#1}\makebox[1.9em][r]{#2\%}}}}
\def\pctblank{\hspace{0.4em}{\footnotesize\makebox[2.7em]{}}}
\renewcommand{\arraystretch}{1.15}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llcc}
\toprule
\multicolumn{2}{l}{\textbf{Truncation point}} & Score~$\uparrow$ & Calls~$\downarrow$ \\"""
TABLE_TAIL = r"""\bottomrule
\end{tabular*}
\caption{Ablation of the golden step on SWE-QA-Pro in the Self setting. $n/2$ and $2n/3$ truncate each trajectory at a fixed ratio of the trajectory length $n$. Shading and percentages are relative to Base.}
\label{tab:golden-step}
\end{table}"""
# Base 행과 나머지 행 사이의 옅은 선
THIN_RULE = r"\arrayrulecolor{black!35}\specialrule{0.4pt}{0pt}{0pt}\arrayrulecolor{black}"


def mean_score_and_calls(run_dir: Path) -> tuple[float, float]:
    """run 하나의 SWE-QA-Pro (평균 점수, 평균 도구 호출)."""
    # 점수: Sol 로 채점한 폴더에서 문항마다 다섯 항목의 합 (Result.total)
    scores = []
    for path in sorted((run_dir / "sweqapro" / "results-gpt-6-sol").glob("*.json")):
        result = Result.model_validate_json(path.read_text())
        if result.total is not None:
            scores.append(result.total)
    # 호출: 궤적의 도구 호출 수. 실패한 문항은 상한으로 센다 (score/metrics/tool_calls.py)
    calls = []
    for path in sorted((run_dir / "sweqapro" / "results").glob("*.json")):
        calls.append(count_calls(Result.model_validate_json(path.read_text())))
    if not scores or not calls:
        raise ValueError(f"No results in {run_dir}")
    return statistics.mean(scores), statistics.mean(calls)


def percent_change(value: float, base: float) -> int:
    """Base 대비 변화율 (%), 정수로 반올림."""
    return round(100 * (value - base) / base)


def shade(value: float, base: float, gain: float, higher_is_better: bool) -> str:
    """Base 대비 변화율을 농도로 한 칸 색. 농도가 MIN_SHADE 미만이면 빈 글자."""
    # 예: Calls 18.10 → 12.19 는 100 × (12.19 − 18.10) / 18.10 = −33 이라 농도 33
    percent = round(gain * 100 * (value - base) / base)
    if abs(percent) < MIN_SHADE:
        return ""
    # Score 는 오르면 좋고 (higher_is_better), Calls 는 내리면 좋다
    color = "tabbad"
    if (percent > 0) == higher_is_better:
        color = "tabgood"
    return rf"\cellcolor{{{color}!{min(abs(percent), MAX_SHADE)}}}"


def percent_label(value: float, base: float) -> str:
    """호출 옆에 적는 ±%. 농도와 달리 자르지 않는다."""
    percent = percent_change(value, base)
    sign = "$+$"
    if percent < 0:
        sign = "$-$"
    return rf"\pct{{{sign}}}{{{abs(percent)}}}"


def main() -> None:
    text_lines = [f"{'model / cut':28s}{'Score':>8}{'Calls':>8}{'ΔScore':>8}{'ΔCalls':>8}"]
    tex_lines = [TABLE_HEAD]
    base_score = None
    base_calls = None
    for model, cut, cut_tex, run_dir in ROWS:
        score, calls = mean_score_and_calls(run_dir)
        # 표에 적는 숫자는 소수 둘째 자리로 반올림한다. 음영과 ±% 는 반올림하지 않은 값으로 계산한다
        score_text = format_number(score, 2)
        calls_text = format_number(calls, 2)
        text_line = f"{model:16s}{cut:12s}{score_text:>8}{calls_text:>8}"
        if cut == "Base":
            # 모델의 첫 행: 모델 이름 줄, Base 행, 옅은 선
            base_score = score
            base_calls = calls
            text_line += f"{'':>16}"
            tex_lines.append(r"\midrule")
            tex_lines.append(rf"\rowcolor{{tabhead}}\multicolumn{{4}}{{l}}{{\textbf{{{model}}}}} \\")
            tex_lines.append(rf" & Base & {score_text} & {calls_text}\pctblank \\")
            tex_lines.append(THIN_RULE)
        else:
            text_line += f"{percent_change(score, base_score):>+7d}%{percent_change(calls, base_calls):>+7d}%"
            # 자른 세 행의 첫 칸은 세 줄에 걸친 Truncation
            first = ""
            if cut == "n/2":
                first = r"\multirow{3}{*}{Truncation}"
            score_cell = shade(score, base_score, SCORE_GAIN, True) + score_text
            calls_cell = shade(calls, base_calls, 1, False) + calls_text + percent_label(calls, base_calls)
            tex_lines.append(rf"{first} & {cut_tex} & {score_cell} & {calls_cell} \\")
        text_lines.append(text_line + f"  {run_dir.name}")
    tex_lines.append(TABLE_TAIL)

    print("\n".join(text_lines))
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "table3.txt").write_text("\n".join(text_lines) + "\n")
    (OUT_DIR / "table3.tex").write_text("\n".join(tex_lines) + "\n")
    print(f"saved {OUT_DIR / 'table3.txt'} and {OUT_DIR / 'table3.tex'}")


if __name__ == "__main__":
    main()
