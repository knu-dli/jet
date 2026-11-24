"""Score: the judge score of each question, the sum of the five criteria."""

import statistics
from pathlib import Path

from models import Result

# 벤치마크마다 점수를 읽는 폴더. 없는 벤치마크(validation)는 results/
SCORE_DIRS = {"sweqapro": "results-gpt-6-sol", "sweqa": "results-claude-sonnet-5-5"}
# 두 저자 채점기의 항목 (SWE-QA-Pro 1~10, SWE-QA 1~20)
CRITERIA = ["correctness", "completeness", "relevance", "clarity", "reasoning"]


def score_dir(run_dir: Path, benchmark: str) -> Path:
    """점수를 읽는 폴더."""
    return run_dir / benchmark / SCORE_DIRS.get(benchmark, "results")


def load_scores(score_dir: Path) -> list[int]:
    """채점된 문항의 총점 (다섯 항목의 합, models.Result.total)."""
    scores = []
    for path in sorted(score_dir.glob("*.json")):
        result = Result.model_validate_json(path.read_text())
        if result.total is not None:
            scores.append(result.total)
    return scores


def mean_criteria(score_dir: Path) -> dict[str, float | None]:
    """채점 항목별 평균 (채점된 문항만). 다섯 항목의 평균을 더하면 표의 Score 다."""
    values = {}
    for name in CRITERIA:
        values[name] = []
    for path in sorted(score_dir.glob("*.json")):
        result = Result.model_validate_json(path.read_text())
        if result.score is None:
            continue
        for name in CRITERIA:
            values[name].append(result.score[name])
    means = {}
    for name in CRITERIA:
        means[name] = statistics.mean(values[name]) if values[name] else None
    return means
