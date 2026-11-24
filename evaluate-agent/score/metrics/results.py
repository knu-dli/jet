"""Finds the folder of a run and reads its result files."""

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from models import Result

HERE = Path(__file__).resolve().parents[2]
# 실행분 폴더가 있는 곳: evaluate-agent/out/, 기준선 둘의 out/, 절제 폴더마다의 out/
OUTPUT_DIRS = [HERE / "out", HERE / "budget_tracker" / "out", HERE / "autorater" / "out"]
OUTPUT_DIRS.extend(sorted((HERE.parent / "ablation").glob("*/out")))


def find_run_dir(run_id: str, output_dirs: list[Path] = OUTPUT_DIRS) -> Path | None:
    """run-id 의 실행분 폴더. 없으면 None."""
    for output_dir in output_dirs:
        if (Path(output_dir) / run_id).is_dir():
            return Path(output_dir) / run_id
    return None


def load_results(run_dir: Path) -> list[Result]:
    """<run_dir>/results/*.json 전부."""
    results = []
    for path in sorted((run_dir / "results").glob("*.json")):
        results.append(Result.model_validate_json(path.read_text()))
    return results


def format_number(value: float | None, digits: int) -> str:
    """원고 규칙: 정확한 평균에서 반올림 (ROUND_HALF_UP). round() 는 5 를 짝수 쪽으로 보낸다."""
    if value is None:
        return "-"
    return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))
