"""Converts the SWE-QA-Pro SFT trajectories to TRL chat rows (messages, tools) for the SFT baseline."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "build-dataset"))

from dataset import load_sft_trajectories  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the SWE-QA-Pro SFT trajectories into training rows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "sftdata"),
        help="output dir (train.jsonl is written here)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories = load_sft_trajectories()
    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for trajectory in trajectories:
            row = {
                "messages": trajectory.messages(),
                "tools": trajectory.tools,
                "trajectory_id": trajectory.trajectory_id,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(trajectories)} rows to {out_dir / 'train.jsonl'}")


if __name__ == "__main__":
    main()
