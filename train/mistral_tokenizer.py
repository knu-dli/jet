"""Saves the Ministral 3 tokenizer in a form that transformers and vLLM read correctly. With the original config,
transformers splits and decodes text wrongly.
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save the Ministral 3 tokenizer as a PreTrainedTokenizerFast that transformers and vLLM read correctly",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="mistralai/Ministral-3-8B-Instruct-2512-BF16", help="HF model id")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "out" / "ministral-8b-tokenizer"),
        help="where to save the tokenizer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(snapshot_download(args.model, allow_patterns=FILES))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if name != "tokenizer_config.json" and (source / name).exists():
            shutil.copy(source / name, output_dir / name)
    config = json.loads((source / "tokenizer_config.json").read_text())
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    (output_dir / "tokenizer_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"saved {output_dir} (tokenizer_class PreTrainedTokenizerFast)")


if __name__ == "__main__":
    main()
