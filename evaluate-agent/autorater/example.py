"""Builds the one-shot example of the autorater prompt from a rollout on a training question."""

import argparse
import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

HERE = Path(__file__).resolve().parent
QUESTION_ID = 420
HALTING_STEP = 4
# 논문 예시의 설명 문체 (무엇이 쓰여 있고, 그래서 답을 추론할 수 있다)
EXPLANATION = (
    "The first two commands locate the optimizer modules, the search for `AVAILABLE_ALGORITHMS` points to "
    "`estimagic/optimization/__init__.py`, and the view of that file shows the mechanism: `COLLECTED_FUNCTIONS` is built "
    "with `inspect.getmembers(..., inspect.isfunction)` from `scipy_optimizers`, and the other optimizer modules are added "
    "under the `IS_*_INSTALLED` flags imported from `estimagic.config`. The question asks how the dictionary is built from "
    "multiple modules with conditional dependency checking, and this file shows exactly that. Therefore, the context does "
    "provide a precise answer to the question."
)


def build_context(result: dict, halting_step: int) -> str:
    """스텝 0..halting_step-1 의 assistant·tool 메시지 목록을 rater 가 받는 형식(repr)으로."""
    messages = []
    for turn in result["trajectory"][:halting_step]:
        calls = []
        for idx, call in enumerate(turn["tool_calls"]):
            calls.append(
                {"name": call["tool"], "args": call["args"], "id": f"call_{turn['step']}_{idx}", "type": "tool_call"}
            )
        messages.append(AIMessage(content=turn["response"], tool_calls=calls))
        for idx, call in enumerate(turn["tool_calls"]):
            messages.append(ToolMessage(content=call["result"], tool_call_id=f"call_{turn['step']}_{idx}"))
    return str(messages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the one-shot example of the Sufficiency Rater from a training-question rollout",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=str,
        default=str(HERE.parent / "out" / "qwen3-8b-nothink-own" / "sweqapro-rl" / "results"),
        help="results dir of the rollout the example is taken from",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=str(HERE / "example.txt"), help="where to write the example"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = json.loads((Path(args.results).expanduser() / f"{QUESTION_ID:03d}.json").read_text())
    example = (
        "### QUESTION\n"
        f"{result['item']['question']}\n"
        "### References\n"
        f"{build_context(result, HALTING_STEP)}\n"
        "### EXPLANATION\n"
        f"{EXPLANATION}\n"
        "### JSON\n"
        '{"Sufficient Context": 1}'
    )
    Path(args.output).expanduser().write_text(example + "\n", encoding="utf-8")
    print(f"saved {args.output} ({len(example)} chars)")


if __name__ == "__main__":
    main()
