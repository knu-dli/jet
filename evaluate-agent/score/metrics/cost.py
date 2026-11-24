"""Cost: the inference cost per question in USD, from the agent tokens and, for the autorater, the rater tokens."""

import argparse
import json
import statistics
from pathlib import Path

from score.metrics.results import find_run_dir, load_results

# USD / 1M tokens (input, output), OpenRouter 2026-11-01.
# agent 둘 다 mistralai/ministral-8b-2512 가격. Qwen3-8B 는 OpenRouter 목록에서 빠져 확인할 수 있는 가격이 없어 같은 값을 쓴다.
AGENT_PRICE = (0.15, 0.15)
RATER_PRICE = (0.10, 0.50)  # openai/gpt-6-luna, 입력 272K 이하 구간 (rater 호출 최대 입력 140K)


def cost_usd(prompt_tokens: int, completion_tokens: int, price: tuple[float, float]) -> float:
    return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1e6


def load_rater_tokens(run_dir: Path) -> dict[int, tuple[int, int]]:
    """문항 번호 → rater 호출 전부의 (prompt, completion) token. Autorater 실행분에만 ratings/ 가 있다."""
    tokens = {}
    for path in sorted((run_dir / "ratings").glob("*.json")):
        data = json.loads(path.read_text())
        prompt, completion = 0, 0
        for rating in data["ratings"]:
            usage = rating.get("token_usage") or {}
            prompt += usage.get("prompt_tokens", 0)
            completion += usage.get("completion_tokens", 0)
        tokens[data["index"]] = (prompt, completion)
    return tokens


def summarize(run_id: str, benchmark: str) -> dict:
    run_dir = find_run_dir(run_id) / benchmark
    rater_tokens = load_rater_tokens(run_dir)
    agent_costs, rater_costs, total_costs = [], [], []
    prompts, completions = [], []
    empty = 0
    for result in load_results(run_dir):
        if not result.trajectory:
            empty += 1
            continue
        prompt, completion = 0, 0
        for turn in result.trajectory:
            usage = turn.token_usage or {}
            prompt += usage.get("prompt_tokens", 0)
            completion += usage.get("completion_tokens", 0)
        agent = cost_usd(prompt, completion, AGENT_PRICE)
        rater_prompt, rater_completion = rater_tokens.get(result.item.index, (0, 0))
        rater = cost_usd(rater_prompt, rater_completion, RATER_PRICE)
        agent_costs.append(agent)
        rater_costs.append(rater)
        total_costs.append(agent + rater)
        prompts.append(prompt)
        completions.append(completion)
    return {
        "n": len(total_costs),
        "empty": empty,
        "prompt": statistics.mean(prompts),
        "completion": statistics.mean(completions),
        "agent": statistics.mean(agent_costs),
        "rater": statistics.mean(rater_costs),
        "total": statistics.mean(total_costs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--run-ids", type=str, nargs="+", required=True, help="run ids")
    parser.add_argument("-b", "--benchmark", type=str, default="sweqapro", help="benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"{'run':40s} {'n':>4s} {'empty':>5s} {'prompt/q':>10s} {'compl/q':>8s} {'agent ¢':>8s} {'rater ¢':>8s} {'total ¢':>8s}")
    for run_id in args.run_ids:
        row = summarize(run_id, args.benchmark)
        print(
            f"{run_id:40s} {row['n']:4d} {row['empty']:5d} {row['prompt']:10.0f} {row['completion']:8.0f} "
            f"{100 * row['agent']:8.2f} {100 * row['rater']:8.2f} {100 * row['total']:8.2f}"
        )


if __name__ == "__main__":
    main()
