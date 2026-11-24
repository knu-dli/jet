"""Calls: the number of tool calls the agent executed for each question. A failed question counts as the step
limit.
"""

from models import Result


def count_calls(result: Result) -> int:
    if result.status != "success":
        return result.max_iterations - 1
    total = 0
    for turn in result.trajectory:  # 턴마다 그 턴에서 실행한 도구 호출 목록. 답을 쓴 턴은 빈 목록
        total += len(turn.tool_calls)
    return total

