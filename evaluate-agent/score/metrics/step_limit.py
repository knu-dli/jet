"""Whether the scaffold forced the answer of a question at the step limit."""

from models import Result

STEP_LIMIT_REASON = "Forced after max iterations"
DEGENERATE_FORCED_ERROR = "Degenerate output detected (forced_final)"


def reached_step_limit(result: Result) -> bool:
    if (result.stop_reason or "").startswith(STEP_LIMIT_REASON):
        return True
    return (result.error or "").startswith(DEGENERATE_FORCED_ERROR)
