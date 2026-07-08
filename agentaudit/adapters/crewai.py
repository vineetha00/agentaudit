"""Convert a CrewAI crew run into an AgentAudit Trace.

Works on the dict form of a ``CrewOutput`` (``crew.kickoff().model_dump()``
or recorded JSON): ``tasks_output`` becomes the steps, ``raw`` the final
output. Accepts a live ``CrewOutput`` object as well.
"""

from __future__ import annotations

from typing import Any, Optional

from ..trace import Step, Trace


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def from_crewai(
    crew_output: Any,
    workflow: str = "crewai_run",
    task: str = "",
    expected_output: Optional[str] = None,
) -> Trace:
    """Build a Trace from a CrewAI ``CrewOutput`` (object or dict).

    Each task output becomes a step: the task description is the step input,
    the raw result the step output. CrewAI reports token usage only at the
    crew level, so totals are split evenly across steps and each step is
    marked ``metadata["tokens_estimated"] = True``.
    """
    tasks = _get(crew_output, "tasks_output") or []
    usage = _get(crew_output, "token_usage") or {}
    n = len(tasks)
    tokens_in_each, tokens_out_each = 0, 0
    if n:
        tokens_in_each = int(_get(usage, "prompt_tokens", 0) or 0) // n
        tokens_out_each = int(_get(usage, "completion_tokens", 0) or 0) // n
    estimated = bool(usage) and n > 0

    steps = []
    for i, t in enumerate(tasks, start=1):
        name = _get(t, "name")
        steps.append(
            Step(
                id=str(name) if name else f"s{i}",
                agent=str(_get(t, "agent", "unknown")),
                input=str(_get(t, "description", "") or ""),
                output=str(_get(t, "raw", "") or ""),
                tokens_in=tokens_in_each,
                tokens_out=tokens_out_each,
                metadata={"tokens_estimated": True} if estimated else {},
            )
        )
    return Trace(
        workflow=workflow,
        steps=steps,
        final_output=str(_get(crew_output, "raw", "") or (steps[-1].output if steps else "")),
        expected_output=expected_output,
        task=task,
    )
