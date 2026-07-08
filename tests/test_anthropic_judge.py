"""Live validation of AnthropicJudge attribution across four failure types.

Requires ANTHROPIC_API_KEY and makes real API calls (small $ cost). Skipped
entirely when the key isn't set, so it never runs in offline/CI environments.

Each case pairs an example trace with a failure description and the step
where the defect should be attributed. See PLAN.md Stage 1: the four failure
types are numeric transposition (demo_trace), context dropped at a handoff,
tool output misread, and an explicit instruction ignored.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentaudit import AnthropicJudge, Trace, attribute_failure

EXAMPLES = Path(__file__).parent.parent / "examples"

no_key = not os.environ.get("ANTHROPIC_API_KEY")

CASES = [
    (
        "demo_trace.json",
        "equipment revenue wrong, total transposed to 12.2 instead of 8.6",
        "s2",
        "researcher",
    ),
    (
        "context_dropped_trace.json",
        "contractor expense was wrongly auto-approved even though there was no prior "
        "approval on file for travel, which should have required manual review under policy 4.2",
        "s2",
        "policy_checker",
    ),
    (
        "tool_misread_trace.json",
        "SKU-118 available-to-promise count is wrong, reported as 1200 instead of the correct 250 available units",
        "s2",
        "analyzer",
    ),
    (
        "instruction_ignored_trace.json",
        "Q3 closed-won revenue total is wrong, reported as 842000 instead of the correct "
        "692000 -- a Q4 forecasted deal was included despite explicit instructions to exclude it",
        "s2",
        "analyst",
    ),
]


@pytest.mark.skipif(no_key, reason="ANTHROPIC_API_KEY not set")
@pytest.mark.parametrize("filename,failure,expected_step,expected_agent", CASES)
def test_anthropic_judge_attributes_correctly(filename, failure, expected_step, expected_agent):
    trace = Trace.load(EXAMPLES / filename)
    result = attribute_failure(trace, AnthropicJudge(), failure, strategy="auto")
    assert result.origin_step_id == expected_step
    assert result.origin_agent == expected_agent
