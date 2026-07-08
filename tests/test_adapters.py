import json
from pathlib import Path

import pytest

from agentaudit import HeuristicJudge, audit
from agentaudit.adapters import from_crewai, from_langgraph
from agentaudit.report import render_markdown

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def langgraph_events():
    return json.loads((FIXTURES / "langgraph_run.json").read_text())


@pytest.fixture
def crewai_output():
    return json.loads((FIXTURES / "crewai_run.json").read_text())


def test_langgraph_fixture_roundtrips_through_audit(langgraph_events):
    trace = from_langgraph(
        langgraph_events,
        workflow="pricing_brief",
        task="Summarize competitor pricing for the Starter, Pro, and Enterprise tiers.",
    )
    assert trace.agents == ["planner", "researcher", "analyst", "writer"]
    assert len(trace.steps) == 4  # __start__ skipped
    assert trace.steps[0].input == trace.task
    assert trace.steps[1].model == "claude-haiku-4-5"
    assert trace.steps[1].tokens_in == 1200
    assert "Starter $49/mo" in trace.final_output  # content-blocks message parsed

    report = audit(
        trace,
        HeuristicJudge(),
        failure_description="Starter tier price wrong, reported 49 instead of 29",
    )
    assert report.attribution.origin_step_id == "s2"
    assert report.attribution.origin_agent == "researcher"
    md = render_markdown(report)
    assert "# AgentAudit report — pricing_brief" in md
    assert "Failure attribution" in md


def test_langgraph_handles_non_message_state():
    events = [{"scorer": {"score": 0.87}}]
    trace = from_langgraph(events)
    assert trace.steps[0].agent == "scorer"
    assert "0.87" in trace.steps[0].output


def test_crewai_fixture_roundtrips_through_audit(crewai_output):
    trace = from_crewai(
        crewai_output,
        workflow="churn_report",
        task="Report on Q3 churn by segment.",
    )
    assert trace.agents == ["Data Researcher", "Churn Analyst", "Report Writer"]
    assert [s.id for s in trace.steps] == ["research_churn", "analyze_churn", "write_report"]
    assert trace.final_output.startswith("Q3 churn report")
    # crew-level token usage split evenly and flagged as estimated
    assert all(s.tokens_in == 2100 and s.tokens_out == 700 for s in trace.steps)
    assert all(s.metadata["tokens_estimated"] for s in trace.steps)

    report = audit(trace, HeuristicJudge())
    assert len(report.handoffs) == 2
    md = render_markdown(report)
    assert "Churn Analyst" in md


def test_crewai_without_token_usage():
    trace = from_crewai({"raw": "done", "tasks_output": [
        {"description": "d", "raw": "out", "agent": "solo"},
    ]})
    assert trace.steps[0].id == "s1"
    assert trace.steps[0].tokens_in == 0
    assert trace.steps[0].metadata == {}
