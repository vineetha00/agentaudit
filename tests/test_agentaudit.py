from pathlib import Path

import pytest

from agentaudit import HeuristicJudge, Trace, attribute_failure, audit, cost_summary, detect_loops
from agentaudit.report import render_markdown

DEMO = Path(__file__).parent.parent / "examples" / "demo_trace.json"


@pytest.fixture
def trace():
    return Trace.load(DEMO)


def test_trace_loads_with_expected_shape(trace):
    assert trace.workflow == "market_research_pipeline"
    assert len(trace.steps) == 6
    assert trace.agents == ["planner", "researcher", "analyst", "writer"]
    assert len(trace.handoffs()) == 3


def test_duplicate_step_ids_rejected():
    data = {"workflow": "x", "steps": [
        {"id": "a", "agent": "p", "output": "1"},
        {"id": "a", "agent": "p", "output": "2"},
    ]}
    with pytest.raises(ValueError):
        Trace.from_dict(data)


def test_loop_detection_finds_analyst_retry_loop(trace):
    loops = detect_loops(trace)
    assert len(loops) == 1
    assert loops[0].agent == "analyst"
    assert loops[0].step_ids == ["s3", "s4", "s5"]
    assert loops[0].similarity >= 0.9


def test_loop_detection_ignores_varied_outputs():
    data = {"workflow": "x", "steps": [
        {"id": "1", "agent": "a", "output": "Pulling raw sales figures from the warehouse export."},
        {"id": "2", "agent": "a", "output": "Cross-checking currency conversion rates against finance."},
        {"id": "3", "agent": "a", "output": "Aggregating regional subtotals into the master ledger."},
        {"id": "4", "agent": "a", "output": "Drafting the executive commentary on quarterly trends."},
    ]}
    assert detect_loops(Trace.from_dict(data)) == []


def test_cost_summary_totals_and_warns(trace):
    costs = cost_summary(trace)
    assert set(costs.per_agent) == {"planner", "researcher", "analyst", "writer"}
    analyst = costs.per_agent["analyst"]
    assert analyst.steps == 3
    assert analyst.tokens_in == 7500
    assert costs.total_usd > 0
    assert any("analyst" in w for w in costs.warnings)


def test_attribution_blames_researcher_not_writer(trace):
    result = attribute_failure(
        trace, HeuristicJudge(),
        failure_description="equipment revenue wrong, total transposed to 12.2 instead of 8.6",
    )
    assert result.origin_step_id == "s2"
    assert result.origin_agent == "researcher"
    defective_ids = [sid for sid, v in result.verdicts if v.defective]
    assert "s1" not in defective_ids


def test_attribution_handles_clean_trace():
    data = {"workflow": "x", "task": "t", "steps": [
        {"id": "1", "agent": "a", "output": "all good here"},
    ]}
    result = attribute_failure(Trace.from_dict(data), HeuristicJudge(), "missing quarterly numbers entirely")
    assert result.origin_step_id is None


def test_handoff_scores_in_range(trace):
    report = audit(trace, HeuristicJudge())
    assert len(report.handoffs) == 3
    for hr in report.handoffs:
        assert 1 <= hr.score.score <= 5


def test_structural_issue_on_mismatched_handoff():
    data = {"workflow": "x", "steps": [
        {"id": "1", "agent": "a", "output": "o", "handoff_to": "b"},
        {"id": "2", "agent": "c", "output": "o"},
    ]}
    issues = Trace.from_dict(data).structural_issues()
    assert any("declares handoff to 'b'" in i for i in issues)


def test_full_audit_renders_markdown(trace):
    report = audit(trace, HeuristicJudge(), failure_description="equipment revenue wrong, total 12.2")
    md = render_markdown(report)
    assert "# AgentAudit report" in md
    assert "Failure attribution" in md
    assert "s2" in md
