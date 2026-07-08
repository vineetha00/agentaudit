import json
from pathlib import Path

import pytest

from agentaudit import HeuristicJudge, Trace, attribute_failure, audit, evaluate_gates
from agentaudit.cli import main

DEMO = Path(__file__).parent.parent / "examples" / "demo_trace.json"


class CountingJudge(HeuristicJudge):
    def __init__(self):
        self.defect_calls = 0

    def check_defect(self, step, failure_description):
        self.defect_calls += 1
        return super().check_defect(step, failure_description)


def monotone_trace(n=20, defect_at=7):
    """Defect introduced at step index `defect_at` and propagated downstream."""
    steps = []
    for i in range(n):
        marker = " [defect]" if i >= defect_at else ""
        steps.append({
            "id": f"s{i}",
            "agent": f"agent{i % 4}",
            "output": f"step {i} distinct payload {'x' * (i % 5)}{marker}",
        })
    return Trace.from_dict({"workflow": "long", "steps": steps})


@pytest.mark.parametrize("defect_at", [0, 7, 13, 19])
def test_bisect_finds_origin_with_few_calls(defect_at):
    trace = monotone_trace(20, defect_at)
    judge = CountingJudge()
    result = attribute_failure(trace, judge, "anything", strategy="bisect")
    assert result.origin_step_id == f"s{defect_at}"
    assert judge.defect_calls <= 6  # plan endpoint: 20-step trace, ≤6 judge calls


def test_bisect_matches_linear_on_all_origins():
    for defect_at in range(20):
        trace = monotone_trace(20, defect_at)
        bisect = attribute_failure(trace, CountingJudge(), "anything", strategy="bisect")
        linear = attribute_failure(trace, CountingJudge(), "anything", strategy="linear")
        assert bisect.origin_step_id == linear.origin_step_id


def test_bisect_falls_back_to_linear_when_not_monotone():
    # defect appears mid-trace but is fixed before the end: last step is clean,
    # so bisect's monotonicity check fails and the scan must go linear
    steps = [
        {"id": f"s{i}", "agent": "a", "output": f"payload {i} {'[defect]' if i == 4 else ''} {'y' * i}"}
        for i in range(10)
    ]
    trace = Trace.from_dict({"workflow": "x", "steps": steps})
    judge = CountingJudge()
    result = attribute_failure(trace, judge, "anything", strategy="bisect")
    assert result.origin_step_id == "s4"
    assert judge.defect_calls == 10  # full linear scan after the endpoint check


def test_auto_strategy_stays_linear_for_heuristic_judge():
    trace = monotone_trace(8, 3)
    judge = CountingJudge()
    report = audit(trace, judge, failure_description="anything", handoff_scoring=False)
    assert report.attribution.origin_step_id == "s3"
    assert judge.defect_calls == 8  # heuristic judge is free: judge every step


def test_json_format_output(capsys):
    assert main([str(DEMO), "--format", "json", "--failure", "total transposed to 12.2"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["workflow"] == "market_research_pipeline"
    assert data["attribution"]["origin_step_id"] == "s2"
    assert data["attribution"]["origin_agent"] == "researcher"
    assert len(data["loops"]) == 1
    assert data["costs"]["total_usd"] > 0
    assert len(data["handoffs"]) == 3


def test_fail_on_gate_trips_on_loops(capsys):
    assert main([str(DEMO), "--fail-on", "loops"]) == 1
    assert "FAIL loops:" in capsys.readouterr().err


def test_fail_on_gate_passes_clean_trace(tmp_path, capsys):
    clean = tmp_path / "clean.json"
    clean.write_text(json.dumps({"workflow": "ok", "steps": [
        {"id": "1", "agent": "a", "input": "do the thing", "output": "did the thing"},
    ]}))
    assert main([str(clean), "--fail-on", "loops,structural,cost>1"]) == 0
    assert "all gates passed" in capsys.readouterr().err


def test_fail_on_handoff_threshold():
    trace = Trace.from_dict({"workflow": "x", "steps": [
        {"id": "1", "agent": "a", "output": "alpha beta gamma delta"},
        {"id": "2", "agent": "b", "input": "totally unrelated words here", "output": "z"},
    ]})
    report = audit(trace, HeuristicJudge())
    failures = evaluate_gates(report, "handoff<3")
    assert len(failures) == 1
    assert "handoff<3" in failures[0]


def test_fail_on_rejects_unknown_condition():
    trace = Trace.from_dict({"workflow": "x", "steps": [{"id": "1", "agent": "a", "output": "o"}]})
    report = audit(trace, HeuristicJudge())
    with pytest.raises(ValueError):
        evaluate_gates(report, "bogus")
