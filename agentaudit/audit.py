from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .detectors import CostSummary, LoopFinding, cost_summary, detect_loops
from .judge import DefectVerdict, HandoffScore, HeuristicJudge, Judge
from .trace import Handoff, Trace


@dataclass
class HandoffResult:
    handoff: Handoff
    score: HandoffScore


@dataclass
class AttributionResult:
    origin_step_id: Optional[str]
    origin_agent: Optional[str]
    verdicts: list[tuple[str, DefectVerdict]] = field(default_factory=list)

    @property
    def description(self) -> str:
        if self.origin_step_id is None:
            return "no step's output contained the defect — the failure may originate outside the trace (task spec, tools, or aggregation)"
        return f"defect first appears in step {self.origin_step_id} (agent '{self.origin_agent}') — later agents propagated it"


@dataclass
class AuditReport:
    trace: Trace
    structural_issues: list[str]
    loops: list[LoopFinding]
    costs: CostSummary
    handoffs: list[HandoffResult] = field(default_factory=list)
    attribution: Optional[AttributionResult] = None


def score_handoffs(trace: Trace, judge: Judge) -> list[HandoffResult]:
    return [
        HandoffResult(handoff=h, score=judge.score_handoff(h, trace.task))
        for h in trace.handoffs()
    ]


def attribute_failure(
    trace: Trace,
    judge: Judge,
    failure_description: str,
    strategy: str = "linear",
) -> AttributionResult:
    """Find the earliest step whose output contains the defect.

    strategy:
      - "linear": judge every step in order. Always correct; O(n) judge calls.
      - "bisect": binary search for the origin — O(log n) judge calls. Valid
        only when defect presence is monotone (once introduced, it persists
        downstream). The final step is checked first to verify the defect is
        present at the end; if it isn't, monotonicity can't be assumed and
        the scan falls back to linear.
      - "auto": bisect for LLM judges (where each call costs money and time),
        linear otherwise.
    """
    if strategy == "auto":
        strategy = "linear" if isinstance(judge, HeuristicJudge) else "bisect"
    if strategy not in ("linear", "bisect"):
        raise ValueError(f"unknown attribution strategy '{strategy}'")

    steps = trace.steps
    checked: dict[int, DefectVerdict] = {}

    def check(i: int) -> DefectVerdict:
        if i not in checked:
            checked[i] = judge.check_defect(steps[i], failure_description)
        return checked[i]

    origin_idx: Optional[int] = None
    if strategy == "bisect" and len(steps) > 2 and check(len(steps) - 1).defective:
        lo, hi = 0, len(steps) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if check(mid).defective:
                hi = mid
            else:
                lo = mid + 1
        origin_idx = lo
    else:
        for i in range(len(steps)):
            if check(i).defective and origin_idx is None:
                origin_idx = i

    origin = steps[origin_idx] if origin_idx is not None else None
    verdicts = [(steps[i].id, v) for i, v in sorted(checked.items())]
    return AttributionResult(
        origin_step_id=origin.id if origin else None,
        origin_agent=origin.agent if origin else None,
        verdicts=verdicts,
    )


def audit(
    trace: Trace,
    judge: Judge,
    failure_description: str | None = None,
    handoff_scoring: bool = True,
    attribution_strategy: str = "auto",
) -> AuditReport:
    report = AuditReport(
        trace=trace,
        structural_issues=trace.structural_issues(),
        loops=detect_loops(trace),
        costs=cost_summary(trace),
    )
    if handoff_scoring:
        report.handoffs = score_handoffs(trace, judge)
    if failure_description:
        report.attribution = attribute_failure(
            trace, judge, failure_description, strategy=attribution_strategy
        )
    return report
