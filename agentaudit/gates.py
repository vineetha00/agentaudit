"""CI gate conditions for --fail-on.

A spec is a comma-separated list of conditions, e.g. "loops,handoff<3":

- ``loops``       — fail if any degenerate loop was detected
- ``structural``  — fail if any structural issue was found
- ``handoff<N``   — fail if any handoff scored below N (1-5)
- ``cost>X``      — fail if total spend exceeds $X
- ``attribution`` — fail if failure attribution found an origin step
"""

from __future__ import annotations

import re

from .audit import AuditReport


def evaluate_gates(report: AuditReport, spec: str) -> list[str]:
    """Return one message per tripped gate; empty list means all gates pass."""
    failures: list[str] = []
    for cond in spec.split(","):
        cond = cond.strip()
        if not cond:
            continue
        if cond == "loops":
            for f in report.loops:
                failures.append(f"loops: {f.description}")
        elif cond == "structural":
            for issue in report.structural_issues:
                failures.append(f"structural: {issue}")
        elif m := re.fullmatch(r"handoff<(\d+)", cond):
            threshold = int(m.group(1))
            for hr in report.handoffs:
                if hr.score.score < threshold:
                    failures.append(
                        f"handoff<{threshold}: {hr.handoff.label} scored {hr.score.score}/5 — {hr.score.rationale}"
                    )
        elif m := re.fullmatch(r"cost>(\d+(?:\.\d+)?)", cond):
            limit = float(m.group(1))
            if report.costs.total_usd > limit:
                failures.append(
                    f"cost>{limit}: total spend ${report.costs.total_usd:.4f} exceeds ${limit:.4f}"
                )
        elif cond == "attribution":
            if report.attribution is not None and report.attribution.origin_step_id is not None:
                failures.append(f"attribution: {report.attribution.description}")
        else:
            raise ValueError(
                f"unknown --fail-on condition '{cond}' "
                "(expected: loops, structural, handoff<N, cost>X, attribution)"
            )
    return failures
