from __future__ import annotations

import json

from .audit import AuditReport


def render_markdown(report: AuditReport) -> str:
    t = report.trace
    lines = [
        f"# AgentAudit report — {t.workflow}",
        "",
        f"Agents: {', '.join(t.agents)} · Steps: {len(t.steps)} · Handoffs: {len(t.handoffs())}",
        "",
        "## Structural checks",
    ]
    if report.structural_issues:
        lines += [f"- ⚠️ {i}" for i in report.structural_issues]
    else:
        lines.append("- ✅ no structural issues")

    lines += ["", "## Loop detection"]
    if report.loops:
        lines += [f"- 🔁 {f.description}" for f in report.loops]
    else:
        lines.append("- ✅ no degenerate loops detected")

    lines += ["", "## Cost per agent", "", "| Agent | Steps | Tokens in | Tokens out | USD |", "|---|---|---|---|---|"]
    for entry in sorted(report.costs.per_agent.values(), key=lambda e: -e.usd):
        lines.append(
            f"| {entry.agent} | {entry.steps} | {entry.tokens_in:,} | {entry.tokens_out:,} | ${entry.usd:.4f} |"
        )
    lines.append(f"| **total** | | | | **${report.costs.total_usd:.4f}** |")
    for w in report.costs.warnings:
        lines.append(f"- 💸 {w}")

    if report.handoffs:
        lines += ["", "## Handoff quality", ""]
        for hr in report.handoffs:
            bar = "█" * hr.score.score + "░" * (5 - hr.score.score)
            lines.append(f"- `{bar}` {hr.score.score}/5 — {hr.handoff.label}: {hr.score.rationale}")

    if report.attribution is not None:
        lines += ["", "## Failure attribution", "", f"**{report.attribution.description}**", ""]
        for step_id, verdict in report.attribution.verdicts:
            mark = "❌" if verdict.defective else "✅"
            lines.append(f"- {mark} step {step_id}: {verdict.rationale}")

    return "\n".join(lines) + "\n"


def report_to_dict(report: AuditReport) -> dict:
    t = report.trace
    data: dict = {
        "workflow": t.workflow,
        "agents": t.agents,
        "step_count": len(t.steps),
        "structural_issues": list(report.structural_issues),
        "loops": [
            {"agent": f.agent, "step_ids": f.step_ids, "similarity": round(f.similarity, 4)}
            for f in report.loops
        ],
        "costs": {
            "total_usd": round(report.costs.total_usd, 6),
            "per_agent": {
                e.agent: {
                    "steps": e.steps,
                    "tokens_in": e.tokens_in,
                    "tokens_out": e.tokens_out,
                    "usd": round(e.usd, 6),
                }
                for e in report.costs.per_agent.values()
            },
            "warnings": list(report.costs.warnings),
        },
        "handoffs": [
            {
                "from_agent": hr.handoff.from_step.agent,
                "from_step": hr.handoff.from_step.id,
                "to_agent": hr.handoff.to_step.agent,
                "to_step": hr.handoff.to_step.id,
                "score": hr.score.score,
                "rationale": hr.score.rationale,
            }
            for hr in report.handoffs
        ],
        "attribution": None,
    }
    if report.attribution is not None:
        data["attribution"] = {
            "origin_step_id": report.attribution.origin_step_id,
            "origin_agent": report.attribution.origin_agent,
            "description": report.attribution.description,
            "verdicts": [
                {"step_id": sid, "defective": v.defective, "rationale": v.rationale}
                for sid, v in report.attribution.verdicts
            ],
        }
    return data


def render_json(report: AuditReport) -> str:
    return json.dumps(report_to_dict(report), indent=2) + "\n"
