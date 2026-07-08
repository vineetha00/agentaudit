from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .trace import Trace

DEFAULT_PRICING_PER_MTOK = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-8": (15.00, 75.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "unknown": (3.00, 15.00),
}


@dataclass
class LoopFinding:
    agent: str
    step_ids: list[str]
    similarity: float

    @property
    def description(self) -> str:
        return (
            f"agent '{self.agent}' produced {len(self.step_ids)} near-identical consecutive outputs "
            f"(steps {', '.join(self.step_ids)}, similarity {self.similarity:.0%}) — likely a degenerate retry loop"
        )


def detect_loops(trace: Trace, min_repeats: int = 3, similarity_threshold: float = 0.90) -> list[LoopFinding]:
    findings: list[LoopFinding] = []
    run: list = []

    def flush():
        if len(run) >= min_repeats:
            sims = [
                SequenceMatcher(None, a.output, b.output).ratio()
                for a, b in zip(run, run[1:])
            ]
            findings.append(
                LoopFinding(
                    agent=run[0].agent,
                    step_ids=[s.id for s in run],
                    similarity=min(sims) if sims else 1.0,
                )
            )

    for step in trace.steps:
        if run and step.agent == run[-1].agent:
            sim = SequenceMatcher(None, run[-1].output, step.output).ratio()
            if sim >= similarity_threshold:
                run.append(step)
                continue
        flush()
        run = [step]
    flush()
    return findings


@dataclass
class AgentCost:
    agent: str
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    steps: int = 0


@dataclass
class CostSummary:
    per_agent: dict[str, AgentCost] = field(default_factory=dict)
    total_usd: float = 0.0
    warnings: list[str] = field(default_factory=list)


def cost_summary(
    trace: Trace,
    pricing: dict[str, tuple[float, float]] | None = None,
    budget_share_warn: float = 0.6,
) -> CostSummary:
    pricing = pricing or DEFAULT_PRICING_PER_MTOK
    summary = CostSummary()
    for step in trace.steps:
        rate_in, rate_out = pricing.get(step.model, pricing["unknown"])
        usd = step.tokens_in / 1e6 * rate_in + step.tokens_out / 1e6 * rate_out
        entry = summary.per_agent.setdefault(step.agent, AgentCost(agent=step.agent))
        entry.tokens_in += step.tokens_in
        entry.tokens_out += step.tokens_out
        entry.usd += usd
        entry.steps += 1
        summary.total_usd += usd
    if summary.total_usd > 0:
        for entry in summary.per_agent.values():
            share = entry.usd / summary.total_usd
            if share >= budget_share_warn:
                summary.warnings.append(
                    f"agent '{entry.agent}' consumed {share:.0%} of total spend "
                    f"(${entry.usd:.4f} of ${summary.total_usd:.4f}) — check for retry loops or oversized context"
                )
    return summary
