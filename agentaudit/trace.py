from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Step:
    id: str
    agent: str
    input: str
    output: str
    model: str = "unknown"
    tokens_in: int = 0
    tokens_out: int = 0
    handoff_to: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Handoff:
    from_step: Step
    to_step: Step

    @property
    def label(self) -> str:
        return f"{self.from_step.agent} ({self.from_step.id}) -> {self.to_step.agent} ({self.to_step.id})"


@dataclass
class Trace:
    workflow: str
    steps: list[Step]
    final_output: str = ""
    expected_output: Optional[str] = None
    task: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Trace":
        steps = [
            Step(
                id=s["id"],
                agent=s["agent"],
                input=s.get("input", ""),
                output=s.get("output", ""),
                model=s.get("model", "unknown"),
                tokens_in=int(s.get("tokens_in", 0)),
                tokens_out=int(s.get("tokens_out", 0)),
                handoff_to=s.get("handoff_to"),
                metadata=s.get("metadata", {}),
            )
            for s in data.get("steps", [])
        ]
        ids = [s.id for s in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate step ids in trace")
        trace = cls(
            workflow=data.get("workflow", "unnamed"),
            steps=steps,
            final_output=data.get("final_output", steps[-1].output if steps else ""),
            expected_output=data.get("expected_output"),
            task=data.get("task", ""),
        )
        return trace

    @property
    def agents(self) -> list[str]:
        seen: list[str] = []
        for s in self.steps:
            if s.agent not in seen:
                seen.append(s.agent)
        return seen

    def handoffs(self) -> list[Handoff]:
        result = []
        for a, b in zip(self.steps, self.steps[1:]):
            if b.agent != a.agent:
                result.append(Handoff(from_step=a, to_step=b))
        return result

    def structural_issues(self) -> list[str]:
        issues = []
        for i, s in enumerate(self.steps):
            if s.handoff_to is not None:
                nxt = self.steps[i + 1].agent if i + 1 < len(self.steps) else None
                if nxt is None:
                    issues.append(
                        f"step {s.id}: declares handoff to '{s.handoff_to}' but is the last step (dangling handoff)"
                    )
                elif nxt != s.handoff_to:
                    issues.append(
                        f"step {s.id}: declares handoff to '{s.handoff_to}' but next step ran '{nxt}'"
                    )
        if not self.steps:
            issues.append("trace contains no steps")
        return issues
