from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

from .trace import Handoff, Step


@dataclass
class HandoffScore:
    score: int
    rationale: str


@dataclass
class DefectVerdict:
    defective: bool
    rationale: str


class Judge(Protocol):
    def score_handoff(self, handoff: Handoff, task: str) -> HandoffScore: ...
    def check_defect(self, step: Step, failure_description: str) -> DefectVerdict: ...


class HeuristicJudge:
    """Deterministic judge for offline runs and tests.

    Handoff score is based on lexical overlap between what the sender produced
    and what the receiver was given. Defect checks look for an explicit defect
    marker or the failure description's key phrases inside the step output.
    """

    def score_handoff(self, handoff: Handoff, task: str) -> HandoffScore:
        produced = set(re.findall(r"\w+", handoff.from_step.output.lower()))
        received = set(re.findall(r"\w+", handoff.to_step.input.lower()))
        if not produced:
            return HandoffScore(1, "sender produced no content to hand off")
        overlap = len(produced & received) / len(produced)
        score = 1 + round(overlap * 4)
        return HandoffScore(
            score,
            f"{overlap:.0%} of the sender's output vocabulary reached the receiver's input",
        )

    def check_defect(self, step: Step, failure_description: str) -> DefectVerdict:
        text = step.output.lower()
        if "[defect]" in text:
            return DefectVerdict(True, "explicit defect marker present in output")
        desc = failure_description.lower()
        clean_markers = re.findall(r"(?:instead of|expected|should be)\s+([\d.,$]+)", desc)
        clean_numbers = {m.strip("$,.") for m in clean_markers}
        numbers = {n for n in re.findall(r"\d+(?:\.\d+)?", desc) if n not in clean_numbers}
        if numbers:
            hits = sorted(n for n in numbers if n in text)
            if hits:
                return DefectVerdict(True, f"output contains the failure's numeric signature: {', '.join(hits)}")
            return DefectVerdict(False, "failure's numeric signature absent from output")
        key_phrases = re.findall(r"\w{5,}", desc)
        hits = [p for p in key_phrases if p in text]
        if key_phrases and len(hits) > len(key_phrases) // 2:
            return DefectVerdict(True, f"output contains failure signature terms: {', '.join(hits[:5])}")
        return DefectVerdict(False, "no defect signature found in output")


class AnthropicJudge:
    """LLM judge backed by the Anthropic API. Requires ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set; use HeuristicJudge for offline runs")

    def _call(self, prompt: str) -> dict:
        import requests

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = "".join(b.get("text", "") for b in resp.json()["content"] if b.get("type") == "text")
        cleaned = re.sub(r"```(json)?", "", text).strip()
        return json.loads(cleaned)

    def score_handoff(self, handoff: Handoff, task: str) -> HandoffScore:
        prompt = (
            "You are auditing a multi-agent AI workflow.\n"
            f"Overall task: {task}\n\n"
            f"Agent '{handoff.from_step.agent}' produced this output:\n---\n{handoff.from_step.output}\n---\n\n"
            f"Agent '{handoff.to_step.agent}' then received this input:\n---\n{handoff.to_step.input}\n---\n\n"
            "Score how faithfully the sender's intent and necessary context were transferred, 1 (context lost) "
            "to 5 (fully preserved). Respond with ONLY JSON: {\"score\": <1-5>, \"rationale\": \"<one sentence>\"}"
        )
        data = self._call(prompt)
        return HandoffScore(int(data["score"]), str(data["rationale"]))

    def check_defect(self, step: Step, failure_description: str) -> DefectVerdict:
        prompt = (
            "You are attributing a failure in a multi-agent AI workflow.\n"
            f"The workflow's final output was wrong in this way: {failure_description}\n\n"
            f"Here is the output of agent '{step.agent}' at step {step.id}:\n---\n{step.output}\n---\n\n"
            "Is the defect already present in THIS step's output (as opposed to being introduced later)? "
            "Respond with ONLY JSON: {\"defective\": true/false, \"rationale\": \"<one sentence>\"}"
        )
        data = self._call(prompt)
        return DefectVerdict(bool(data["defective"]), str(data["rationale"]))


def get_judge(name: str, model: str = "claude-sonnet-4-6") -> Judge:
    if name == "heuristic":
        return HeuristicJudge()
    if name == "anthropic":
        return AnthropicJudge(model=model)
    raise ValueError(f"unknown judge '{name}' (expected 'heuristic' or 'anthropic')")
