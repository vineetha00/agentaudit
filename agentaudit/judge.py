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


def handoff_prompt(handoff: Handoff, task: str) -> str:
    return (
        "You are auditing a multi-agent AI workflow.\n"
        f"Overall task: {task}\n\n"
        f"Agent '{handoff.from_step.agent}' produced this output:\n---\n{handoff.from_step.output}\n---\n\n"
        f"Agent '{handoff.to_step.agent}' then received this input:\n---\n{handoff.to_step.input}\n---\n\n"
        "Score how faithfully the sender's intent and necessary context were transferred, 1 (context lost) "
        "to 5 (fully preserved). Respond with ONLY JSON: {\"score\": <1-5>, \"rationale\": \"<one sentence>\"}"
    )


def defect_prompt(step: Step, failure_description: str) -> str:
    return (
        "You are attributing a failure in a multi-agent AI workflow.\n"
        f"The workflow's final output was wrong in this way: {failure_description}\n\n"
        f"Here is the output of agent '{step.agent}' at step {step.id}:\n---\n{step.output}\n---\n\n"
        "Is the defect already present in THIS step's output (as opposed to being introduced later)? "
        "A step that merely contains raw data mentioning a number is not defective unless it asserts "
        "the wrong claim. Respond with ONLY JSON: {\"defective\": true/false, \"rationale\": \"<one sentence>\"}"
    )


def parse_judge_json(text: str) -> dict:
    """Extract the first JSON object from a model reply, tolerating code fences and chatter."""
    cleaned = re.sub(r"```(json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


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
        return parse_judge_json(text)

    def score_handoff(self, handoff: Handoff, task: str) -> HandoffScore:
        data = self._call(handoff_prompt(handoff, task))
        return HandoffScore(int(data["score"]), str(data["rationale"]))

    def check_defect(self, step: Step, failure_description: str) -> DefectVerdict:
        data = self._call(defect_prompt(step, failure_description))
        return DefectVerdict(bool(data["defective"]), str(data["rationale"]))


class OpenAICompatJudge:
    """LLM judge for any OpenAI-compatible endpoint: ollama, vLLM, llama.cpp, LM Studio.

    Defaults to a local ollama server, so the LLM judge can run fully offline:

        agentaudit trace.json --judge openai-compat --model llama3.1 --failure "..."

    Point it at a vLLM server with --judge-url http://host:8000/v1 (api_key via
    OPENAI_API_KEY or AGENTAUDIT_JUDGE_API_KEY when the endpoint requires one).
    """

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("AGENTAUDIT_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout

    def _call(self, prompt: str) -> dict:
        import requests

        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "temperature": 0,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return parse_judge_json(text)

    def score_handoff(self, handoff: Handoff, task: str) -> HandoffScore:
        data = self._call(handoff_prompt(handoff, task))
        return HandoffScore(int(data["score"]), str(data["rationale"]))

    def check_defect(self, step: Step, failure_description: str) -> DefectVerdict:
        data = self._call(defect_prompt(step, failure_description))
        return DefectVerdict(bool(data["defective"]), str(data["rationale"]))


def get_judge(name: str, model: str = "claude-sonnet-4-6", base_url: str | None = None) -> Judge:
    if name == "heuristic":
        return HeuristicJudge()
    if name == "anthropic":
        return AnthropicJudge(model=model)
    if name == "openai-compat":
        kwargs: dict = {}
        if model != "claude-sonnet-4-6":
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAICompatJudge(**kwargs)
    raise ValueError(f"unknown judge '{name}' (expected 'heuristic', 'anthropic', or 'openai-compat')")
