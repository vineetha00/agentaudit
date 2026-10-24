# AgentAudit

[![PyPI version](https://img.shields.io/pypi/v/agentaudit-eval.svg)](https://pypi.org/project/agentaudit-eval/)
[![CI](https://github.com/vineetha00/agentaudit/actions/workflows/ci.yml/badge.svg)](https://github.com/vineetha00/agentaudit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Evaluation framework for multi-agent AI workflows. You can evaluate a single LLM turn with a dozen tools; AgentAudit evaluates the *workflow* — did agents collaborate correctly, where did reasoning break down across handoffs, and which agent caused the failure.

Think `git blame`, but for agent pipelines.

![agentaudit demo](examples/demo.gif)

## What it does

- **Failure attribution** — given a description of what went wrong in the final output, walks the trace and pinpoints the earliest step where the defect appears. Downstream agents that merely propagated the error are exonerated. With the LLM judge, attribution bisects instead of scanning: ~6 judge calls on a 20-step trace instead of 20.
- **Handoff quality scoring** — scores every inter-agent handoff 1–5 on how faithfully the sender's intent and context reached the receiver, with a rationale per handoff.
- **Loop detection** — flags degenerate retry loops (an agent producing near-identical consecutive outputs), a confirmed production failure mode.
- **Cost guardrails** — per-agent token and dollar accounting from a pricing table, with warnings when one agent dominates spend.
- **Structural checks** — dangling or mismatched handoffs, empty traces.

## Quickstart

```bash
pip install agentaudit-eval
agentaudit examples/demo_trace.json \
  --failure "equipment revenue wrong, total transposed to 12.2 instead of 8.6"
```

The PyPI package is named `agentaudit-eval` (`agentaudit` and `agent-audit` were both taken), but the CLI command and Python import are both still just `agentaudit`. For local development, use `pip install -e .` instead.

The demo trace contains a planted defect (the researcher transposes a figure) and a planted retry loop (the analyst spins three times). The report blames the researcher at step s2, exonerates the planner, flags the loop, and warns that the analyst consumed 69% of total spend.

Add `-o report.md` to write the report to a file.

## Judges

Three judge backends decide handoff scores and defect verdicts:

- `--judge heuristic` (default) — deterministic, offline, free. Handoff scores from lexical overlap; defect checks from numeric failure signatures and keyword matching. Good for CI and tests; limited for subtle semantic failures.
- `--judge anthropic` — LLM judge via the Anthropic API (`ANTHROPIC_API_KEY` required). Semantic judgment of handoff fidelity and defect presence. Use `--model` to pick the judge model.
- `--judge openai-compat` — the same LLM judge against any OpenAI-compatible server: a local **ollama** (the default, `http://localhost:11434/v1`), **vLLM**, llama.cpp, or LM Studio. `--model` picks the served model, `--judge-url` the endpoint. Fully offline failure attribution with a local model: `agentaudit trace.json --judge openai-compat --model llama3.1 --failure "..."`.

The `Judge` protocol in `agentaudit/judge.py` is two methods; adding another backend is a small class.

### Judge validation

The judges are validated on a 60-trace benchmark (`benchmarks/`): 3 pipeline scenarios × 5 failure families (numeric error, unit/magnitude error, tool output misread, context dropped at a handoff, instruction ignored) × 4 variants, each with the failure planted at a known step, so every trace carries a ground-truth origin step and per-step defect labels. Half the traces contain a *trap*: a pre-origin step whose raw tool output incidentally contains the failure's numeric signature without asserting the wrong claim — the case that separates reading for assertions from grepping for numbers.

Results (full tables in [`benchmarks/VALIDATION.md`](benchmarks/VALIDATION.md)):

| Judge | Attribution acc. | Step agreement | Cohen's κ | Trap FPR |
|---|---|---|---|---|
| heuristic | 18.3% | 62.0% | 0.18 | 0.59 |
| Claude Haiku 5.5 | 45.0% | 80.3% | 0.55 | 0.35 |
| **Claude Sonnet 5.5** | **86.7%** | **96.7%** | **0.92** | **0.06** |
| llama3.1-8B (ollama) | 0.0% | 31.7% | 0.00 | 0.00 |

The Sonnet judge attributes 100% of numeric-error and tool-misread failures and misses mostly on omission-type failures (instruction ignored: 50%), the genuinely hard family. The trap column is why an LLM judge exists at all: the heuristic false-positives on 59% of steps that merely contain the failure's numbers, Sonnet on 6%. llama3.1-8B is an honest negative result — it answers "not defective" on every one of 300 steps under either prompt wording, so a small local model cannot currently be trusted as the attribution judge (the `openai-compat` backend still works fine against a served frontier-class model, e.g. via vLLM).

Reproduce with:

```bash
python benchmarks/generate_traces.py            # deterministic corpus, seed 7
python benchmarks/run_validation.py --judge heuristic
python benchmarks/run_validation.py --judge openai-compat --model llama3.1 --workers 3
python benchmarks/run_validation.py --judge anthropic --model claude-haiku-5-5   # or --judge claude-cli on a subscription
python benchmarks/run_validation.py --report    # writes benchmarks/VALIDATION.md
```

Construction labels are the default ground truth; `benchmarks/label_sheet.csv` is a ready-to-fill sheet for independent human labels (`--human-labels` rescoring, which also reports human-vs-construction agreement). The earlier four hand-built example traces (`examples/*.json`, `tests/test_anthropic_judge.py`) remain as smoke tests.

## Trace format

A trace is a JSON file: workflow name, task, final output, optional expected output, and an ordered list of steps. Each step records the agent, its input and output, model, token counts, and an optional declared `handoff_to`:

```json
{
  "workflow": "my_pipeline",
  "task": "what the workflow was asked to do",
  "final_output": "...",
  "steps": [
    {"id": "s1", "agent": "planner", "input": "...", "output": "...",
     "model": "claude-sonnet-4-6", "tokens_in": 350, "tokens_out": 120,
     "handoff_to": "researcher"}
  ]
}
```

Any framework's run can be exported to this shape — AutoGen, a hand-rolled orchestrator, or anything else. For LangGraph and CrewAI there are native adapters (below), no export step needed.

## Framework adapters

### Auditing a LangGraph run in 5 lines

```python
from agentaudit import HeuristicJudge, audit, render_markdown
from agentaudit.adapters import from_langgraph

events = list(graph.stream({"messages": [("user", task)]}, stream_mode="updates"))
trace = from_langgraph(events, task=task)
print(render_markdown(audit(trace, HeuristicJudge(), failure_description="what went wrong")))
```

`from_langgraph` consumes `stream_mode="updates"` events — live message objects or their serialized dict form, so a run recorded to JSON replays identically. Each graph node becomes an agent; token counts and model names are read from the messages' `usage_metadata` and `response_metadata`.

### CrewAI

```python
from agentaudit.adapters import from_crewai

result = crew.kickoff()
trace = from_crewai(result.model_dump(), task="...")
```

Each task output becomes a step (description in, raw result out). CrewAI only reports crew-level token usage, so totals are split evenly across steps and marked `tokens_estimated` in the step metadata.

## CI gates

`--format json` emits the full report as JSON for programmatic consumption, and `--fail-on` turns the audit into a gate — exit code 1 if any condition trips:

```bash
agentaudit trace.json --fail-on "loops,handoff<3,cost>0.50,structural"
```

Conditions: `loops` (degenerate retry loops), `structural` (dangling/mismatched handoffs), `handoff<N` (any handoff scored below N), `cost>X` (total spend over $X), `attribution` (a failure origin was found). As a GitHub Action step:

```yaml
- name: Audit agent pipeline trace
  run: |
    pip install agentaudit-eval
    agentaudit artifacts/last_run_trace.json --fail-on "loops,handoff<3" --format json -o audit.json
```

## Python API

```python
from agentaudit import Trace, HeuristicJudge, audit, render_markdown

trace = Trace.load("examples/demo_trace.json")
report = audit(trace, HeuristicJudge(), failure_description="total transposed to 12.2")
print(render_markdown(report))
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The test suite covers trace loading and validation, loop detection (positive and negative), cost accounting and budget warnings, handoff scoring bounds, failure attribution (blames origin, exonerates upstream, handles clean traces, bisect vs linear equivalence and call budgets), framework adapters against recorded fixtures, JSON output, CI gates, structural checks, and report rendering. All tests run offline via the heuristic judge — no API key or live framework calls in CI.

## Honest scope

This is v0.2.0. The deterministic layer (loops, cost, structure) is solid. The heuristic judge is intentionally simple — it exists so the pipeline runs offline and testably; it's good for CI and regression tests but limited on subtle semantic failures. The LLM judge has been validated against four distinct failure types (see Judges above) and is the recommended judge for real attribution work. Bisect attribution assumes defect presence is monotone once introduced; it verifies the defect is present at the final step before bisecting and falls back to a linear scan otherwise. See PLAN.md for remaining milestones.
