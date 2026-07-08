# PLAN.md — AgentAudit: from v0.1.0 to shipped

Each stage has a visible endpoint. Do not start a stage until the previous
stage's endpoint is verified. Any Claude Code session (or human) should be able
to pick up the next unchecked stage cold.

## Stage 0 — Done (this build)

- [x] Trace data model with validation (`agentaudit/trace.py`)
- [x] Loop detection + per-agent cost accounting (`agentaudit/detectors.py`)
- [x] Judge protocol with HeuristicJudge (offline) and AnthropicJudge (API) (`agentaudit/judge.py`)
- [x] Handoff scoring + backward failure attribution (`agentaudit/audit.py`)
- [x] Markdown report + CLI (`agentaudit/report.py`, `agentaudit/cli.py`)
- [x] 10 passing offline tests, demo trace with planted defect and loop

Endpoint (verified): `pytest` green; `agentaudit examples/demo_trace.json --failure "..."` blames s2/researcher, flags the analyst loop, warns on 69% spend share.

## Stage 1 — Validate the LLM judge (needs ANTHROPIC_API_KEY, ~$1 of usage) — Done

- [x] Run the demo: `agentaudit examples/demo_trace.json --judge anthropic --failure "equipment revenue wrong, total transposed to 12.2 instead of 8.6"`
- [x] Confirm attribution still lands on s2 and handoff rationales are sensible
- [x] Create 3 more example traces with different failure types: context dropped at a handoff (`examples/context_dropped_trace.json`), tool output misread (`examples/tool_misread_trace.json`), instruction ignored (`examples/instruction_ignored_trace.json`). Confirm the judge attributes each correctly; tune the two prompts in `judge.py` if not
- [x] Add `tests/test_anthropic_judge.py` gated behind `@pytest.mark.skipif(no key)`

Endpoint (verified 2026-07-08): all 4 traces attributed correctly by AnthropicJudge on the
first run — no prompt tuning needed. The judge correctly exonerated upstream steps that
merely *contained* the failure's numeric signature (e.g. raw tool output showing 1200)
from steps that *asserted* the wrong claim, something the heuristic judge's numeric-match
approach can't distinguish. Bisect (Stage 3) also exercised correctly against the live
judge: 4 judge calls to attribute the 6-step demo trace instead of 6.

## Stage 2 — Framework adapters (the adoption feature)

- [x] `agentaudit/adapters/langgraph.py`: convert a LangGraph run (list of state snapshots / events) into a Trace
- [x] `agentaudit/adapters/crewai.py`: same for CrewAI task outputs
- [x] One test per adapter using a recorded fixture (JSON of a real run) — no live framework calls in CI
- [x] README section: "Auditing a LangGraph run in 5 lines"

Endpoint: a recorded LangGraph fixture round-trips through `audit()` and renders a report.

## Stage 3 — Bisect attribution + JSON output

- [x] Replace linear scan in `attribute_failure` with binary search over steps when using the LLM judge (defect presence is monotone once introduced; verify this assumption per trace by checking endpoints first, fall back to linear if not monotone)
- [x] `--format json` on the CLI for programmatic consumption / CI gates
- [x] `--fail-on` flag (e.g. `--fail-on loops,handoff<3`) so it can gate a GitHub Action, mirroring the PromptOps pattern

Endpoint: bisect cuts judge calls on a 20-step trace from 20 to ≤6; CI gate demo in README.

## Stage 4 — Publish

- [x] Push to GitHub under vineetha00 with MIT license, CI workflow running the offline tests (https://github.com/vineetha00/agentaudit, CI green)
- [x] Record a demo GIF: pipeline fails → agentaudit blames the right agent (`examples/demo.gif`, regenerate with `scripts/make_demo_gif.py`)
- [x] PyPI release — `agentaudit` and the plan's fallback `agent-audit` are both blocked by PyPI's name-similarity rule (they normalize to the same "ultranormalized" name and `agent-audit` is already registered to someone else). Registered as `agentaudit-eval` instead; the `agentaudit` CLI command and Python import are unaffected since `[project.name]` and `[project.scripts]` are independent. Live at https://pypi.org/project/agentaudit-eval/0.2.0/, verified with a clean-venv install (2026-07-08)
- [ ] Resume bullet + portfolio entry; pairs with PromptOps as the "production LLM reliability" story

Endpoint (verified 2026-07-08): `pip install agentaudit-eval` works from a clean machine and installs the `agentaudit` CLI command; README GIF renders on GitHub.

## Explicitly out of scope for v1

Live instrumentation/tracing SDK (OpenTelemetry-style) — export-to-JSON is the
contract. Multi-trace statistical evals. A web UI. All of these are v2 territory
and should not block publishing.
