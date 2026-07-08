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

## Stage 1 — Validate the LLM judge (needs ANTHROPIC_API_KEY, ~$1 of usage)

- [ ] Run the demo: `agentaudit examples/demo_trace.json --judge anthropic --failure "equipment revenue wrong, total transposed to 12.2 instead of 8.6"`
- [ ] Confirm attribution still lands on s2 and handoff rationales are sensible
- [ ] Create 3 more example traces with different failure types: context dropped at a handoff, tool output misread, instruction ignored. Confirm the judge attributes each correctly; tune the two prompts in `judge.py` if not
- [ ] Add `tests/test_anthropic_judge.py` gated behind `@pytest.mark.skipif(no key)`

Endpoint: 4 traces, all attributed correctly by the LLM judge, prompts frozen.

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
- [ ] PyPI release (`pip install agentaudit`) — name confirmed free 2026-07-07; v0.2.0 sdist+wheel built and twine-checked; upload needs a PyPI API token: `python -m twine upload dist/*`
- [ ] Resume bullet + portfolio entry; pairs with PromptOps as the "production LLM reliability" story

Endpoint: `pip install agentaudit` works from a clean machine; README GIF renders on GitHub.

## Explicitly out of scope for v1

Live instrumentation/tracing SDK (OpenTelemetry-style) — export-to-JSON is the
contract. Multi-trace statistical evals. A web UI. All of these are v2 territory
and should not block publishing.
