# AgentAudit report — market_research_pipeline

Agents: planner, researcher, analyst, writer · Steps: 6 · Handoffs: 3

## Structural checks
- ✅ no structural issues

## Loop detection
- 🔁 agent 'analyst' produced 3 near-identical consecutive outputs (steps s3, s4, s5, similarity 100%) — likely a degenerate retry loop

## Cost per agent

| Agent | Steps | Tokens in | Tokens out | USD |
|---|---|---|---|---|
| analyst | 3 | 7,500 | 1,800 | $0.0165 |
| planner | 1 | 350 | 120 | $0.0029 |
| writer | 1 | 400 | 90 | $0.0026 |
| researcher | 1 | 900 | 210 | $0.0020 |
| **total** | | | | **$0.0238** |
- 💸 agent 'analyst' consumed 69% of total spend ($0.0165 of $0.0238) — check for retry loops or oversized context

## Handoff quality

- `██░░░` 2/5 — planner (s1) -> researcher (s2): 30% of the sender's output vocabulary reached the receiver's input
- `███░░` 3/5 — researcher (s2) -> analyst (s3): 53% of the sender's output vocabulary reached the receiver's input
- `███░░` 3/5 — analyst (s5) -> writer (s6): 47% of the sender's output vocabulary reached the receiver's input

## Failure attribution

**defect first appears in step s2 (agent 'researcher') — later agents propagated it**

- ✅ step s1: failure's numeric signature absent from output
- ❌ step s2: output contains the failure's numeric signature: 12.2
- ❌ step s3: output contains the failure's numeric signature: 12.2
- ❌ step s4: output contains the failure's numeric signature: 12.2
- ❌ step s5: output contains the failure's numeric signature: 12.2
- ❌ step s6: output contains the failure's numeric signature: 12.2
