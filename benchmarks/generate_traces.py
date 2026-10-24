"""Generate the judge-validation benchmark: 60 multi-agent traces with planted failures.

Every trace is built from a scenario template with a failure injected at a known
step, so each comes with ground-truth labels by construction:

  - the origin step (first step whose OUTPUT asserts the defect), and
  - a per-step defective/clean label (monotone: once asserted, downstream steps
    propagate it, which is also what makes bisect attribution valid).

Five failure families, matching the Stage-1 example traces plus two more:

  numeric_error        a figure is asserted wrong partway through the pipeline
  unit_error           right quantity, wrong magnitude/units ($8.6M -> $8,600)
  tool_misread         the tool output is correct; the next agent misreads it
  context_dropped      a handoff loses a constraint; the receiver decides wrong
  instruction_ignored  the task carries an explicit instruction a step violates

Traces deliberately include heuristic traps: in half of them, one PRE-origin
step's raw output incidentally contains the defect's numeric signature (an SKU,
a ticket id, an unrelated row) without asserting the wrong claim. A judge that
greps for numbers blames that step; a judge that reads for assertions does not.

Outputs (deterministic, seed=7):
  benchmarks/traces/<id>.json        60 trace files in the standard schema
  benchmarks/labels.json             construction labels (origin + per-step)
  benchmarks/label_sheet.csv         per-step sheet for independent human labels

Usage:  python benchmarks/generate_traces.py [--n-variants 4] [--seed 7]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

FAMILIES = ["numeric_error", "unit_error", "tool_misread", "context_dropped", "instruction_ignored"]

FIRST = ["Alex", "Priya", "Jordan", "Mina", "Diego", "Sam", "Noor", "Elena", "Ravi", "Kai"]
LAST = ["Kim", "Patel", "Alvarez", "Chen", "Okafor", "Novak", "Haddad", "Silva"]


def _person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def _transpose(num: str) -> str:
    """Swap two adjacent differing digits: 8.6 -> 6.8, 2450 -> 2540."""
    digits = [c for c in num]
    idxs = [i for i, c in enumerate(digits) if c.isdigit()]
    for a, b in zip(idxs, idxs[1:]):
        if digits[a] != digits[b]:
            digits[a], digits[b] = digits[b], digits[a]
            return "".join(digits)
    digits[idxs[-1]] = str((int(digits[idxs[-1]]) + 3) % 10)
    return "".join(digits)


def _steps(rows: list[tuple[str, str, str, str]]) -> list[dict]:
    steps = []
    for i, (agent, model, inp, out) in enumerate(rows):
        step = {
            "id": f"s{i + 1}",
            "agent": agent,
            "model": model,
            "input": inp,
            "output": out,
            "tokens_in": 180 + 40 * i,
            "tokens_out": 90 + 20 * i,
        }
        if i + 1 < len(rows):
            step["handoff_to"] = rows[i + 1][0]
        steps.append(step)
    return steps


# ---------------------------------------------------------------------------
# Scenario builders. Each returns (trace_dict, origin_idx) for a given family.
# `origin_pick` chooses among the scenario's legal injection points (0-based
# step indices); `trap` adds the pre-origin numeric distractor when possible.
# ---------------------------------------------------------------------------


def quarterly_finance(rng: random.Random, family: str, origin_pick: int, trap: bool):
    whole = rng.choice(["7", "8", "9", "11"])
    frac = rng.choice(["2", "4", "6"])
    good = f"{whole}.{frac}"
    bad = f"{frac}.{whole}" if family != "unit_error" else None
    bad_unit = f"${whole},{frac}00"  # $8,600 instead of $8.6M
    region = rng.choice(["EMEA", "APAC", "LATAM", "NA"])
    quarter = rng.choice(["Q2", "Q3", "Q4"])
    instruction = "Report every figure in USD millions with one decimal place."
    constraint = f"exclude the divested {rng.choice(['Lisbon', 'Osaka', 'Austin'])} unit from all totals"
    task = (
        f"Produce the {quarter} {region} equipment-revenue summary for the board. "
        f"{instruction} Per the controller's note, {constraint}."
    )

    good_claim = f"{quarter} {region} equipment revenue is ${good}M"
    if family == "numeric_error" or family == "tool_misread":
        bad_claim = f"{quarter} {region} equipment revenue is ${bad}M"
        failure = f"the report states equipment revenue of ${bad}M instead of ${good}M"
    elif family == "unit_error":
        bad_claim = f"{quarter} {region} equipment revenue is {bad_unit}"
        failure = f"the report states equipment revenue of {bad_unit} rather than ${good}M — the magnitude is off by three orders"
    elif family == "context_dropped":
        bad_claim = f"{quarter} {region} equipment revenue is ${good}M including all units"
        failure = f"the summary includes the divested unit the controller said to {constraint.split(' from')[0]}"
    else:  # instruction_ignored
        bad_claim = f"{quarter} {region} equipment revenue is {int(float(good) * 1000)} thousand USD"
        failure = "figures are reported in thousands of USD, violating the instruction to report in USD millions with one decimal"

    trap_value = bad if bad else f"{frac}.{whole}"
    trap_line = f" Unrelated rows: freight weight {trap_value} t on manifest M-{rng.randint(100, 999)}." if trap else ""
    raw = (
        f"ledger query [{quarter}/{region}]: equipment line items sum ${good}M; "
        f"divested-unit rows excluded per flag.{trap_line}"
    )

    candidates = {"numeric_error": [1, 2, 3], "unit_error": [1, 2, 3], "tool_misread": [1],
                  "context_dropped": [1, 2], "instruction_ignored": [1, 2, 3]}[family]
    origin = candidates[origin_pick % len(candidates)]

    def claim(i: int) -> str:
        return bad_claim if i >= origin else good_claim

    def con(i: int) -> str:
        if family == "context_dropped" and i >= origin:
            return ""
        return f" ({constraint})"

    rows = [
        ("data_puller", "tool", f"Pull {quarter} {region} equipment revenue from the ledger.{con(0)}", raw),
        ("analyst", "claude-haiku-4-5", f"Ledger data: {raw.split('. Unrelated')[0]}{con(1)}",
         f"Analysis: {claim(1)}{con(1)}. Growth vs prior quarter is within guidance."),
        ("fact_checker", "claude-haiku-4-5", f"Verify: {claim(1)}{con(2)}",
         f"Verified: {claim(2)}{con(2)}. Sources consistent."),
        ("summarizer", "claude-haiku-4-5", f"Draft board summary. Verified figure: {claim(2)}{con(3)}",
         f"Board summary: {claim(3)}. Outlook stable; no restatements this period."),
        ("publisher", "claude-haiku-4-5", f"Publish: {claim(3)}",
         f"Published to board portal: {claim(4)}. Distribution complete."),
    ]
    final = rows[-1][3]
    expected = f"Published to board portal: {good_claim}" + (f" ({constraint})" if family == "context_dropped" else "")
    return {
        "workflow": f"quarterly_finance_{region.lower()}",
        "task": task,
        "expected_output": expected,
        "final_output": final,
        "steps": _steps(rows),
    }, origin, failure


def support_triage(rng: random.Random, family: str, origin_pick: int, trap: bool):
    users = rng.choice(["1340", "2150", "1680", "3420"])
    bad = _transpose(users)
    tck = f"TCK-{rng.randint(4000, 9000)}"
    instruction = f"Include the ticket id {tck} in every message."
    constraint = "enterprise-tier tickets must page the on-call engineer"
    task = (
        f"Triage inbound ticket {tck} and route it with the right priority. "
        f"{instruction} Note: {constraint}."
    )

    good_claim = f"outage affects {users} users; priority P1, SLA 2h"
    if family in ("numeric_error", "tool_misread"):
        bad_claim = f"outage affects {bad} users; priority P2, SLA 8h"
        failure = f"the ticket was routed P2 for {bad} affected users, but the ticket shows {users} users, which is P1"
    elif family == "unit_error":
        bad_claim = f"outage affects {users} seats at one site; priority P3, SLA 24h"
        failure = f"the outage was scoped to a single site instead of {users} users fleet-wide, so it was routed P3 instead of P1"
    elif family == "context_dropped":
        bad_claim = f"outage affects {users} users; priority P1, SLA 2h, email queue only"
        failure = "the on-call engineer was never paged although the customer is enterprise tier"
    else:
        bad_claim = f"outage affects {users} users; priority P1, SLA 2h"
        failure = f"messages omit the ticket id {tck}, violating the triage instruction"

    trap_line = f" Related closed ticket TCK-{bad} (hardware, last year)." if trap and family in ("numeric_error", "tool_misread") else ""
    raw = (
        f"ticket {tck} [enterprise tier]: login outage reported, {users} affected users "
        f"across all regions, started 09:12 UTC.{trap_line}"
    )

    origin = {"numeric_error": [1, 2], "unit_error": [1, 2], "tool_misread": [1],
              "context_dropped": [2], "instruction_ignored": [1, 2, 3]}[family]
    origin = origin[origin_pick % len(origin)]

    def claim(i: int) -> str:
        return bad_claim if i >= origin else good_claim

    def tid(i: int) -> str:
        if family == "instruction_ignored" and i >= origin:
            return ""
        return f" [{tck}]"

    def page(i: int) -> str:
        if family == "context_dropped":
            return "" if i >= origin else " Enterprise tier: page on-call."
        return " Enterprise tier: page on-call." if i <= 1 else ""

    rows = [
        ("ticket_reader", "tool", f"Fetch ticket {tck}.", raw),
        ("classifier", "claude-haiku-4-5", f"Classify severity.{tid(1)} Ticket: {raw.split('. Related')[0]}.{page(1)}",
         f"Classification:{tid(1)} {claim(1)}.{page(1)}"),
        ("sla_checker", "claude-haiku-4-5", f"Confirm SLA.{tid(2)} {claim(1)}.{page(2)}",
         f"SLA check:{tid(2)} {claim(2)}.{page(2)}"),
        ("router", "claude-haiku-4-5", f"Route.{tid(3)} {claim(2)}.{page(3)}",
         f"Routed:{tid(3)} {claim(3)}. Queue assignment done.{page(3)}"),
        ("responder", "claude-haiku-4-5", f"Notify customer.{tid(4)} {claim(3)}.",
         f"Customer notified:{tid(4)} {claim(4)}. Status page updated."),
    ]
    final = rows[-1][3]
    expected = f"Customer notified: [{tck}] {good_claim}. On-call paged (enterprise tier). Status page updated."
    return {
        "workflow": "support_triage",
        "task": task,
        "expected_output": expected,
        "final_output": final,
        "steps": _steps(rows),
    }, origin, failure


def inventory_restock(rng: random.Random, family: str, origin_pick: int, trap: bool):
    qty = rng.choice(["2450", "1830", "3260", "1520"])
    bad = _transpose(qty)
    sku = f"SKU-{rng.randint(10000, 99999)}"
    supplier = rng.choice(["Northstar Supply", "Vexel Logistics", "Harbor & Co"])
    person = _person(rng)
    instruction = "Round every order quantity to full cases of 24 units."
    constraint = f"orders above $50,000 need sign-off from {person} before purchase"
    task = (
        f"Decide the restock order for {sku} and place it with {supplier}. "
        f"{instruction} Finance rule: {constraint}."
    )

    good_claim = f"reorder {qty} units of {sku}"
    if family in ("numeric_error", "tool_misread"):
        bad_claim = f"reorder {bad} units of {sku}"
        failure = f"the purchase order was placed for {bad} units instead of {qty}"
    elif family == "unit_error":
        bad_claim = f"reorder {qty} cases of {sku}"
        failure = f"the order was placed for {qty} cases (24 units each) instead of {qty} units — 24x too much stock"
    elif family == "context_dropped":
        bad_claim = f"reorder {qty} units of {sku}, auto-approved"
        failure = f"a purchase above $50,000 was placed without {person}'s sign-off"
    else:
        odd = str(int(qty) + 7)
        bad_claim = f"reorder {odd} units of {sku}"
        failure = f"the order quantity {odd} is not a multiple of a 24-unit case, violating the case-rounding instruction"

    trap_line = f" Aisle ref A-{bad}." if trap and family in ("numeric_error", "tool_misread") else ""
    raw = (
        f"stock scan {sku}: on-hand 312, 30-day demand forecast {qty} units, "
        f"supplier {supplier}, projected PO value $58,900.{trap_line}"
    )

    origin = {"numeric_error": [1, 2, 3], "unit_error": [1, 2], "tool_misread": [1],
              "context_dropped": [2, 3], "instruction_ignored": [1, 2]}[family]
    origin = origin[origin_pick % len(origin)]

    def claim(i: int) -> str:
        return bad_claim if i >= origin else good_claim

    def con(i: int) -> str:
        if family == "context_dropped" and i >= origin:
            return ""
        return f" PO value $58,900: requires {person}'s sign-off."

    rows = [
        ("stock_scanner", "tool", f"Scan stock level for {sku}.", raw),
        ("demand_forecaster", "claude-haiku-4-5", f"Forecast reorder from scan: {raw.split('. Aisle')[0]}.{con(1)}",
         f"Forecast: {claim(1)}.{con(1)}"),
        ("policy_checker", "claude-haiku-4-5", f"Check purchasing policy: {claim(1)}.{con(2)}",
         f"Policy check passed: {claim(2)}.{con(2)}"),
        ("purchaser", "claude-haiku-4-5", f"Place order: {claim(2)}.{con(3)}",
         f"PO issued to {supplier}: {claim(3)}.{con(3)}"),
        ("confirmer", "claude-haiku-4-5", f"Confirm: {claim(3)}.",
         f"Order confirmed with {supplier}: {claim(4)}. ETA 6 days."),
    ]
    final = rows[-1][3]
    expected = f"Order confirmed with {supplier}: {good_claim} after {person}'s sign-off. ETA 6 days."
    return {
        "workflow": "inventory_restock",
        "task": task,
        "expected_output": expected,
        "final_output": final,
        "steps": _steps(rows),
    }, origin, failure


SCENARIOS = [quarterly_finance, support_triage, inventory_restock]


def generate(out_dir: Path, seed: int = 7, n_variants: int = 4) -> dict:
    rng = random.Random(seed)
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    labels: dict = {}

    for scenario in SCENARIOS:
        for family in FAMILIES:
            for v in range(n_variants):
                trap = v % 2 == 1
                trace, origin, failure = scenario(rng, family, origin_pick=v, trap=trap)
                tid = f"{scenario.__name__}__{family}__v{v}"
                trace["metadata"] = {"benchmark_id": tid, "family": family, "trap": trap}
                (traces_dir / f"{tid}.json").write_text(json.dumps(trace, indent=1))
                labels[tid] = {
                    "family": family,
                    "trap": trap,
                    "failure_description": failure,
                    "origin_step_id": trace["steps"][origin]["id"],
                    "step_labels": {
                        s["id"]: (i >= origin) for i, s in enumerate(trace["steps"])
                    },
                }

    (out_dir / "labels.json").write_text(json.dumps(labels, indent=1))

    with (out_dir / "label_sheet.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trace_id", "failure_description", "step_id", "agent",
                    "output", "label_construction", "label_human", "notes"])
        for tid, lab in labels.items():
            trace = json.loads((traces_dir / f"{tid}.json").read_text())
            for s in trace["steps"]:
                w.writerow([tid, lab["failure_description"], s["id"], s["agent"],
                            s["output"], int(lab["step_labels"][s["id"]]), "", ""])
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-variants", type=int, default=4)
    ap.add_argument("--out", default=str(Path(__file__).parent))
    args = ap.parse_args()
    labels = generate(Path(args.out), seed=args.seed, n_variants=args.n_variants)
    fams: dict = {}
    for lab in labels.values():
        fams[lab["family"]] = fams.get(lab["family"], 0) + 1
    print(f"generated {len(labels)} traces: " + ", ".join(f"{k}={v}" for k, v in sorted(fams.items())))


if __name__ == "__main__":
    main()
