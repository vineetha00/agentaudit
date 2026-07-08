from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .audit import audit
from .gates import evaluate_gates
from .judge import get_judge
from .report import render_json, render_markdown
from .trace import Trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentaudit",
        description="Evaluate a multi-agent workflow trace: handoffs, failure attribution, loops, cost.",
    )
    parser.add_argument("trace", help="path to a trace JSON file")
    parser.add_argument("--judge", choices=["heuristic", "anthropic"], default="heuristic")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="judge model (anthropic judge only)")
    parser.add_argument("--failure", default=None, help="describe the failure to run attribution")
    parser.add_argument(
        "--attribution",
        choices=["auto", "linear", "bisect"],
        default="auto",
        help="attribution strategy: bisect cuts LLM judge calls to O(log n) on monotone defects "
        "(auto = bisect for the anthropic judge, linear for heuristic)",
    )
    parser.add_argument("--no-handoffs", action="store_true", help="skip handoff scoring")
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="output format (json for programmatic consumption / CI)",
    )
    parser.add_argument(
        "--fail-on",
        default=None,
        metavar="SPEC",
        help="exit 1 if any condition trips, e.g. 'loops,handoff<3,cost>0.5,structural,attribution'",
    )
    parser.add_argument("-o", "--output", default=None, help="write the report to this path")
    args = parser.parse_args(argv)

    trace = Trace.load(args.trace)
    judge = get_judge(args.judge, model=args.model)
    report = audit(
        trace,
        judge,
        failure_description=args.failure,
        handoff_scoring=not args.no_handoffs,
        attribution_strategy=args.attribution,
    )
    rendered = render_json(report) if args.format == "json" else render_markdown(report)
    if args.output:
        Path(args.output).write_text(rendered)
        print(f"report written to {args.output}")
    else:
        print(rendered, end="")

    if args.fail_on:
        try:
            failures = evaluate_gates(report, args.fail_on)
        except ValueError as e:
            parser.error(str(e))
        if failures:
            for f in failures:
                print(f"FAIL {f}", file=sys.stderr)
            return 1
        print(f"all gates passed ({args.fail_on})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
