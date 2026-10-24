"""Run a judge over the benchmark corpus and score it against labels.

For every trace, the judge's check_defect runs on every step (linear scan, so
we get a verdict per step, not just the bisect path). Metrics:

  trace level   attribution accuracy: predicted origin step == labeled origin
  step level    precision / recall / F1 / Cohen's kappa on defective-vs-clean,
                plus the trap slice: false-positive rate on pre-origin steps
                that contain the defect's numeric signature without asserting it

Judges:
  heuristic     the deterministic lexical judge (free, instant)
  openai-compat any OpenAI-compatible server; default local ollama llama3.1
  anthropic     the Anthropic API judge (needs ANTHROPIC_API_KEY)
  claude-cli    same Claude models driven through the `claude` CLI, for running
                the validation on a Claude subscription without an API key

By default scores against the construction labels (benchmarks/labels.json).
Pass --human-labels benchmarks/label_sheet.csv once the label_human column is
filled in to score against human labels instead (and report human-vs-construction
agreement as a sanity check on the corpus itself).

Usage:
  python benchmarks/run_validation.py --judge heuristic
  python benchmarks/run_validation.py --judge openai-compat --model llama3.1 --workers 3
  python benchmarks/run_validation.py --judge claude-cli --model claude-haiku-5-5 --workers 6
  python benchmarks/run_validation.py --report   # combine results/*.json into VALIDATION.md
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentaudit.judge import DefectVerdict, defect_prompt, get_judge, parse_judge_json  # noqa: E402
from agentaudit.trace import Trace  # noqa: E402

BENCH = Path(__file__).resolve().parent


class ClaudeCLIJudge:
    """Drives the AnthropicJudge prompts through the `claude` CLI (-p mode).

    Lets the validation run against real Claude models on a Claude Code
    subscription, no ANTHROPIC_API_KEY required. Not part of the library
    surface — a benchmark utility only.
    """

    def __init__(self, model: str = "claude-haiku-5-5"):
        self.model = model

    def _call(self, prompt: str) -> dict:
        out = subprocess.run(
            ["claude", "-p", prompt, "--model", self.model],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if out.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {out.stderr[:200]}")
        return parse_judge_json(out.stdout)

    def check_defect(self, step, failure_description: str) -> DefectVerdict:
        data = self._call(defect_prompt(step, failure_description))
        return DefectVerdict(bool(data["defective"]), str(data["rationale"]))

    def score_handoff(self, handoff, task: str):  # pragma: no cover - unused here
        raise NotImplementedError


def build_judge(name: str, model: str | None, base_url: str | None):
    if name == "claude-cli":
        return ClaudeCLIJudge(model=model or "claude-haiku-5-5")
    kwargs = {}
    if model:
        kwargs["model"] = model
    return get_judge(name, base_url=base_url, **kwargs)


def load_labels(human_csv: str | None) -> tuple[dict, dict | None]:
    construction = json.loads((BENCH / "labels.json").read_text())
    if not human_csv:
        return construction, None
    human: dict = {}
    with open(human_csv) as f:
        for row in csv.DictReader(f):
            if row["label_human"].strip() == "":
                continue
            human.setdefault(row["trace_id"], {})[row["step_id"]] = row["label_human"].strip() in ("1", "true", "True")
    return construction, human


def judge_trace(judge, trace: Trace, failure: str) -> dict:
    verdicts = {}
    for step in trace.steps:
        for attempt in range(3):
            try:
                v = judge.check_defect(step, failure)
                break
            except Exception as e:  # transient model/parse errors
                if attempt == 2:
                    v = DefectVerdict(False, f"judge error after retries: {e}")
                time.sleep(1.5 * (attempt + 1))
        verdicts[step.id] = {"defective": v.defective, "rationale": v.rationale}
    origin = next((sid for sid, v in verdicts.items() if v["defective"]), None)
    return {"origin": origin, "verdicts": verdicts}


def kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    po = (tp + tn) / n
    pyes = ((tp + fp) / n) * ((tp + fn) / n)
    pno = ((fn + tn) / n) * ((fp + tn) / n)
    pe = pyes + pno
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(preds: dict, labels: dict, human: dict | None) -> dict:
    tp = fp = fn = tn = 0
    trap_fp = trap_n = 0
    trace_hits = 0
    by_family: dict = {}
    human_step_agree = human_steps = 0

    for tid, pred in preds.items():
        lab = labels[tid]
        fam = by_family.setdefault(lab["family"], {"traces": 0, "hits": 0})
        fam["traces"] += 1
        gold_steps = lab["step_labels"]
        if human and tid in human:
            gold_steps = {**gold_steps, **human[tid]}
        gold_origin = next((sid for sid, d in sorted(gold_steps.items()) if d), lab["origin_step_id"])
        if pred["origin"] == gold_origin:
            trace_hits += 1
            fam["hits"] += 1
        for sid, v in pred["verdicts"].items():
            g = gold_steps[sid]
            p = v["defective"]
            tp += p and g
            fp += p and not g
            fn += (not p) and g
            tn += (not p) and (not g)
            if human and tid in human and sid in human[tid]:
                human_steps += 1
                human_step_agree += human[tid][sid] == lab["step_labels"][sid]
            if lab["trap"] and not g:
                # pre-origin steps of trap traces: did the judge false-positive?
                trap_n += 1
                trap_fp += p

    n_steps = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "traces": len(preds),
        "attribution_accuracy": trace_hits / len(preds) if preds else 0.0,
        "step_level": {
            "n": n_steps,
            "agreement": (tp + tn) / n_steps if n_steps else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "cohens_kappa": kappa(tp, fp, fn, tn),
        },
        "trap_slice": {"pre_origin_steps": trap_n, "false_positive_rate": trap_fp / trap_n if trap_n else None},
        "by_family": {
            k: {"attribution_accuracy": v["hits"] / v["traces"], "traces": v["traces"]}
            for k, v in sorted(by_family.items())
        },
    }
    if human:
        out["human_labels"] = {
            "labeled_steps": human_steps,
            "human_vs_construction_agreement": human_step_agree / human_steps if human_steps else None,
        }
    return out


def run(args) -> None:
    labels, human = load_labels(args.human_labels)
    judge = build_judge(args.judge, args.model, args.judge_url)
    tids = sorted(labels.keys())
    if args.limit:
        tids = tids[: args.limit]

    t0 = time.time()
    preds: dict = {}

    def work(tid: str):
        trace = Trace.load(BENCH / "traces" / f"{tid}.json")
        return tid, judge_trace(judge, trace, labels[tid]["failure_description"])

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (tid, pred) in enumerate(pool.map(work, tids), 1):
            preds[tid] = pred
            print(f"[{i}/{len(tids)}] {tid}: origin={pred['origin']} (gold {labels[tid]['origin_step_id']})", flush=True)

    elapsed = time.time() - t0
    metrics = score(preds, labels, human)
    result = {
        "judge": args.judge,
        "model": args.model,
        "labels": "human" if human else "construction",
        "elapsed_s": round(elapsed, 1),
        "metrics": metrics,
        "predictions": preds,
    }
    out_dir = BENCH / "results"
    out_dir.mkdir(exist_ok=True)
    name = f"{args.judge}{'_' + args.model.replace(':', '-').replace('/', '-') if args.model else ''}"
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in metrics.items() if k != "by_family"}, indent=1))
    print(f"wrote {out_path} ({elapsed:.0f}s)")


def report() -> None:
    rows = []
    for p in sorted((BENCH / "results").glob("*.json")):
        r = json.loads(p.read_text())
        m = r["metrics"]
        s = m["step_level"]
        trap = m["trap_slice"]["false_positive_rate"]
        rows.append({
            "judge": f"{r['judge']}" + (f" ({r['model']})" if r.get("model") else ""),
            "labels": r.get("labels", "construction"),
            "traces": m["traces"],
            "attr": m["attribution_accuracy"],
            "agree": s["agreement"],
            "kappa": s["cohens_kappa"],
            "f1": s["f1"],
            "trap_fpr": trap,
            "by_family": m["by_family"],
        })
    lines = [
        "# Judge validation",
        "",
        "60 benchmark traces (3 scenarios x 5 failure families x 4 variants), each with a",
        "planted failure at a known step. Labels are by construction (`benchmarks/labels.json`);",
        "`benchmarks/label_sheet.csv` supports independent human relabeling via `--human-labels`.",
        "Generated by `benchmarks/generate_traces.py` (deterministic, seed 7).",
        "",
        "| Judge | Labels | Traces | Attribution acc. | Step agreement | Cohen's kappa | F1 | Trap FPR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        trap = f"{r['trap_fpr']:.2f}" if r["trap_fpr"] is not None else "-"
        lines.append(
            f"| {r['judge']} | {r['labels']} | {r['traces']} | {r['attr']:.1%} | {r['agree']:.1%} "
            f"| {r['kappa']:.2f} | {r['f1']:.2f} | {trap} |"
        )
    lines += ["", "Per failure family (attribution accuracy):", ""]
    fams = sorted({f for r in rows for f in r["by_family"]})
    lines.append("| Judge | " + " | ".join(fams) + " |")
    lines.append("|---|" + "---|" * len(fams))
    for r in rows:
        cells = [f"{r['by_family'][f]['attribution_accuracy']:.0%}" if f in r["by_family"] else "-" for f in fams]
        lines.append(f"| {r['judge']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "- **Attribution accuracy**: predicted origin step == labeled origin step (exact match).",
        "- **Step agreement / kappa / F1**: per-step defective-vs-clean verdicts against labels.",
        "- **Trap FPR**: false-positive rate on pre-origin steps that contain the failure's",
        "  numeric signature in raw tool output without asserting the wrong claim — the case",
        "  that separates reading for assertions from grepping for numbers.",
        "",
    ]
    (BENCH / "VALIDATION.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", choices=["heuristic", "openai-compat", "anthropic", "claude-cli"], default="heuristic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--judge-url", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--human-labels", default=None)
    ap.add_argument("--report", action="store_true", help="combine results/*.json into VALIDATION.md")
    args = ap.parse_args()
    if args.report:
        report()
    else:
        run(args)


if __name__ == "__main__":
    main()
