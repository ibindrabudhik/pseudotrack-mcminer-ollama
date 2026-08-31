#!/usr/bin/env python
"""Collect this bundle's metrics into readable tables.

Run 1 (--mode full) prints McMiner-S and McMiner-M accuracy per arm, split by
group type. The split matters: correct-only bags are at or near ceiling and add
a constant to every arm's pooled number, so a pooled comparison between arms is
mostly measuring how many correct bags there were.

Run 2 (--mode correct_only) prints the false-positive control on its own --
bag-level abstention next to per-code abstention, which is the comparison the
correct-only run exists to make.

Usage:
    python scripts/summarize.py --mode full
    python scripts/summarize.py --mode correct_only --arms baseline rag
"""
import argparse
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tag_for(model, arm, mode):
    slug = model.replace(":", "-").replace("/", "-")
    suffix = "_correctbags" if mode == "correct_only" else ""
    return "ollama_%s_%s%s" % (slug, arm, suffix)


def pct(x):
    return "%6.2f%%" % (x * 100)


def summarize_full(args):
    print("\n=== McMiner-S (per-code mining, aligned into bags) ===")
    print("%-9s %8s %10s %14s %14s %7s"
          % ("arm", "bags", "overall", "misconception", "correct-only", "judged"))
    print("-" * 68)
    for arm in args.arms:
        tag = tag_for(args.model, arm, "full")
        m = load("results/evaluations/%s/single_multi/evaluation_metrics.json" % tag)
        if not m:
            print("%-9s %s" % (arm, "(missing)"))
            continue
        o = m["standard_metrics"]["overall_metrics"]
        j = load("results/evaluations/%s/single_multi/judge_details_single.json" % tag) or {}
        js = j.get("summary", {})
        flag = "  <-- CHECK" if js.get("judge_parse_failures") else ""
        print("%-9s %8d %10s %14s %14s %7s%s"
              % (arm, o["total_bags"], pct(o["overall_accuracy"]),
                 pct(o["misconception_accuracy"]), pct(o["correct_only_accuracy"]),
                 js.get("judge_calls", 0), flag))

    print("\n=== McMiner-M (whole-bag mining) ===")
    print("%-9s %8s %10s %14s %14s %7s"
          % ("arm", "bags", "overall", "misconception", "correct-only", "judged"))
    print("-" * 68)
    for arm in args.arms:
        tag = tag_for(args.model, arm, "full")
        m = load("results/evaluations/%s/multi/evaluation_metrics.json" % tag)
        r = load("results/evaluations/%s/multi/claude_evaluation_results.json" % tag)
        if not m:
            print("%-9s %s" % (arm, "(missing)"))
            continue
        o = m["standard_metrics"]["overall_metrics"]
        s = (r or {}).get("summary", {})
        flag = "  <-- CHECK" if s.get("judge_parse_failures") else ""
        print("%-9s %8d %10s %14s %14s %7s%s"
              % (arm, o["total_bags"], pct(o["overall_accuracy"]),
                 pct(o["misconception_accuracy"]), pct(o["correct_only_accuracy"]),
                 s.get("judge_calls", 0), flag))

    print("\nRead the misconception column, not 'overall'. Correct-only bags sit at or")
    print("near ceiling and add the same constant to every arm, so the pooled number")
    print("compresses whatever difference the arms actually have.")
    print("'judged' is how many LLM judge calls fed that row; a CHECK flag means some")
    print("judge replies did not parse and those scores should not be trusted.")


def summarize_correct_only(args):
    print("\n=== correct-only bags: false-positive control ===")
    print("Ground truth is NONE everywhere. A bag/code 'matches' only when the miner")
    print("predicted no misconception at all. Zero judge calls -- both scorers decide")
    print("by rule (correct_bag_rule / empty_check), so these numbers do not depend")
    print("on the judge model.\n")

    print("%-9s %6s %14s %8s %14s"
          % ("arm", "bags", "bag abstention", "codes", "code abstention"))
    print("-" * 58)
    rows = []
    for arm in args.arms:
        tag = tag_for(args.model, arm, "correct_only")
        multi = load("results/evaluations/%s/multi/evaluation_metrics.json" % tag)
        grouped = load("results/%s/single_multi/grouped_predictions.json" % tag)
        singles = load("results/%s/single/predictions.json" % tag)
        if not multi:
            print("%-9s %s" % (arm, "(missing)"))
            continue
        o = multi["standard_metrics"]["overall_metrics"]

        # Per-code abstention over EVERY mined correct code, not only the ones
        # that landed in a bag. The bag former keeps one code per problem, so
        # the bagged subset is a fraction of what was mined.
        n_abstain = n_total = 0
        for p in singles or []:
            if (p.get("ground_truth_misconception") or {}).get("id") != "NONE":
                continue
            n_total += 1
            if p.get("no_predicted_misconceptions") or not (p.get("predicted_misconceptions") or []):
                n_abstain += 1
        code_rate = (n_abstain / n_total) if n_total else float("nan")
        rows.append((arm, o["correct_only_accuracy"], code_rate, n_total, grouped))
        print("%-9s %6d %14s %8d %14s"
              % (arm, o["correct_only_count"], pct(o["correct_only_accuracy"]),
                 n_total, pct(code_rate)))

    if not rows:
        return

    # Per-program view. The 96 correct files are far fewer distinct programs --
    # a problem whose misconception bank marked 12 entries "inapplicable"
    # contributes 12 rows for ONE program, so the per-row rate is weighted by an
    # artefact of dataset construction. Majority vote per program removes that.
    print("\n=== per-program view (removes the duplicate-row weighting) ===")
    print("%-9s %9s %12s %14s %s"
          % ("arm", "programs", "majority ok", "inconsistent", "(same program, different answers)"))
    print("-" * 78)
    for arm, _, _, _, _ in rows:
        tag = tag_for(args.model, arm, "correct_only")
        singles = load("results/%s/single/predictions.json" % tag) or []
        by_prog = defaultdict(list)
        for p in singles:
            if (p.get("ground_truth_misconception") or {}).get("id") != "NONE":
                continue
            abstained = bool(p.get("no_predicted_misconceptions")
                             or not (p.get("predicted_misconceptions") or []))
            by_prog[p.get("problem_id")].append(abstained)
        if not by_prog:
            continue
        maj = sum(1 for v in by_prog.values() if sum(v) * 2 > len(v))
        incons = sum(1 for v in by_prog.values() if len(set(v)) > 1)
        print("%-9s %9d %12s %14d"
              % (arm, len(by_prog), "%d/%d" % (maj, len(by_prog)), incons))
    print("\n'inconsistent' counts programs that got BOTH answers across their repeated")
    print("rows -- same program, same prompt, same temperature. That is model noise, and")
    print("it bounds how much of any arm-to-arm difference here is real.")

    print("\n=== bags vs codes ===")
    for arm, bag_rate, code_rate, n, _ in rows:
        if code_rate == code_rate:  # not NaN
            print("  %-9s bagged %s   individually %s   gap %+.1f pp"
                  % (arm, pct(bag_rate), pct(code_rate), (bag_rate - code_rate) * 100))
    print("\nA large positive gap means the model abstains far more readily when shown")
    print("five correct programs at once than when shown one alone. Bag size and prompt")
    print("wording change together here, so this run cannot say which causes it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "correct_only"], default="full")
    ap.add_argument("--arms", nargs="+", default=["baseline", "rag", "ref", "rag_ref"])
    ap.add_argument("--model", default=os.getenv("MODEL", "qwen3.6-mcminer:latest"))
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.mode == "full":
        summarize_full(args)
    else:
        summarize_correct_only(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
