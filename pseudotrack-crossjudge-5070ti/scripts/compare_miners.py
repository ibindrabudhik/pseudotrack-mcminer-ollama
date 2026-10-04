#!/usr/bin/env python3
"""Compare the two miners head to head on the correct-bags-only run.

    python scripts/compare_miners.py
    python scripts/compare_miners.py --arms baseline rag

Direction A mines with gpt-oss:20b, direction B with qwen3.6:27b. In a
correct-only run the judge is never called (ground truth is NONE, so both
scorers decide by rule), so this comparison is completely free of judge choice
and judge bias -- the only thing that differs between the two columns is the
model that wrote the predictions.

Three views, in increasing order of how much they can be trusted:

  bag level     -- the headline metric, but n is tiny (4 bags at 1 pass)
  code level    -- all 96 mined correct codes; finer grained, but the 96 rows
                   are only 19 distinct programs, weighted by an artefact of
                   dataset construction
  program level -- majority vote per distinct program; removes that weighting
                   and also exposes self-consistency (same program, same prompt,
                   different answer)

A bag fails if ANY of its codes draws a spurious misconception, so the bag rate
is always the harsher of the two.
"""
import argparse
import json
import os
import re
from collections import defaultdict

DIRECTIONS = {
    "A": ("gpt-oss-mcminer:latest", "gpt-oss:20b  mines -> qwen3.6 judges"),
    "B": ("qwen3.6-mcminer:latest", "qwen3.6:27b  mines -> gpt-oss judges"),
}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tag_for(model, arm, direction):
    """Must match MODEL_TAG in scripts/_common.sh."""
    slug = re.sub(r"[:/]", "-", model)
    return "dir%s_%s_%s_correctbags" % (direction, slug, arm)


def pct(x):
    return "   n/a" if x is None else "%5.1f%%" % (x * 100)


def correct_singles(tag):
    """Every mined correct code for a tag, as (problem_id, abstained)."""
    out = []
    for p in load("results/%s/single/predictions.json" % tag) or []:
        if (p.get("ground_truth_misconception") or {}).get("id") != "NONE":
            continue
        abstained = bool(p.get("no_predicted_misconceptions")
                         or not (p.get("predicted_misconceptions") or []))
        out.append((p.get("problem_id"), abstained))
    return out


def stats(model, arm, direction):
    tag = tag_for(model, arm, direction)
    multi = load("results/evaluations/%s/multi/evaluation_metrics.json" % tag)
    rows = correct_singles(tag)

    bag_rate = bag_n = None
    if multi:
        o = multi["standard_metrics"]["overall_metrics"]
        bag_rate, bag_n = o.get("correct_only_accuracy"), o.get("correct_only_count")

    code_rate = (sum(1 for _, a in rows if a) / len(rows)) if rows else None

    by_prog = defaultdict(list)
    for pid, a in rows:
        by_prog[pid].append(a)
    maj = sum(1 for v in by_prog.values() if sum(v) * 2 > len(v)) if by_prog else None
    incons = sum(1 for v in by_prog.values() if len(set(v)) > 1) if by_prog else None
    prog_rate = (maj / len(by_prog)) if by_prog else None

    return {"tag": tag, "found": multi is not None or bool(rows),
            "bag_rate": bag_rate, "bag_n": bag_n,
            "code_rate": code_rate, "code_n": len(rows),
            "prog_rate": prog_rate, "prog_n": len(by_prog),
            "maj": maj, "incons": incons}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["baseline", "rag", "ref", "rag_ref"])
    ap.add_argument("--model-a", default=os.getenv("MODEL_A", DIRECTIONS["A"][0]))
    ap.add_argument("--model-b", default=os.getenv("MODEL_B", DIRECTIONS["B"][0]))
    args = ap.parse_args()

    models = {"A": args.model_a, "B": args.model_b}

    print("\n" + "=" * 78)
    print("  MINER vs MINER  --  correct bags only (abstention = getting it right)")
    print("=" * 78)
    print("  A: %s   [%s]" % (args.model_a, DIRECTIONS["A"][1]))
    print("  B: %s   [%s]" % (args.model_b, DIRECTIONS["B"][1]))
    print("\n  Higher is better everywhere: these are rates of correctly saying")
    print("  'no misconception' about code that has none. ZERO judge calls were")
    print("  made, so nothing here depends on the judge model.\n")

    data = {d: {arm: stats(models[d], arm, d) for arm in args.arms} for d in ("A", "B")}

    any_found = any(s["found"] for d in data for s in data[d].values())
    if not any_found:
        print("  No results found. Has the run finished? Expected directories like:")
        print("    results/%s" % tag_for(args.model_a, args.arms[0], "A"))
        print("    results/%s" % tag_for(args.model_b, args.arms[0], "B"))
        return

    # ---- bag level ----------------------------------------------------------
    print("--- bag level (a bag fails if ANY of its codes is a false positive) ---")
    print("%-9s %18s %18s %12s" % ("arm", "A gpt-oss", "B qwen3.6", "diff (A-B)"))
    print("-" * 62)
    for arm in args.arms:
        a, b = data["A"][arm], data["B"][arm]
        diff = (a["bag_rate"] - b["bag_rate"]) * 100 \
            if a["bag_rate"] is not None and b["bag_rate"] is not None else None
        print("%-9s %10s (n=%-3s) %10s (n=%-3s) %11s"
              % (arm, pct(a["bag_rate"]), a["bag_n"] if a["bag_n"] is not None else "?",
                 pct(b["bag_rate"]), b["bag_n"] if b["bag_n"] is not None else "?",
                 "n/a" if diff is None else "%+6.1f pp" % diff))

    # ---- code level ---------------------------------------------------------
    print("\n--- code level (every mined correct code) ---")
    print("%-9s %18s %18s %12s" % ("arm", "A gpt-oss", "B qwen3.6", "diff (A-B)"))
    print("-" * 62)
    for arm in args.arms:
        a, b = data["A"][arm], data["B"][arm]
        diff = (a["code_rate"] - b["code_rate"]) * 100 \
            if a["code_rate"] is not None and b["code_rate"] is not None else None
        print("%-9s %10s (n=%-3d) %10s (n=%-3d) %11s"
              % (arm, pct(a["code_rate"]), a["code_n"],
                 pct(b["code_rate"]), b["code_n"],
                 "n/a" if diff is None else "%+6.1f pp" % diff))

    # ---- program level ------------------------------------------------------
    print("\n--- program level (majority vote per distinct program) ---")
    print("%-9s %18s %18s %22s" % ("arm", "A gpt-oss", "B qwen3.6", "inconsistent A / B"))
    print("-" * 72)
    for arm in args.arms:
        a, b = data["A"][arm], data["B"][arm]
        print("%-9s %10s (%2s/%-2s) %10s (%2s/%-2s) %18s"
              % (arm,
                 pct(a["prog_rate"]), a["maj"] if a["maj"] is not None else "?", a["prog_n"],
                 pct(b["prog_rate"]), b["maj"] if b["maj"] is not None else "?", b["prog_n"],
                 "%s / %s" % (a["incons"], b["incons"])))

    print("\n'inconsistent' counts programs that got BOTH answers across their repeated")
    print("rows -- same program, same prompt, same temperature. That is pure model")
    print("noise, and it bounds how much of any A-vs-B difference above is real.")
    print("If inconsistent is a large share of programs, treat small gaps as noise.")

    # ---- the honest caveat --------------------------------------------------
    worst_n = min((s["bag_n"] or 0) for d in data for s in data[d].values())
    print("\n" + "-" * 78)
    if worst_n and worst_n <= 8:
        print("CAUTION: as few as %d bags per cell. One bag is %.1f pp, so the bag-level"
              % (worst_n, 100.0 / worst_n))
        print("table cannot separate these miners. Raise CORRECT_PASSES (5 passes gives")
        print("~20 bags) and prefer the code- and program-level rows for any claim.")
    print("Reminder: the dataset contains only 19 distinct correct programs, so the")
    print("effective sample is 19 no matter how many rows or passes you run.")
    print("-" * 78)


if __name__ == "__main__":
    main()
