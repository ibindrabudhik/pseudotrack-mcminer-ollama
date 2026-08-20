#!/usr/bin/env python
"""Compare two (or more) judges over the same predictions.

Reports per-judge accuracy, then the part that actually matters: PER-CASE
agreement and Cohen's kappa.

Aggregate rates can match exactly while the underlying verdicts differ. That is
not hypothetical -- in an earlier run one arm's two judges produced identical
McMiner-S totals to the decimal while disagreeing on 14 individual predictions
(9 one way, 5 the other). Reading the totals alone would have concluded "the
judges agree", which was false.

Kappa corrects for agreement expected by chance:
    <0.40 poor   0.40-0.60 moderate   0.60-0.80 substantial   >0.80 strong

Usage:
    python scripts/compare_judges.py --judges gpt-oss-judge qwen36-judge \
        --arms baseline rag ref rag_ref
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def kappa(a, b):
    """Cohen's kappa for two boolean sequences. Returns (agreement, kappa)."""
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / float(n)
    pa, pb = sum(a) / float(n), sum(b) / float(n)
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, ((po - pe) / (1 - pe) if pe < 1 else float("nan"))


def judged(details):
    """Only cases an LLM actually judged; rule-scored ones carry no signal."""
    return {d["prediction_id"]: d for d in details if d.get("method") == "llm_judge"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out-root", default="results")
    args = ap.parse_args()
    os.chdir(ROOT)

    # ------------------------------------------------------------- accuracy
    print("\n=== accuracy by judge ===")
    print("%-16s %-9s %8s %9s %8s %9s %7s"
          % ("judge", "arm", "S std", "S novel", "M std", "M novel", "fails"))
    print("-" * 74)
    have = {}
    for j in args.judges:
        for a in args.arms:
            s = load("%s/%s/%s/single_multi/evaluation_metrics.json" % (args.out_root, j, a))
            m = load("%s/%s/%s/multi/claude_evaluation_results.json" % (args.out_root, j, a))
            if not s or not m:
                print("%-16s %-9s %s" % (j, a, "(missing)"))
                continue
            ss = s["standard_metrics"]["overall_metrics"]["overall_accuracy"]
            sn = s["with_novel_metrics"]["overall_metrics"]["overall_accuracy"]
            ms = m["summary"]["match_rate"]
            mn = m["summary"]["match_with_novel_rate"]
            fails = m["summary"].get("judge_parse_failures", 0)
            sj = load("%s/%s/%s/single_multi/judge_details_single.json" % (args.out_root, j, a))
            if sj:
                fails += sj.get("summary", {}).get("judge_parse_failures", 0)
            have[(j, a)] = True
            flag = "  <-- CHECK" if fails else ""
            print("%-16s %-9s %7.2f%% %8.2f%% %7.2f%% %8.2f%% %7d%s"
                  % (j, a, ss * 100, sn * 100, ms * 100, mn * 100, fails, flag))

    if len(args.judges) < 2:
        print("\n(only one judge -- no agreement analysis)")
        return 0

    # ------------------------------------------------------------- agreement
    ja, jb = args.judges[0], args.judges[1]
    print("\n=== per-case agreement: %s vs %s ===" % (ja, jb))
    print("%-9s %-10s %5s %8s %7s %12s %12s"
          % ("arm", "miner", "n", "agree", "kappa", ja[:11] + "-only", jb[:11] + "-only"))
    print("-" * 74)

    for a in args.arms:
        if (ja, a) not in have or (jb, a) not in have:
            continue
        # McMiner-M: aligned by list order (correct-only bags share a prediction_id,
        # so an id-keyed join would silently collapse them).
        A = load("%s/%s/%s/multi/claude_evaluation_results.json" % (args.out_root, ja, a))["evaluation_details"]
        B = load("%s/%s/%s/multi/claude_evaluation_results.json" % (args.out_root, jb, a))["evaluation_details"]
        if len(A) == len(B):
            am = [bool(x["match"]) for x in A]
            bm = [bool(x["match"]) for x in B]
            po, k = kappa(am, bm)
            print("%-9s %-10s %5d %7.1f%% %7.2f %12d %12d"
                  % (a, "McMiner-M", len(am), po * 100, k,
                     sum(1 for x, y in zip(am, bm) if x and not y),
                     sum(1 for x, y in zip(am, bm) if y and not x)))

        # McMiner-S: aligned by prediction_id (unique for single predictions).
        pa = load("%s/%s/%s/single_multi/judge_details_single.json" % (args.out_root, ja, a))
        pb = load("%s/%s/%s/single_multi/judge_details_single.json" % (args.out_root, jb, a))
        if pa and pb:
            da, db = judged(pa["evaluation_details"]), judged(pb["evaluation_details"])
            ids = sorted(set(da) & set(db))
            if ids:
                am = [bool(da[i]["match"]) for i in ids]
                bm = [bool(db[i]["match"]) for i in ids]
                po, k = kappa(am, bm)
                print("%-9s %-10s %5d %7.1f%% %7.2f %12d %12d"
                      % (a, "McMiner-S", len(ids), po * 100, k,
                         sum(1 for x, y in zip(am, bm) if x and not y),
                         sum(1 for x, y in zip(am, bm) if y and not x)))

    print("\nkappa: <0.40 poor | 0.40-0.60 moderate | 0.60-0.80 substantial | >0.80 strong")
    print("Equal accuracy totals do NOT imply equal verdicts -- read the kappa column.")
    print("A judge whose '-only' count is consistently higher is the more lenient one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
