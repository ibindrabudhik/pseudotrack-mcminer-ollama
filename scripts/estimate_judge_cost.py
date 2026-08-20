#!/usr/bin/env python
"""Preflight + cost estimate for re-judging mined predictions with a hosted model.

Does three things before any money is spent:

  1. Verifies the OpenRouter key works and that the judge model actually exists
     on the account (a typo'd model id otherwise fails once per prediction).
  2. Counts exactly how many judge API calls each arm needs, using the same
     skip rules as compute_eval_metrics_multi.py (correct-only bags and empty
     predictions are scored by rule, with no API call).
  3. Renders the real judge prompts and prices them against OpenRouter's LIVE
     per-token rates, so the estimate reflects current pricing rather than a
     figure baked in when this script was written.

Exit codes: 0 ok, 1 preflight failure.

Usage:
    python scripts/estimate_judge_cost.py --arms baseline rag ref rag_ref \
        --tag-prefix ollama_gpt-oss-mcminer-latest --judge-model openai/gpt-5
"""
import argparse
import importlib.util
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_judge_module():
    """Import compute_eval_metrics_multi so we reuse its exact prompt building."""
    path = os.path.join(ROOT, "src", "compute_eval_metrics_multi.py")
    spec = importlib.util.spec_from_file_location("cem", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch_models(api_key, base_url):
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r).get("data", [])


def openrouter_price(model_id):
    """Look up per-token pricing from OpenRouter's PUBLIC model list.

    Used as a reference rate when judging through OpenAI directly, whose
    /v1/models carries no pricing. OpenRouter lists provider list prices, so
    this is a close proxy rather than a billing guarantee. Returns (in, out) in
    USD per token, or (0, 0) if the model is not listed.
    """
    candidates = [model_id, f"openai/{model_id}"]
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/models")
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r).get("data", [])
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, None
    for c in candidates:
        m = next((x for x in data if x.get("id") == c), None)
        if m:
            p = m.get("pricing", {}) or {}
            return float(p.get("prompt") or 0), float(p.get("completion") or 0), c
    return 0.0, 0.0, None


def approx_tokens(text):
    """~3.7 chars/token for English prose mixed with code. Deliberately rough:
    it is an estimate for a spend decision, not billing."""
    return max(1, int(len(text) / 3.7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tag-prefix", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--provider", default="openrouter", choices=["openrouter", "openai"])
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--max-output-tokens", type=int, default=4000,
                    help="Judge max_tokens; used as the worst-case output estimate")
    ap.add_argument("--misconceptions-file",
                    default="dataset/pseudocode_track/misconceptions_22.json")
    ap.add_argument("--input-dir", default="dataset/pseudocode_track/pseudocode_codes")
    args = ap.parse_args()

    os.chdir(ROOT)
    problems = []

    # ---------------------------------------------------------------- 1. key
    env_var = "OPENAI_API_KEY" if args.provider == "openai" else "OPENROUTER_API_KEY"
    key = os.getenv(env_var, "")
    if not key or key == "ollama":
        print(f"ERROR: {env_var} is unset or still the local dummy value 'ollama'.")
        print(f"       A real key is required to judge with {args.provider}:")
        print(f"         export {env_var}=...")
        return 1
    if args.provider == "openrouter" and not key.startswith("sk-or-"):
        print("WARNING: OpenRouter keys normally start with 'sk-or-'. An OpenAI "
              "'sk-proj-' key will NOT authenticate here.")

    try:
        models = fetch_models(key, args.base_url)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not reach {args.base_url}/models -- {e}")
        print("       If this is a 401, the key is not valid for this provider.")
        return 1
    print(f"OK  key accepted; {len(models)} models visible at {args.base_url}")

    match = next((m for m in models if m.get("id") == args.judge_model), None)
    if match is None:
        near = [m["id"] for m in models if args.judge_model.split("/")[-1][:6] in m.get("id", "")]
        print(f"ERROR: judge model '{args.judge_model}' not found on this account.")
        if near:
            print("       Did you mean one of:")
            for n in near[:10]:
                print(f"         {n}")
        return 1
    ctx = match.get("context_length")
    print(f"OK  model '{args.judge_model}' available"
          + (f"; context={ctx}" if ctx else ""))

    # OpenAI's /v1/models carries no pricing; fall back to OpenRouter's public
    # list price for the same model as a reference rate.
    pricing = match.get("pricing", {}) or {}
    p_in = float(pricing.get("prompt") or 0)
    p_out = float(pricing.get("completion") or 0)
    if not (p_in or p_out):
        p_in, p_out, src = openrouter_price(args.judge_model)
        if p_in or p_out:
            print(f"    pricing unavailable from {args.provider}; using OpenRouter's "
                  f"listed rate for '{src}' as a reference")
        else:
            print("    WARNING: no pricing found for this model. Token counts below "
                  "are exact; the dollar estimate will read $0.00 -- check the "
                  "provider's pricing page yourself before proceeding.")
    print(f"    rates: ${p_in*1e6:.2f}/M input, ${p_out*1e6:.2f}/M output")

    # ------------------------------------------------- 2/3. count and price
    cem = load_judge_module()
    misc_map = cem.load_misc_map(args.misconceptions_file)
    code_index = cem.build_code_index(args.input_dir)
    tpl = cem.load_judge_template()

    grand = {"calls": 0, "in": 0, "out": 0}
    print(f"\n{'arm':10} {'M calls':>8} {'S calls':>8} {'total':>7} {'input tok':>11} {'est USD':>9}")
    print("-" * 60)

    for arm in args.arms:
        tag = f"{args.tag_prefix}_{arm}"
        multi_f = f"results/{tag}/multi/multi_predictions.json"
        single_f = f"results/{tag}/single/predictions.json"
        for f in (multi_f, single_f):
            if not os.path.exists(f):
                problems.append(f"missing predictions for arm '{arm}': {f}")
        if problems:
            continue

        # Build the exact judged set for each side.
        #
        # McMiner-M: every bag in multi_predictions.json, minus correct-only bags
        # and empty predictions (compute_eval_metrics_multi scores those by rule).
        #
        # McMiner-S: NOT every single prediction. evaluate_single_multi_predictions
        # only judges single predictions that landed inside a `misconception` bag
        # (see its misconception_pred_ids loop), and bags cover a subset of the
        # corpus -- so counting predictions.json directly over-estimates badly.
        to_judge = []

        for p in cem.load_json(multi_f):
            gt, is_correct_bag = cem.resolve_gt(p, misc_map)
            items = p.get("predicted_misconceptions") or []
            if is_correct_bag or p.get("no_predicted_misconceptions", False) or not items:
                continue
            to_judge.append(("M", p, gt))
        m_calls = len(to_judge)

        grouped_f = f"results/{tag}/single_multi/grouped_predictions.json"
        if os.path.exists(grouped_f):
            for group in cem.load_json(grouped_f):
                if group.get("group_type") != "misconception":
                    continue
                for sp in group.get("single_predictions", []):
                    gt, is_correct_bag = cem.resolve_gt(sp, misc_map)
                    items = sp.get("predicted_misconceptions") or []
                    if is_correct_bag or sp.get("no_predicted_misconceptions", False) or not items:
                        continue
                    if not gt:
                        continue
                    to_judge.append(("S", sp, gt))
        else:
            problems.append(f"missing aligned predictions for arm '{arm}': {grouped_f}")
            continue

        in_tok = 0
        for _side, p, gt in to_judge:
            prompt = (tpl.replace("{code}", cem.get_code_for_prediction(p, code_index))
                         .replace("{gt_description}", gt)
                         .replace("{predicted}", cem.format_predicted(p)))
            in_tok += approx_tokens(prompt)

        calls = len(to_judge)
        out_tok = calls * args.max_output_tokens
        cost = in_tok * p_in + out_tok * p_out
        grand["calls"] += calls; grand["in"] += in_tok; grand["out"] += out_tok
        print(f"{arm:10} {m_calls:8} {calls-m_calls:8} {calls:7} {in_tok:11,} ${cost:8.2f}")

    if problems:
        print("\nERROR: preflight failed")
        for p in problems:
            print(f"  - {p}")
        print("\nMine first (RUN_EVAL=0 bash run_all.sh) or fix --tag-prefix.")
        return 1

    lo = grand["in"] * p_in + grand["calls"] * 300 * p_out          # if judge is terse
    hi = grand["in"] * p_in + grand["out"] * p_out                  # if it maxes out
    print("-" * 60)
    print(f"{'TOTAL':10} {'':8} {'':8} {grand['calls']:7} {grand['in']:11,}")
    print(f"\nEstimated cost: ${lo:.2f} (terse ~300 output tok/call) "
          f"to ${hi:.2f} (worst case, all {args.max_output_tokens} output tok used)")
    print("Output tokens dominate for a reasoning judge and are the least predictable"
          "\npart of this estimate -- treat the upper bound as the number that matters.")
    json.dump({"calls": grand["calls"], "input_tokens": grand["in"],
               "est_usd_low": round(lo, 2), "est_usd_high": round(hi, 2),
               "price_in_per_mtok": p_in * 1e6, "price_out_per_mtok": p_out * 1e6},
              open("results/.judge_cost_estimate.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
