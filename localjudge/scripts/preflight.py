#!/usr/bin/env python
"""Preflight for the local dual-judge run.

Checks, in order of how expensive the mistake would be:

  1. Ollama reachable, both judge models present.
  2. Judge inputs present for every arm.
  3. Exact judge-call counts (same skip rules as the judge itself).
  4. LIVE PROBE: does each judge's thinking switch actually take effect, and
     does it emit a parseable <evaluation> block within the token budget?

Check 4 is the one worth waiting for. Both models here reason before answering,
those tokens come out of the same budget as the answer, and when the budget runs
out the reply is truncated -- which the parser turns into a SCORE, not an error.
Measured previously: gpt-oss at default effort burned 4000/4000 tokens and
returned zero content; qwen3.6 with thinking on took >10 min for one call.

Usage:
    python scripts/preflight.py --arms baseline rag ref rag_ref
    python scripts/preflight.py --arms baseline --no-probe     # skip step 4
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Per-judge "stop thinking" switches. Kept in sync with scripts/_common.sh --
# see the long comment there for why each family needs a different one.
QWEN_NO_THINK = json.dumps({"chat_template_kwargs": {"enable_thinking": False}})
JUDGES = {
    "gpt-oss-judge": {"LLM_REASONING_EFFORT": "low"},
    "qwen36-judge": {"LLM_EXTRA_BODY": QWEN_NO_THINK},
}


def judge_profile(name):
    """Look up the thinking-control env for a judge, ignoring any :tag suffix.

    Ollama reports models as 'qwen36-judge:latest', so an exact dict lookup on
    the bare name silently returns {} -- leaving the model at its DEFAULT
    thinking setting. That failure is invisible: the run still works, just
    10x slower or with empty replies.
    """
    base = name.split(":", 1)[0]
    return dict(JUDGES.get(name) or JUDGES.get(base) or {})


def load_cem():
    spec = importlib.util.spec_from_file_location(
        "cem", os.path.join(ROOT, "src", "compute_eval_metrics_multi.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def http_json(url, timeout=15):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
        return json.load(r)


def stop_model(name):
    try:
        subprocess.run(["ollama", "stop", name], capture_output=True, timeout=60)
    except Exception:  # noqa: BLE001
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["baseline", "rag", "ref", "rag_ref"])
    ap.add_argument("--judges", nargs="+", default=list(JUDGES))
    ap.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ap.add_argument("--pred-root", default="predictions")
    ap.add_argument("--max-tokens", type=int, default=int(os.getenv("JUDGE_MAX_TOKENS", "3000")))
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    fail = []

    # ---------------------------------------------------------------- 1. server
    try:
        tags = http_json(args.base_url + "/api/tags")
    except Exception as e:  # noqa: BLE001
        print("FAIL  Ollama not reachable at %s -- %s" % (args.base_url, e))
        print("      Start it with:  ollama serve")
        return 1
    names = {m["name"] for m in tags.get("models", [])}
    print("OK    Ollama up at %s; %d models" % (args.base_url, len(names)))

    for j in args.judges:
        if not any(n == j or n.startswith(j + ":") for n in names):
            fail.append("judge model '%s' not built. Run: bash scripts/build_models.sh" % j)
        else:
            print("OK    judge model '%s' present" % j)

    # ------------------------------------------------------------- 2/3. inputs
    cem = load_cem()
    misc = cem.load_misc_map("dataset/pseudocode_track/misconceptions_22.json")
    idx = cem.build_code_index("dataset/pseudocode_track/pseudocode_codes")
    tpl = cem.load_judge_template()
    print("OK    judge template loaded (%d chars)" % len(tpl))

    print("\n%-10s %8s %8s %7s %15s" % ("arm", "M calls", "S calls", "total", "max prompt tok"))
    print("-" * 54)
    grand = 0
    for arm in args.arms:
        multi_f = "%s/%s/multi/multi_predictions.json" % (args.pred_root, arm)
        grouped_f = "%s/%s/single_multi/grouped_predictions.json" % (args.pred_root, arm)
        missing = [f for f in (multi_f, grouped_f) if not os.path.exists(f)]
        if missing:
            for f in missing:
                fail.append("missing judge input for arm '%s': %s" % (arm, f))
            continue

        biggest = calls_m = calls_s = 0
        for p in cem.load_json(multi_f):
            gt, correct_bag = cem.resolve_gt(p, misc)
            items = p.get("predicted_misconceptions") or []
            if correct_bag or p.get("no_predicted_misconceptions") or not items:
                continue
            calls_m += 1
            biggest = max(biggest, len(tpl) + len(cem.get_code_for_prediction(p, idx)))
        for g in cem.load_json(grouped_f):
            if g.get("group_type") != "misconception":
                continue
            for sp in g.get("single_predictions", []):
                gt, correct_bag = cem.resolve_gt(sp, misc)
                items = sp.get("predicted_misconceptions") or []
                if correct_bag or sp.get("no_predicted_misconceptions") or not items or not gt:
                    continue
                calls_s += 1
        tok = int(biggest / 3.7)
        grand += calls_m + calls_s
        print("%-10s %8d %8d %7d %15s" % (arm, calls_m, calls_s, calls_m + calls_s, "{:,}".format(tok)))
        if tok > 14000:
            fail.append("arm '%s' has a ~%d token prompt; num_ctx 16384 leaves little room" % (arm, tok))
    print("-" * 54)
    print("%-10s %8s %8s %7d   x%d judges = %d calls"
          % ("TOTAL", "", "", grand, len(args.judges), grand * len(args.judges)))

    # ------------------------------------------------------------- 4. live probe
    if not args.no_probe and not fail:
        print("\n=== live probe: one real judge call per model ===")
        preds = cem.load_json("%s/%s/single/predictions.json" % (args.pred_root, args.arms[0]))
        p = next(x for x in preds
                 if (x.get("predicted_misconceptions") or [])
                 and not x.get("no_predicted_misconceptions")
                 and (x.get("ground_truth_misconception") or {}).get("description"))
        gt, _ = cem.resolve_gt(p, misc)
        prompt = (tpl.replace("{code}", cem.get_code_for_prediction(p, idx))
                     .replace("{gt_description}", gt)
                     .replace("{predicted}", cem.format_predicted(p)))
        os.environ["OPENROUTER_BASE_URL"] = args.base_url + "/v1"
        os.environ.setdefault("OPENROUTER_API_KEY", "ollama")
        os.environ["JUDGE_MAX_TOKENS"] = str(args.max_tokens)

        for j in args.judges:
            for k in ("LLM_REASONING_EFFORT", "LLM_EXTRA_BODY"):
                os.environ.pop(k, None)
            prof = judge_profile(j)
            if not prof:
                print("  %-15s WARNING: no thinking-control profile; using model "
                      "defaults (likely slow / may return empty content)" % j)
            os.environ.update(prof)
            client = cem.create_judge_client("openrouter", j)
            t0 = time.time()
            try:
                raw = cem.call_judge(client, "openrouter", j, prompt)
            except Exception as e:  # noqa: BLE001
                print("  %-15s FAILED after %.0fs: %s: %s" % (j, time.time() - t0, type(e).__name__, e))
                fail.append("judge '%s' could not complete a call" % j)
                continue
            dt = time.time() - t0
            pr = cem.parse_judge_response(raw)
            print("  %-15s %s %6.1fs  %5d chars  parse_ok=%s  match=%s"
                  % (j, "OK " if pr["parse_ok"] else "BAD", dt, len(raw), pr["parse_ok"], pr["match"]))
            if not pr["parse_ok"]:
                fail.append("judge '%s' returned no parseable <evaluation> block -- raise "
                            "JUDGE_MAX_TOKENS or check its thinking switch" % j)
                print("      raw head: %r" % raw[:200])
            elif dt > 90:
                print("      WARNING: %.0fs/call x %d calls = ~%.1fh for this judge alone. "
                      "Thinking may still be on." % (dt, grand, dt * grand / 3600.0))
            stop_model(j)

    if fail:
        print("\nPREFLIGHT FAILED")
        for f in fail:
            print("  - %s" % f)
        return 1
    print("\nPreflight OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
