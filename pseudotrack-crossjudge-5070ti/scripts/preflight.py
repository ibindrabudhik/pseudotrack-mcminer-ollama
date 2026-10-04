#!/usr/bin/env python
"""Preflight for the qwen3.6-mines / gpt-oss-judges bundle.

Checks, in order of how expensive the mistake would be:

  1. Ollama reachable; BOTH models built; their num_ctx is big enough.
  2. Dataset + every prompt template the requested arms will ask for.
  3. Exact call counts -- bags, single codes, and how many of those reach the
     judge -- by running the real bag former, not by estimating.
  4. LIVE PROBE: one real mining call and one real judge call, verifying each
     model's "stop thinking" switch takes effect and that the reply parses.

Check 4 is the one worth waiting for. Both models here reason before answering,
those tokens come out of the same budget as the answer, and when the budget runs
out the reply is truncated -- which the parsers turn into a SCORE or a "no
misconception found", not an error. Measured on this dataset: gpt-oss at default
effort burned 4000/4000 tokens and returned zero content; qwen3.6 with thinking
on took >10 min for one call.

Usage:
    python scripts/preflight.py --mode full
    python scripts/preflight.py --mode correct_only --arms baseline
    python scripts/preflight.py --mode full --no-probe        # skip step 4
"""
import argparse
import contextlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

ollama_client = None  # bound in main() once sys.path includes the bundle root

# Reasoning control now lives in ONE place -- THINK_BY_MODEL in
# utils/ollama_client.py, applied by model name. Nothing to mirror here, and
# nothing that can leak from one pipeline step into the next.

# arm -> (single template, multi template, needs rag, needs ref)
ARM_SPEC = {
    "baseline": ("zeroshot", "zeroshot-no-reasoning-multi", False, False),
    "rag": ("zeroshot-rag", "zeroshot-no-reasoning-multi-rag", True, False),
    "ref": ("zeroshot-ref", "zeroshot-no-reasoning-multi-ref", False, True),
    "rag_ref": ("zeroshot-rag-ref", "zeroshot-no-reasoning-multi-rag-ref", True, True),
}

DATASET = "dataset/pseudocode_track"
IN = DATASET + "/pseudocode_codes"
NONE_IN = DATASET + "/pseudocode_codes_none"
MISC = DATASET + "/misconceptions_22.json"
PROBLEMS = DATASET + "/problems_pseudocode.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def stop_model(name):
    try:
        subprocess.run(["ollama", "stop", name], capture_output=True, timeout=60)
    except Exception:  # noqa: BLE001
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "correct_only"], default="full")
    ap.add_argument("--arms", nargs="+", default=list(ARM_SPEC))
    ap.add_argument("--miner", default=os.getenv("MODEL", "qwen3.6-mcminer:latest"))
    ap.add_argument("--judge", default=os.getenv("JUDGE_MODEL", "gpt-oss-judge:latest"))
    ap.add_argument("--host", default=os.getenv("OLLAMA_HOST_URL", "http://localhost:11434"))
    ap.add_argument("--correct-passes", type=int, default=1)
    ap.add_argument("--bag-size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, SRC)
    sys.path.insert(0, ROOT)
    # Imported here rather than at module scope: sys.path is only correct now.
    global ollama_client
    from utils import ollama_client  # noqa: PLC0415
    os.environ.setdefault("OLLAMA_HOST_URL", args.host)
    fail, warn = [], []

    print("=== 1. Ollama + models ===")
    try:
        names = set(ollama_client.list_models(args.host))
    except Exception as e:  # noqa: BLE001
        print("FAIL  Ollama not reachable at %s -- %s" % (args.host, e))
        print("      Start it with:  ollama serve")
        return 1
    print("OK    Ollama up at %s; %d models installed" % (args.host, len(names)))

    # num_ctx is checked against the MEASURED prompt sizes in step 3, not against
    # a rule of thumb -- the parent repo's 32K default came from an estimate that
    # turned out to be 2.5x too high. Here we only record what each model was
    # built with; step 3 decides whether it is enough.
    ctx_of = {}
    for role, model in (("miner", args.miner), ("judge", args.judge)):
        if not any(n == model or n.startswith(model.split(":")[0] + ":") for n in names):
            fail.append("%s model '%s' not built. Run: bash scripts/build_models.sh" % (role, model))
            continue
        ctx = ollama_client.num_ctx_of(model, args.host)
        ctx_of[role] = ctx
        print("OK    %-5s '%s' present, num_ctx=%s"
              % (role, model, ctx if ctx else "unknown"))

    if args.miner.split(":")[0] == args.judge.split(":")[0]:
        warn.append("miner and judge are the same model -- that is self-evaluation, "
                    "which this bundle exists to avoid")

    # ------------------------------------------------------------ 2. inputs
    print("\n=== 2. dataset + templates ===")
    for p in (MISC, PROBLEMS):
        print(("OK    " if os.path.exists(p) else "FAIL  ") + p)
        if not os.path.exists(p):
            fail.append("missing dataset file: " + p)
    for d, n in ((IN, 209), (NONE_IN, 96)):
        if not os.path.isdir(d):
            fail.append("missing dataset dir: " + d)
            print("FAIL  " + d)
            continue
        got = len([f for f in os.listdir(d) if f.endswith(".json")])
        print("OK    %-46s %d json files%s" % (d, got, "" if got == n else " (expected %d)" % n))

    tpl_dir = os.path.join(SRC, "prompt_templates", "mining-pseudocode")
    for arm in args.arms:
        if arm not in ARM_SPEC:
            fail.append("unknown arm '%s'" % arm)
            continue
        single, multi, needs_rag, needs_ref = ARM_SPEC[arm]
        # The correct-only run never mines the corrupted codes, so it never
        # loads the single-code template for anything but the NONE pass -- which
        # uses the same file. Check both either way; they are cheap.
        for t in (single, multi):
            p = os.path.join(tpl_dir, t + ".md")
            if not os.path.exists(p):
                fail.append("missing template for arm '%s': %s" % (arm, p))
        if needs_rag:
            for p in ("dataset/retrival_openai_embedding_large.csv",
                      "dataset/retrival_correct_codes.csv"):
                if not os.path.exists(p):
                    fail.append("arm '%s' needs %s" % (arm, p))
        if needs_ref and not os.path.exists("dataset/Submission_Code_with_reference_from_APR.csv"):
            fail.append("arm '%s' needs dataset/Submission_Code_with_reference_from_APR.csv" % arm)
    judge_tpl = os.path.join(SRC, "prompt_templates", "evaluation-pseudocode",
                             "judge_prediction_match.md")
    if not os.path.exists(judge_tpl):
        fail.append("missing judge template: " + judge_tpl)
    print("OK    templates for arms: %s" % " ".join(args.arms))

    # -------------------------------------------------------- 3. call counts
    # Counted by running the REAL bag former, not by estimating. The correct-bag
    # count in particular is not obvious: cover-all mode partitions the *unique
    # problems* that have a correct solution, so 96 files collapse to far fewer
    # bags than a naive 96/5 would suggest.
    print("\n=== 3. work this run will do ===")
    rim = load_module("run_infer_misc_multi", os.path.join(SRC, "run_infer_misc_multi.py"))
    random.seed(args.seed)
    problems = rim.load_json_data(PROBLEMS)
    corrupted = rim.load_corrupted_codes(IN)
    correct_sols = rim.get_correct_solutions(problems)
    correct_bags = rim.create_correct_only_bags(
        problems, 0, "fixed", None, None, args.bag_size,
        cover_all=True, passes=args.correct_passes)
    n_correct_codes = len([f for f in os.listdir(NONE_IN) if f.endswith(".json")])

    print("  correct programs available : %d unique (from %d NONE files)"
          % (len(correct_sols), n_correct_codes))
    print("  correct-only bags          : %d  (bag size %d, %d pass(es), cover-all)"
          % (len(correct_bags), args.bag_size, args.correct_passes))

    groups = {}
    if args.mode == "correct_only":
        per_arm_mining = len(correct_bags) + n_correct_codes
        print("  McMiner-S calls per arm    : %d (correct codes only; the 209 corrupted "
              "codes are skipped)" % n_correct_codes)
        print("  mining calls per arm       : %d" % per_arm_mining)
        print("  judge calls per arm        : 0")
        print()
        print("  Correct-only bags are scored BY RULE, not by a model: ground truth is")
        print("  NONE, so a bag matches iff the miner predicted nothing. gpt-oss is")
        print("  checked below but will not be asked anything during the run.")
    else:
        groups = rim.group_codes_by_misconception(corrupted, "fixed", None, None, args.bag_size, None)
        groups = {k: v for k, v in groups.items() if len(v) >= 2}
        misc_bags = 0
        for codes in groups.values():
            uniq = len({c.get("problem_id") for c in codes})
            # floor, not ceil: the bag former breaks out when fewer than bag_size
            # unique problems remain, so the short tail bag is DROPPED.
            misc_bags += uniq // args.bag_size
        per_arm_mining = misc_bags + len(correct_bags) + len(corrupted) + n_correct_codes
        print("  misconception bags         : ~%d (from %d groups of >=2 codes)"
              % (misc_bags, len(groups)))
        print("  McMiner-S calls per arm    : %d corrupted + %d correct = %d"
              % (len(corrupted), n_correct_codes, len(corrupted) + n_correct_codes))
        print("  mining calls per arm       : ~%d" % per_arm_mining)
        print("  judge calls per arm        : <= %d bags + <= %d single predictions"
              % (misc_bags, len(corrupted)))
        print("    (only non-empty misconception predictions reach the judge; empty ones")
        print("     are auto-scored as non-match and correct bags are rule-scored)")
    print("  ARMS                       : %d  ->  ~%d mining calls total"
          % (len(args.arms), per_arm_mining * len(args.arms)))

    # -- does the miner's context window actually fit the biggest prompt? -----
    # Built from the real bags with the real retrieval indexes, because a prompt
    # longer than num_ctx is truncated by the server with NO error: the model
    # then answers confidently about a program it only half saw.
    print("\n  worst-case mining prompt per arm (measured, not estimated):")
    print("  %-10s %12s %10s %14s" % ("arm", "max chars", "~tokens", "+4000 resp"))
    worst = 0
    for arm in args.arms:
        if arm not in ARM_SPEC:
            continue
        _, multi_t, needs_rag, needs_ref = ARM_SPEC[arm]
        rag_idx = ref_idx = None
        with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
            if needs_rag:
                import rag_retrieval
                rag_idx = rag_retrieval.load_index("dataset/retrival_openai_embedding_large.csv",
                                                   "dataset/retrival_correct_codes.csv", 3)
            if needs_ref:
                import ref_retrieval
                ref_idx = ref_retrieval.load_reference_index(
                    "dataset/Submission_Code_with_reference_from_APR.csv", "Reference_Code")
        random.seed(args.seed)
        tpl = rim.load_prompt_template(multi_t, tpl_dir)
        # The bag former narrates every step; here we only want the sizes.
        with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
            batches = rim.generate_multi_mining_batches(
                {} if args.mode == "correct_only" else groups, problems, tpl, 0.15, "fixed",
                None, None, args.bag_size, None,
                correct_bags_only=(args.mode == "correct_only"),
                correct_bags_cover_all=True, correct_bags_passes=args.correct_passes,
                rag_index=rag_idx, rag_top_k=3, ref_index=ref_idx)
        mx = max((len(m[0]["content"]) for _, m in batches), default=0)
        tok = int(mx / 3.7)
        worst = max(worst, tok + 4000)
        print("  %-10s %12d %10d %14d" % (arm, mx, tok, tok + 4000))

    ctx = ctx_of.get("miner")
    if ctx and worst:
        if ctx < worst:
            fail.append("miner num_ctx=%d is smaller than the worst prompt+response (%d tokens). "
                        "Prompts WILL be silently truncated. Rebuild with a larger num_ctx."
                        % (ctx, worst))
            print("  FAIL  miner num_ctx=%d < %d needed" % (ctx, worst))
        else:
            print("  OK    miner num_ctx=%d covers the worst case (%d tokens), %.1fx headroom"
                  % (ctx, worst, ctx / float(worst)))
            # Only worth flagging above the bundle's own 16K build: below that
            # the headroom is cheap, and the ratio looks inflated whenever you
            # run a subset of arms (rag_ref is the one with the long prompts).
            if ctx > 16384 and ctx > worst * 3:
                warn.append("miner num_ctx=%d is %.1fx the measured worst case (%d) for the arms "
                            "checked here. Every extra token costs KV cache, which is what decides "
                            "whether the model stays resident. Rebuild at 16384: "
                            "bash scripts/build_models.sh"
                            % (ctx, ctx / float(worst), worst))

    # ------------------------------------------------------------ 4. probe
    if not args.no_probe and not fail:
        print("\n=== 4. live probe: one real call per model ===")
        cem = load_module("cem", os.path.join(SRC, "compute_eval_metrics_multi.py"))
        os.environ["OLLAMA_HOST_URL"] = args.host

        # -- miner: a real bag prompt, through the real parser ---------------
        tpl = rim.load_prompt_template("zeroshot-no-reasoning-multi", tpl_dir)
        bag = correct_bags[0] if correct_bags else None
        if bag is None:
            warn.append("no correct bags to probe the miner with")
        else:
            prompt = rim.create_multi_mining_prompt(tpl, problems, bag)
            client = ollama_client.OllamaClient(model=args.miner, host=args.host)
            t0 = time.time()
            try:
                raw = client.create_message(
                    [{"role": "user", "content": prompt}],
                    kwargs={"model": args.miner, "max_tokens": 4000, "temperature": 0.1})
                dt = time.time() - t0
                parsed = rim.parse_multi_mining_response(raw)
                ok = parsed.get("parse_success") or parsed.get("no_predicted_misconceptions")
                print("  %-24s %s %6.1fs  think=%-7r %5d chars  parse_ok=%s"
                      % (args.miner, "OK " if ok else "BAD", dt,
                         ollama_client.think_for(args.miner), len(raw or ""), bool(ok)))
                if not ok:
                    fail.append("miner '%s' returned nothing parseable" % args.miner)
                    print("      raw head: %r" % (raw or "")[:200])
                elif dt > 120:
                    print("      WARNING: %.0fs for ONE call. x%d calls x%d arms = ~%.1fh. "
                          "Thinking may still be on, or the model is spilling to CPU."
                          % (dt, per_arm_mining, len(args.arms),
                             dt * per_arm_mining * len(args.arms) / 3600.0))
            except Exception as e:  # noqa: BLE001
                print("  %-24s FAILED after %.0fs: %s" % (args.miner, time.time() - t0, e))
                fail.append("miner '%s' could not complete a call" % args.miner)
        ollama_client.unload(args.miner, args.host)
        stop_model(args.miner)

        # -- judge: a real judge prompt, through the real parser -------------
        # Probed even in correct-only mode. It makes no calls during that run,
        # but a broken judge would only surface on the next full run, hours
        # later, and this costs one request.
        misc_map = cem.load_misc_map(MISC)
        code_index = cem.build_code_index(IN)
        sample = code_index[sorted(code_index)[0]]
        gt = misc_map.get(sample.get("misconception_id"), "an off-by-one loop bound")
        code = ""
        for sol in sample.get("solutions", []) or []:
            if sol.get("generated_code") and sol["generated_code"] != "NONE":
                code = sol["generated_code"]
                break
        prompt = (cem.load_judge_template()
                  .replace("{code}", code or "READ n / FOR i = 1 TO n / PRINT i")
                  .replace("{gt_description}", gt if isinstance(gt, str) else str(gt))
                  .replace("{predicted}", "1. Loop bound is off by one, so the last "
                                          "element is never processed."))
        client = cem.create_judge_client("ollama", args.judge)
        t0 = time.time()
        try:
            raw = cem.call_judge(client, "ollama", args.judge, prompt)
            dt = time.time() - t0
            pr = cem.parse_judge_response(raw)
            print("  %-24s %s %6.1fs  think=%-7r %5d chars  parse_ok=%s  match=%s"
                  % (args.judge, "OK " if pr["parse_ok"] else "BAD", dt,
                     ollama_client.think_for(args.judge), len(raw or ""),
                     pr["parse_ok"], pr["match"]))
            if not pr["parse_ok"]:
                fail.append("judge '%s' returned no parseable <evaluation> block -- raise "
                            "JUDGE_MAX_TOKENS or check its think setting" % args.judge)
                print("      raw head: %r" % (raw or "")[:200])
            elif dt > 90 and args.mode == "full":
                print("      WARNING: %.0fs/call. Thinking may still be on." % dt)
        except Exception as e:  # noqa: BLE001
            print("  %-24s FAILED after %.0fs: %s" % (args.judge, time.time() - t0, e))
            fail.append("judge '%s' could not complete a call" % args.judge)
        ollama_client.unload(args.judge, args.host)
        stop_model(args.judge)

    for w in warn:
        print("\nWARNING: %s" % w)
    if fail:
        print("\nPREFLIGHT FAILED")
        for f in fail:
            print("  - %s" % f)
        return 1
    print("\nPreflight OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
