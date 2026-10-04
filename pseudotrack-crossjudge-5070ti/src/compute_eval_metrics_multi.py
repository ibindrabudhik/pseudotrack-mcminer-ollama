#!/usr/bin/env python3
"""
LLM-as-judge scorer for mined-misconception predictions (pseudocode track).

This is the judge component invoked by `evaluate_single_multi_predictions.py`
(via `--use-claude-eval`). For each misconception prediction it asks a judge LLM
whether the model's predicted misconception(s) match the ground-truth
misconception exhibited by the pseudocode, producing both a strict `match` and a
novelty-aware `match_with_novel` verdict.

It writes `<output-dir>/claude_evaluation_results.json` in the exact schema that
`evaluate_single_multi_predictions.py` reads:

    {
      "evaluation_details": [
        {"prediction_id": ..., "match": bool, "match_with_novel": bool,
         "confidence": "high|medium|low", "rationale": ..., "method": ...},
        ...
      ],
      "summary": {...}
    }

The judge is a local Ollama model ($JUDGE_MODEL, or --judge-model). There is no
provider switch and no API key: this talks to Ollama's native /api/chat through
utils/ollama_client.py.

Note for anyone comparing against the paper: its main table uses GPT-5 as judge,
so numbers produced here are NOT comparable to it. What this bundle buys instead
is that the judge is a different model from the miner, which the earlier local
runs were not.

Correct-only bags never reach the judge at all -- their ground truth is NONE, so
they are decided by rule (method "correct_bag_rule").

Usage (normally called for you by the evaluator, but runnable standalone):
    python src/compute_eval_metrics_multi.py \
        --predictions-file <single/predictions.json> \
        --misconceptions-file dataset/pseudocode_track/misconceptions_22.json \
        --input-dir dataset/pseudocode_track/pseudocode_codes \
        --output-dir temp_eval_output \
        --use-claude-eval \
        --judge-model gpt-oss-judge:latest
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ollama_client
from utils.ollama_client import OllamaClient, OllamaError


# ------------------------------------------------------------------ templates
def load_judge_template() -> str:
    """Load the prediction-match judge prompt."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        script_dir, "prompt_templates", "evaluation-pseudocode", "judge_prediction_match.md"
    )
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # Drop a leading markdown title line if present.
    lines = content.split("\n")
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
        if lines and lines[0].strip() == "":
            lines = lines[1:]
    return "\n".join(lines)


# ------------------------------------------------------------------- loading
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_code_index(input_dir: str) -> Dict[str, Dict[str, Any]]:
    """Map source filename -> loaded code file (for pulling generated_code)."""
    index = {}
    for filename in os.listdir(input_dir):
        if filename.endswith(".json") and filename not in ("summary.json", "filtering_report.json"):
            index[filename] = load_json(os.path.join(input_dir, filename))
    return index


def get_code_for_prediction(pred: Dict[str, Any], code_index: Dict[str, Dict[str, Any]]) -> str:
    """Recover the pseudocode string a prediction was made on.

    Single-prediction format: one `source_file` (+ optional `solution_index`).
    Whole-bag (multi) format: `group_info.source_files` — concatenate every code in the bag.
    """
    # Multi / whole-bag: concatenate the bag's codes.
    source_files = (pred.get("group_info", {}) or {}).get("source_files")
    if source_files:
        blocks = []
        for i, sf in enumerate(source_files, 1):
            data = code_index.get(sf)
            if not data:
                continue
            sols = data.get("solutions", [])
            code = (sols[0].get("generated_code", "") if sols else "") or ""
            if code:
                blocks.append(f"# Code {i}\n{code}")
        return "\n\n".join(blocks)

    # Single prediction.
    source_file = pred.get("source_file", "")
    sol_idx = pred.get("solution_index", 0)
    data = code_index.get(source_file)
    if not data:
        return ""
    solutions = data.get("solutions", [])
    if 0 <= sol_idx < len(solutions):
        return solutions[sol_idx].get("generated_code", "") or ""
    if solutions:
        return solutions[0].get("generated_code", "") or ""
    return ""


def load_misc_map(misconceptions_file: str) -> Dict[int, str]:
    """misconception_id -> description, from the misconception bank (list or dict shaped)."""
    data = load_json(misconceptions_file)
    items = data.values() if isinstance(data, dict) else data
    out = {}
    for m in items:
        try:
            out[int(m["id"])] = (m.get("description") or "").strip()
        except (KeyError, ValueError, TypeError):
            continue
    return out


def resolve_gt(pred: Dict[str, Any], misc_map: Dict[int, str]) -> Tuple[str, bool]:
    """Return (gt_description, is_correct_only_bag) for a prediction, supporting both formats.

    - Single format carries `ground_truth_misconception.description` directly.
    - Whole-bag format carries `group_type` + `misconception_id` (GT id); correct-only bags have
      gt == NONE, in which case the correct answer is 'predicted no misconception'.
    """
    # Correct-only bag (whole-bag format): ground truth is NONE.
    group_type = pred.get("group_type")
    gt_field = (pred.get("group_info", {}) or {}).get("gt_misconception")
    if group_type == "correct_only" or gt_field == "NONE":
        return "", True

    # Single format: explicit ground_truth_misconception.
    gt = pred.get("ground_truth_misconception", {}) or {}
    if gt.get("description"):
        return gt["description"].strip(), False

    # Whole-bag misconception bag: derive description from misconception_id via the bank.
    mid = pred.get("misconception_id")
    try:
        return misc_map.get(int(mid), ""), False
    except (TypeError, ValueError):
        return "", False


def format_predicted(pred: Dict[str, Any]) -> str:
    """Render the model's predicted misconception(s) for the judge prompt."""
    items = pred.get("predicted_misconceptions", []) or []
    if not items:
        return "(the model predicted NO misconception)"
    blocks = []
    for i, m in enumerate(items, 1):
        desc = (m.get("description") or "").strip()
        expl = (m.get("explanation") or "").strip()
        block = f"{i}. {desc}"
        if expl:
            block += f"\n   Explanation: {expl}"
        blocks.append(block)
    return "\n".join(blocks)


# --------------------------------------------------------------------- judge
def create_judge_client(provider: str, model: str):
    """The judge is a local Ollama model. `provider` is accepted and ignored,
    so the inherited call sites keep working; there is only one backend."""
    return OllamaClient(model=model, host=os.getenv("OLLAMA_HOST_URL"))


def call_judge(client, provider: str, model: str, prompt: str) -> str:
    """Send one judge prompt; return the raw text response.

    JUDGE_MAX_TOKENS raises the 1500-token default. Reasoning judges spend part
    of this budget thinking before they answer, and when it runs out the reply
    is truncated or empty -- which parse_judge_response would then turn into a
    score via its non-neutral defaults. The client raises on an empty reply
    rather than returning "", so that failure surfaces as a judge error instead
    of a fabricated verdict.

    JUDGE_TEMPERATURE set to an empty string omits temperature entirely.
    """
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": model, "max_tokens": int(os.getenv("JUDGE_MAX_TOKENS", "1500"))}
    temp = os.getenv("JUDGE_TEMPERATURE", "0.0")
    if temp != "":
        kwargs["temperature"] = float(temp)
    return client.create_message(messages, kwargs=kwargs)


def parse_judge_response(text: str) -> Dict[str, Any]:
    """Parse the <evaluation> block into match / match_with_novel / confidence.

    Also reports `parse_ok`: whether the judge actually emitted the tags, rather
    than the defaults being used. This matters because the defaults are not
    neutral — a response truncated before </match> scores as a non-match, and
    one truncated after it inherits match into match_with_novel. Either way a
    clipped judge reply silently becomes a score. `parse_ok` makes that
    countable instead of invisible; see the parse-failure warning in main().
    """
    found = {}

    def _yn(tag: str, default: bool = False) -> bool:
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", re.DOTALL | re.IGNORECASE)
        found[tag] = m is not None
        if not m:
            return default
        return m.group(1).strip().upper().startswith("Y")

    conf_m = re.search(r"<confidence>\s*(.*?)\s*</confidence>", text or "", re.DOTALL | re.IGNORECASE)
    rat_m = re.search(r"<rationale>\s*(.*?)\s*</rationale>", text or "", re.DOTALL | re.IGNORECASE)

    match = _yn("match", False)
    match_with_novel = _yn("match_with_novel", match)
    # match_with_novel is by definition >= match.
    match_with_novel = match_with_novel or match
    return {
        "match": match,
        "match_with_novel": match_with_novel,
        "confidence": (conf_m.group(1).strip().lower() if conf_m else "unknown"),
        "rationale": (rat_m.group(1).strip() if rat_m else ""),
        # Both verdicts must be present. Requiring only <match> would pass a
        # reply truncated between the two tags, which then silently inherits
        # match into match_with_novel via the rule above.
        "parse_ok": bool(found.get("match") and found.get("match_with_novel")),
    }


# ---------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-as-judge scorer for mined misconceptions")
    parser.add_argument("--predictions-file", required=True,
                        help="Single predictions.json (subset) to judge")
    parser.add_argument("--misconceptions-file", required=True,
                        help="Misconception bank JSON (id/description)")
    parser.add_argument("--input-dir", required=True,
                        help="Directory of the code files the predictions were made on")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write claude_evaluation_results.json")
    parser.add_argument("--use-claude-eval", action="store_true",
                        help="Accepted for compatibility with the evaluator (always LLM-judged)")
    parser.add_argument("--judge-provider",
                        default="ollama", choices=["ollama"],
                        help="Kept so existing call sites keep working; the only backend "
                             "is local Ollama.")
    parser.add_argument("--judge-model",
                        default=os.getenv("JUDGE_MODEL", "gpt-oss-judge:latest"),
                        help="Judge model tag (default: $JUDGE_MODEL)")
    args = parser.parse_args()

    predictions = load_json(args.predictions_file)
    code_index = build_code_index(args.input_dir)
    template = load_judge_template()
    misc_map = load_misc_map(args.misconceptions_file)
    # Whole-bag (multi) predictions carry group_type; single predictions carry ground_truth_misconception.
    is_multi = any("group_type" in p for p in predictions)
    if is_multi:
        print(f"📦 Detected whole-bag (McMiner-M) predictions: deriving ground truth from "
              f"misconception_id via {os.path.basename(args.misconceptions_file)}; "
              f"correct-only bags score as match when the model predicts NONE.")

    # Only misconception predictions (with a real GT) need judging; empty
    # predictions are auto-scored as non-match without an API call.
    judge_endpoint = os.getenv("OLLAMA_HOST_URL", "http://localhost:11434")
    print(f"🔌 JUDGE -> model={args.judge_model}  host={judge_endpoint}  "
          f"think={ollama_client.think_for(args.judge_model)!r}")

    # Mirror the loop's own skip rules exactly. This banner used to count only
    # empty predictions, so a correct-only bag that DID predict something was
    # announced as "will be sent to the judge" and then rule-scored instead --
    # the printed plan disagreed with what the run actually did. In a
    # correct-bags-only run every line of it was wrong.
    empty_count = correct_bag_count = 0
    for p in predictions:
        _, is_correct_bag = resolve_gt(p, misc_map)
        if is_correct_bag:
            correct_bag_count += 1
        elif (p.get("no_predicted_misconceptions", False)
              or len(p.get("predicted_misconceptions", []) or []) == 0):
            empty_count += 1
    to_judge_count = len(predictions) - empty_count - correct_bag_count
    print(f"Judging {len(predictions)} predictions: "
          f"{correct_bag_count} correct-only bags scored by rule (ground truth is NONE, no API call), "
          f"{empty_count} auto-scored as non-match (miner predicted NONE, no API call), "
          f"{to_judge_count} will be sent to {args.judge_provider}/{args.judge_model} ...")
    if to_judge_count == 0:
        if correct_bag_count == len(predictions):
            print("ℹ️  No LLM judge call is needed this run: every prediction is a correct-only "
                  "bag, whose ground truth is NONE. Both scorers decide those by rule, so the "
                  f"result does not depend on {args.judge_model} at all.")
        else:
            print("⚠️  No predictions require an LLM judge call this run — that's why you won't see any "
                  f"{args.judge_model} usage. All misconception predictions here were empty (the miner "
                  "found nothing to judge). This is common with small --max-files/--max-requests smoke tests.")
    client = None

    # Abort on a run of consecutive judge failures instead of grinding through
    # every prediction. An exhausted balance, a revoked key or a bad model id
    # fails EVERY remaining call, and the handler below scores each failure as
    # match=False -- so continuing does not degrade gracefully, it manufactures a
    # plausible-looking low score. Observed for real: a credit exhaustion partway
    # through wrote a 0.00% match rate from 106 consecutive 429s.
    abort_after = int(os.getenv("JUDGE_ABORT_AFTER", "5"))
    consecutive_errors = 0

    details: List[Dict[str, Any]] = []
    for pred in tqdm(predictions, desc="Judging predictions"):
        pred_id = pred.get("prediction_id")
        gt_desc, is_correct_bag = resolve_gt(pred, misc_map)
        group_type = pred.get("group_type", "misconception" if not is_correct_bag else "correct_only")
        predicted_items = pred.get("predicted_misconceptions", []) or []
        no_pred = pred.get("no_predicted_misconceptions", False) or len(predicted_items) == 0

        # Correct-only bag: ground truth is NONE, so predicting NO misconception IS correct.
        if is_correct_bag:
            details.append({
                "prediction_id": pred_id,
                "group_type": group_type,
                "match": bool(no_pred),
                "match_with_novel": bool(no_pred),
                "confidence": "high",
                "rationale": ("Correct-only bag: model correctly predicted no misconception."
                              if no_pred else
                              "Correct-only bag: model predicted a misconception (false positive)."),
                "method": "correct_bag_rule",
            })
            continue

        if no_pred:
            details.append({
                "prediction_id": pred_id,
                "group_type": group_type,
                "match": False,
                "match_with_novel": False,
                "confidence": "high",
                "rationale": "Model predicted no misconception.",
                "method": "empty_prediction",
            })
            continue

        code = get_code_for_prediction(pred, code_index)
        prompt = (template
                  .replace("{code}", code)
                  .replace("{gt_description}", gt_desc)
                  .replace("{predicted}", format_predicted(pred)))

        if client is None:
            client = create_judge_client(args.judge_provider, args.judge_model)

        try:
            raw = call_judge(client, args.judge_provider, args.judge_model, prompt)
            parsed = parse_judge_response(raw)
            parsed["method"] = "llm_judge"
            # Keep the raw reply whenever the tags were missing, so a run with a
            # bad token budget or a chatty judge can be diagnosed after the fact
            # instead of just producing quietly wrong numbers.
            if not parsed.get("parse_ok"):
                parsed["raw_response"] = (raw or "")[:2000]
            consecutive_errors = 0
        except Exception as e:  # noqa: BLE001 - one bad call shouldn't kill the run
            print(f"  ! judge error on {pred_id}: {e}")
            parsed = {"match": False, "match_with_novel": False,
                      "confidence": "unknown", "rationale": f"judge error: {e}",
                      "method": "judge_error", "parse_ok": False}
            consecutive_errors += 1
            if abort_after > 0 and consecutive_errors >= abort_after:
                print(f"\n❌ ABORTING: {consecutive_errors} consecutive judge failures.")
                print("   Every remaining call would fail the same way and be scored as a")
                print("   non-match, producing a wrong score rather than an error. Nothing")
                print(f"   was written to {args.output_dir}.")
                print(f"   Last error: {e}")
                print("   Fix the cause (credits / key / model id), then re-run this arm.")
                print("   Set JUDGE_ABORT_AFTER=0 to disable this guard.")
                return 1

        parsed["prediction_id"] = pred_id
        parsed["group_type"] = group_type
        details.append(parsed)

    matched = sum(1 for d in details if d["match"])
    matched_novel = sum(1 for d in details if d["match_with_novel"])
    judged = [d for d in details if d.get("method") in ("llm_judge", "judge_error")]
    unparsed = [d for d in judged if not d.get("parse_ok")]
    result = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "summary": {
            "total": len(details),
            "match": matched,
            "match_with_novel": matched_novel,
            "match_rate": matched / len(details) if details else 0.0,
            "match_with_novel_rate": matched_novel / len(details) if details else 0.0,
            "judge_calls": len(judged),
            "judge_parse_failures": len(unparsed),
            "judge_parse_success_rate": (1 - len(unparsed) / len(judged)) if judged else 1.0,
        },
        "evaluation_details": details,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "claude_evaluation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_file}: {matched}/{len(details)} match, "
          f"{matched_novel}/{len(details)} match_with_novel")
    if unparsed:
        print(f"⚠️  {len(unparsed)}/{len(judged)} judge replies had no parseable <evaluation> "
              f"block and fell back to defaults — these scores are NOT trustworthy. "
              f"Raw replies are kept under evaluation_details[].raw_response. Most likely "
              f"cause: max_tokens (1500) consumed by the judge's reasoning tokens before it "
              f"reached the answer.")

    # For whole-bag (McMiner-M) runs, also emit an evaluation_metrics.json with the same
    # overall / misconception / correct_only breakdown schema as the baseline multi evals, so the
    # numbers slot directly into the summary tooling.
    if is_multi:
        def _rate(sub, key):
            return (sum(1 for d in sub if d[key]) / len(sub)) if sub else 0.0
        misc = [d for d in details if d.get("group_type") != "correct_only"]
        corr = [d for d in details if d.get("group_type") == "correct_only"]
        metrics = {
            "standard_metrics": {"overall_metrics": {
                "total_bags": len(details),
                "correct_bags": matched,
                "overall_accuracy": _rate(details, "match"),
                "misconception_accuracy": _rate(misc, "match"),
                "correct_only_accuracy": _rate(corr, "match"),
                "misconception_count": len(misc),
                "correct_only_count": len(corr),
            }},
            "with_novel_metrics": {"overall_metrics": {
                "total_bags": len(details),
                "correct_bags": matched_novel,
                "overall_accuracy": _rate(details, "match_with_novel"),
                "misconception_accuracy": _rate(misc, "match_with_novel"),
                "correct_only_accuracy": _rate(corr, "match_with_novel"),
                "misconception_count": len(misc),
                "correct_only_count": len(corr),
            }},
        }
        metrics_file = os.path.join(args.output_dir, "evaluation_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        sm = metrics["standard_metrics"]["overall_metrics"]
        nm = metrics["with_novel_metrics"]["overall_metrics"]
        print(f"Wrote {metrics_file}")
        print(f"  STD overall={sm['overall_accuracy']:.3f} misc={sm['misconception_accuracy']:.3f} "
              f"correct={sm['correct_only_accuracy']:.3f}")
        print(f"  NOV overall={nm['overall_accuracy']:.3f} misc={nm['misconception_accuracy']:.3f} "
              f"correct={nm['correct_only_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
