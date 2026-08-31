#!/bin/bash
# =============================================================================
#  RUN 2 -- correct-only bags, mined and checked through the SAME pipeline.
#
#  Takes every correct program in the dataset, partitions them into bags exactly
#  the way McMiner's bag former does (create_correct_only_bags(cover_all=True) in
#  src/run_infer_misc_multi.py), mines each bag with qwen3.6:27b, and scores the
#  result with the same two evaluators run 1 uses.
#
#  Ground truth for every bag is NONE. A bag scores as a match only if the model
#  predicts *no* misconception -- so this run measures exactly one thing: the
#  false-positive rate on code that has nothing wrong with it.
#
#    SMOKE=1 bash run_correct_only.sh    # 6 codes / 6 bags -- do this first
#    bash run_correct_only.sh            # all four arms
#    CORRECT_PASSES=5 bash run_correct_only.sh   # 5 shufflings -> ~20 bags/arm
#
#  ---------------------------------------------------------------------------
#  READ THIS: this run makes ZERO judge calls.
#
#  Both scorers decide correct-only cases by rule, not by asking a model:
#    compute_eval_metrics_multi.py    -> method "correct_bag_rule"
#    evaluate_single_multi_predictions.py -> method "empty_check"
#  An LLM judge is only consulted when there is a ground-truth misconception
#  description to compare a prediction against, and here there is none.
#
#  So gpt-oss is still configured, preflighted and probed -- but it will not be
#  asked anything, and the numbers this run produces are deterministic given the
#  mined predictions. That is a property of the metric, not a bug, and it means
#  these results are NOT affected by judge choice at all.
#  ---------------------------------------------------------------------------
#
#  Scale per arm (defaults): 4 bags + 96 single correct codes = 100 mining
#  calls. Those 96 files are only 19 distinct programs -- see README "Why 96
#  correct files are 19 programs".
# =============================================================================
set -uo pipefail
export RUN_MODE=correct_only
source "$(dirname "${BASH_SOURCE[0]}")/scripts/_common.sh"

declare -a FAILED=() PASSED=()

echo "################################################################"
echo "#  RUN 2: correct-only bags (false-positive control)"
echo "#    miner  : ${MODEL}"
echo "#    judge  : ${JUDGE_MODEL}  (configured; 0 calls expected -- see header)"
echo "#    arms   : ${ARMS}"
echo "#    bagging: cover-all, ${CORRECT_PASSES} pass(es) over every correct program"
echo "################################################################"

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  # shellcheck disable=SC2086
  if ! "${PYTHON}" scripts/preflight.py --mode correct_only --arms ${ARMS} \
        --miner "${MODEL}" --judge "${JUDGE_MODEL}" --host "${OLLAMA_HOST_URL}" \
        --correct-passes "${CORRECT_PASSES}"; then
    echo
    echo "Preflight failed -- nothing was run. (SKIP_PREFLIGHT=1 to bypass.)"
    exit 1
  fi
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo
  echo "DRY_RUN=1 -- stopping before mining."
  exit 0
fi

START=$(date +%s)
for arm in ${ARMS}; do
  echo
  echo "########################################################################"
  echo "#  ARM: ${arm}  (correct-only bags)"
  echo "########################################################################"
  if bash "scripts/run_${arm}.sh"; then
    PASSED+=("${arm}")
  else
    echo "!! arm '${arm}' FAILED (exit $?) -- continuing with the rest"
    FAILED+=("${arm}")
  fi
done
ELAPSED=$(( $(date +%s) - START ))

echo
echo "========================= SUMMARY ========================="
printf '  elapsed: %dh %dm %ds\n' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
[[ ${#PASSED[@]} -gt 0 ]] && echo "  ok     : ${PASSED[*]}"
[[ ${#FAILED[@]} -gt 0 ]] && echo "  failed : ${FAILED[*]}"
echo "==========================================================="

# shellcheck disable=SC2086
"${PYTHON}" scripts/summarize.py --mode correct_only --arms ${ARMS} --model "${MODEL}" || true

[[ ${#FAILED[@]} -eq 0 ]]
