#!/bin/bash
# =============================================================================
#  CORRECT-BAGS-ONLY RUN  --  the false-positive / specificity experiment.
#
#  Takes every CORRECT program in the dataset, partitions them into bags exactly
#  the way McMiner's bag former does, mines each bag and each individual correct
#  code, and scores the result. Ground truth for every bag is NONE, so a bag
#  counts as correct only if the model predicts *no* misconception.
#
#  This measures exactly one thing: how often each miner invents a misconception
#  in code that has nothing wrong with it.
#
#  TWO DIRECTIONS, run back to back by default:
#     A : gpt-oss:20b  mines  ->  qwen3.6:27b  judges
#     B : qwen3.6:27b  mines  ->  gpt-oss:20b  judges
#
#  ---------------------------------------------------------------------------
#  READ THIS FIRST: in a correct-only run the judge makes ZERO calls.
#
#  Ground truth is NONE, so both scorers decide by rule, not by asking a model:
#      compute_eval_metrics_multi.py        -> method "correct_bag_rule"
#      evaluate_single_multi_predictions.py -> method "empty_check"
#  A judge is only consulted when there is a ground-truth misconception
#  description to compare a prediction against, and here there is none.
#
#  So A and B differ ONLY in the miner. Both judges are still built,
#  preflighted and probed (so the same folder can run the full track later,
#  where the judge does matter), but neither is asked anything, and these
#  results are completely free of judge choice and judge bias.
#
#  That is a feature: it is the cleanest comparison in the whole project.
#  Use run_full.sh when you want the judge to actually do something.
#  ---------------------------------------------------------------------------
#
#  Usage:
#    SMOKE=1 bash run_correct_only.sh         # 6 codes/bags per arm -- DO THIS FIRST
#    bash run_correct_only.sh                 # both directions, all four arms
#    DIRECTIONS=A bash run_correct_only.sh    # just gpt-oss mining
#    ARMS=baseline bash run_correct_only.sh   # just the baseline arm
#    CORRECT_PASSES=5 bash run_correct_only.sh  # 5 shufflings -> ~20 bags/arm
#
#  Scale per arm per direction (defaults): 4 bags + 96 single correct codes
#  = 100 mining calls. Those 96 files are only 19 distinct programs -- see
#  README "Why 96 correct files are 19 programs".
# =============================================================================
set -uo pipefail
export RUN_MODE=correct_only

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DIRECTIONS="${DIRECTIONS:-A B}"
ARMS="${ARMS:-baseline rag ref rag_ref}"
CORRECT_PASSES="${CORRECT_PASSES:-1}"
PYTHON="${PYTHON:-python}"
export CORRECT_PASSES

declare -a FAILED=() PASSED=()
START=$(date +%s)

echo "################################################################"
echo "#  CORRECT-BAGS-ONLY RUN  (false-positive control)"
echo "#    directions : ${DIRECTIONS}"
echo "#    arms       : ${ARMS}"
echo "#    bagging    : cover-all, ${CORRECT_PASSES} pass(es) over every correct program"
echo "#    judge calls: 0 expected -- decided by rule, see header"
echo "################################################################"

for d in ${DIRECTIONS}; do
  export DIRECTION="${d}"

  # Resolve this direction's model names for the preflight + summary, using the
  # same mapping _common.sh uses. Kept in sync deliberately: a mismatch here
  # would preflight one model and mine with another.
  if [[ "${d}" == "A" ]]; then
    MINER="${MODEL_A:-gpt-oss-mcminer:latest}";  JUDGE="${JUDGE_A:-qwen36-judge:latest}"
    LABEL="A: gpt-oss mines -> qwen3.6 judges"
  else
    MINER="${MODEL_B:-qwen3.6-mcminer:latest}"; JUDGE="${JUDGE_B:-gpt-oss-judge:latest}"
    LABEL="B: qwen3.6 mines -> gpt-oss judges"
  fi
  export MODEL="${MINER}" JUDGE_MODEL="${JUDGE}"

  echo
  echo "################################################################"
  echo "#  DIRECTION ${LABEL}"
  echo "#    miner: ${MINER}"
  echo "#    judge: ${JUDGE}   [INERT in correct-only mode]"
  echo "################################################################"

  if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    # shellcheck disable=SC2086
    if ! "${PYTHON}" scripts/preflight.py --mode correct_only --arms ${ARMS} \
          --miner "${MINER}" --judge "${JUDGE}" \
          --host "${OLLAMA_HOST_URL:-http://localhost:11434}" \
          --correct-passes "${CORRECT_PASSES}"; then
      echo
      echo "Preflight failed for direction ${d} -- skipping it. (SKIP_PREFLIGHT=1 to bypass.)"
      FAILED+=("${d}/preflight")
      continue
    fi
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1 -- stopping before mining."
    continue
  fi

  for arm in ${ARMS}; do
    echo
    echo "======================================================================"
    echo "  DIRECTION ${d} / ARM ${arm}"
    echo "======================================================================"
    # MODEL_TAG must not leak between arms: _common.sh only defaults it.
    unset MODEL_TAG
    if bash "scripts/run_${arm}.sh"; then
      PASSED+=("${d}/${arm}")
    else
      echo "!! ${d}/${arm} FAILED (exit $?) -- continuing with the rest"
      FAILED+=("${d}/${arm}")
    fi
  done

  echo
  echo "---- summary for direction ${d} (${LABEL}) ----"
  # shellcheck disable=SC2086
  "${PYTHON}" scripts/summarize.py --mode correct_only --arms ${ARMS} \
      --model "${MINER}" --direction "${d}" || true
done

ELAPSED=$(( $(date +%s) - START ))

echo
echo "========================= OVERALL SUMMARY ========================="
printf '  elapsed: %dh %dm %ds\n' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
[[ ${#PASSED[@]} -gt 0 ]] && echo "  ok     : ${PASSED[*]}"
[[ ${#FAILED[@]} -gt 0 ]] && echo "  failed : ${FAILED[*]}"
echo "==================================================================="

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  echo
  # shellcheck disable=SC2086
  "${PYTHON}" scripts/compare_miners.py --arms ${ARMS} || true
fi

[[ ${#FAILED[@]} -eq 0 ]]
