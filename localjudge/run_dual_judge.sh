#!/bin/bash
# =============================================================================
#  Judge already-mined McMiner predictions with TWO local models, then compare.
#
#    bash run_dual_judge.sh                  # preflight, then both judges
#    DRY_RUN=1 bash run_dual_judge.sh        # preflight only, no judging
#    JUDGES="gpt-oss-judge" bash run_dual_judge.sh    # one judge only
#    ARMS="baseline" bash run_dual_judge.sh           # one arm only
#
#  Mining is NOT re-run. This reads predictions/ and writes results/.
#
#  Judges run STRICTLY ONE AT A TIME, and the previous model is unloaded before
#  the next one starts. On a 16 GB card, gpt-oss (~14 GB) and qwen3.6:27b
#  (~17 GB) cannot both be resident; leaving the first loaded pushes the second
#  much further into system RAM than it needs to go.
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/_common.sh"

JUDGES="${JUDGES:-gpt-oss-judge qwen36-judge}"
DRY_RUN="${DRY_RUN:-0}"

echo "================================================================"
echo "  LOCAL DUAL-JUDGE"
echo "  judges  : ${JUDGES}"
echo "  arms    : ${ARMS}"
echo "  endpoint: ${OPENROUTER_BASE_URL}"
echo "  inputs  : ${PRED_ROOT}/<arm>/"
echo "  outputs : ${OUT_ROOT}/<judge>/<arm>/"
echo "  budget  : JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS} abort_after=${JUDGE_ABORT_AFTER}"
echo "================================================================"

# shellcheck disable=SC2086
if ! "${PYTHON}" scripts/preflight.py --arms ${ARMS} --judges ${JUDGES} \
      --base-url "${OLLAMA_BASE_URL}" --pred-root "${PRED_ROOT}" \
      --max-tokens "${JUDGE_MAX_TOKENS}"; then
  echo
  echo "Preflight failed -- nothing was judged."
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "DRY_RUN=1 -- stopping before judging."
  exit 0
fi

declare -a OK=() BAD=()
START=$(date +%s)

for judge in ${JUDGES}; do
  echo
  echo "################################################################"
  echo "#  JUDGE: ${judge}"
  echo "################################################################"
  judge_env "${judge}"
  echo "  thinking control: LLM_REASONING_EFFORT='${LLM_REASONING_EFFORT:-}' LLM_EXTRA_BODY='${LLM_EXTRA_BODY:-}'"
  export JUDGE_PROVIDER="openrouter"
  export JUDGE_MODEL="${judge}"

  for arm in ${ARMS}; do
    src="${PRED_ROOT}/${arm}"
    out_s="${OUT_ROOT}/${judge}/${arm}/single_multi"
    out_m="${OUT_ROOT}/${judge}/${arm}/multi"
    mkdir -p "${out_s}" "${out_m}"
    echo
    echo "-- ${judge} / ${arm} ------------------------------------------"

    ok=1
    echo "   [1/2] McMiner-S"
    if ! "${PYTHON}" src/evaluate_single_multi_predictions.py \
          --grouped-predictions-file "${src}/single_multi/grouped_predictions.json" \
          --input-dir "${IN}" --misconceptions-file "${MISC}" \
          --output-dir "${out_s}"; then
      echo "   !! McMiner-S FAILED (${judge}/${arm})"; ok=0
    fi

    echo "   [2/2] McMiner-M"
    if ! "${PYTHON}" src/compute_eval_metrics_multi.py \
          --predictions-file "${src}/multi/multi_predictions.json" \
          --misconceptions-file "${MISC}" --input-dir "${IN}" \
          --output-dir "${out_m}" \
          --judge-provider openrouter --judge-model "${judge}"; then
      echo "   !! McMiner-M FAILED (${judge}/${arm})"; ok=0
    fi

    if [[ "${ok}" == "1" ]]; then OK+=("${judge}/${arm}"); else BAD+=("${judge}/${arm}"); fi
  done

  unload_model "${judge}"
done

ELAPSED=$(( $(date +%s) - START ))
echo
echo "========================= SUMMARY ========================="
printf '  elapsed: %dh %dm %ds\n' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
[[ ${#OK[@]}  -gt 0 ]] && echo "  ok     : ${OK[*]}"
[[ ${#BAD[@]} -gt 0 ]] && echo "  failed : ${BAD[*]}"
echo "==========================================================="

# shellcheck disable=SC2086
"${PYTHON}" scripts/compare_judges.py --judges ${JUDGES} --arms ${ARMS} --out-root "${OUT_ROOT}"

[[ ${#BAD[@]} -eq 0 ]]
