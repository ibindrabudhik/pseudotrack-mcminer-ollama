#!/bin/bash
# =============================================================================
#  RUN 1 -- the full McMiner pseudotrack, four arms.
#
#    MINED  by qwen3.6:27b   (local Ollama, 16K context, thinking off)
#    JUDGED by gpt-oss:20b   (local Ollama, 16K context, reasoning_effort=low)
#
#  Two different models, so neither one grades its own output.
#
#    SMOKE=1 bash run_all.sh      # 6 codes per arm -- do this first
#    bash run_all.sh              # full 209-code run
#    ARMS="baseline rag" bash run_all.sh
#
#  Arms are independent; a failure in one does not stop the others. Re-running
#  skips McMiner-M bag mining if bags already exist (FORCE=1 to rebuild), so an
#  interrupted run resumes cheaply.
# =============================================================================
set -uo pipefail
export RUN_MODE=full
source "$(dirname "${BASH_SOURCE[0]}")/scripts/_common.sh"

declare -a FAILED=() PASSED=()

echo "################################################################"
echo "#  RUN 1: full pseudotrack -- misconception bags + correct bags"
echo "#    miner : ${MODEL}"
echo "#    judge : ${JUDGE_MODEL}"
echo "#    arms  : ${ARMS}"
echo "################################################################"

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  # shellcheck disable=SC2086
  if ! "${PYTHON}" scripts/preflight.py --mode full --arms ${ARMS} \
        --miner "${MODEL}" --judge "${JUDGE_MODEL}" --host "${OLLAMA_HOST_URL}"; then
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
  echo "#  ARM: ${arm}"
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
"${PYTHON}" scripts/summarize.py --mode full --arms ${ARMS} --model "${MODEL}" || true

[[ ${#FAILED[@]} -eq 0 ]]
