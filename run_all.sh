#!/bin/bash
# =============================================================================
# Run all four McMiner pseudotrack arms (baseline, RAG, REF, RAG+REF) against a
# local Ollama model. Each arm runs BOTH McMiner-S and McMiner-M.
#
#   SMOKE=1 bash run_all.sh      # 6 codes per arm — do this first
#   bash run_all.sh              # full 209-code run
#
# Arms are independent; a failure in one does not stop the others (each is
# reported at the end). Re-running skips McMiner-M bag mining if bags already
# exist, so an interrupted run resumes cheaply.
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ARMS="${ARMS:-baseline rag ref rag_ref}"
declare -a FAILED=() PASSED=()

START=$(date +%s)
for arm in ${ARMS}; do
  echo
  echo "########################################################################"
  echo "#  ARM: ${arm}"
  echo "########################################################################"
  if bash "scripts/run_${arm}.sh"; then
    PASSED+=("${arm}")
  else
    echo "!! arm '${arm}' FAILED (exit $?) — continuing with the rest"
    FAILED+=("${arm}")
  fi
done
ELAPSED=$(( $(date +%s) - START ))

echo
echo "========================= SUMMARY ========================="
printf '  elapsed: %dh %dm %ds\n' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
[[ ${#PASSED[@]} -gt 0 ]] && echo "  ok     : ${PASSED[*]}"
[[ ${#FAILED[@]} -gt 0 ]] && echo "  failed : ${FAILED[*]}"
echo "  metrics under: results/evaluations/*/"
echo "==========================================================="

[[ ${#FAILED[@]} -eq 0 ]]
