#!/bin/bash
# =============================================================================
#  FULL TRACK  --  misconception bags + correct bags, both miners, both judges.
#
#  This is the run where the JUDGE ACTUALLY DOES SOMETHING. Unlike
#  run_correct_only.sh (where ground truth is NONE and every bag is decided by
#  rule), misconception bags carry a ground-truth description, so each non-NONE
#  prediction is sent to the judge for a match / match_with_novel verdict.
#
#  TWO DIRECTIONS:
#     A : gpt-oss:20b  mines  ->  qwen3.6:27b  judges
#     B : qwen3.6:27b  mines  ->  gpt-oss:20b  judges
#
#  Neither model grades its own output, so the pair also measures how much of
#  any score is judge choice rather than miner skill: the same predictions in
#  direction A are judged by the model that produced direction B's, and vice
#  versa.
#
#  COST WARNING. Per arm per direction this is ~37 bag-mining calls + 209
#  corrupted-code calls + 96 correct-code calls + up to ~165 judge calls.
#  Across 4 arms x 2 directions that is roughly 4,000 model calls. On a 5070 Ti
#  budget for DAYS, not hours, and run it one arm at a time:
#
#     SMOKE=1 bash run_full.sh                        # always first
#     DIRECTIONS=A ARMS=baseline bash run_full.sh     # one cell at a time
#     bash run_full.sh                                # everything (very long)
#
#  Arms are independent and McMiner-M bag mining is skipped when bags already
#  exist (FORCE=1 to rebuild), so an interrupted run resumes cheaply.
#
#  BEFORE QUOTING ANY NUMBER from this run, check the judge health lines in the
#  summary. A previous qwen-as-judge run silently failed 31% of its calls and
#  scored every failure as a non-match. See README "What went wrong before".
# =============================================================================
set -uo pipefail
export RUN_MODE=full

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DIRECTIONS="${DIRECTIONS:-A B}"
ARMS="${ARMS:-baseline rag ref rag_ref}"
PYTHON="${PYTHON:-python}"

declare -a FAILED=() PASSED=()
START=$(date +%s)

echo "################################################################"
echo "#  FULL TRACK  (misconception bags + correct bags)"
echo "#    directions: ${DIRECTIONS}"
echo "#    arms      : ${ARMS}"
echo "#    judge     : ACTIVE (misconception bags are judged)"
echo "################################################################"

for d in ${DIRECTIONS}; do
  export DIRECTION="${d}"
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
  echo "################################################################"

  if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    # shellcheck disable=SC2086
    if ! "${PYTHON}" scripts/preflight.py --mode full --arms ${ARMS} \
          --miner "${MINER}" --judge "${JUDGE}" \
          --host "${OLLAMA_HOST_URL:-http://localhost:11434}"; then
      echo "Preflight failed for direction ${d} -- skipping it."
      FAILED+=("${d}/preflight"); continue
    fi
  fi

  [[ "${DRY_RUN:-0}" == "1" ]] && { echo "DRY_RUN=1 -- stopping before mining."; continue; }

  for arm in ${ARMS}; do
    echo
    echo "======================================================================"
    echo "  DIRECTION ${d} / ARM ${arm}"
    echo "======================================================================"
    unset MODEL_TAG
    if bash "scripts/run_${arm}.sh"; then
      PASSED+=("${d}/${arm}")
    else
      echo "!! ${d}/${arm} FAILED (exit $?) -- continuing"
      FAILED+=("${d}/${arm}")
    fi
  done

  echo
  echo "---- summary for direction ${d} (${LABEL}) ----"
  # shellcheck disable=SC2086
  "${PYTHON}" scripts/summarize.py --mode full --arms ${ARMS} \
      --model "${MINER}" --direction "${d}" || true
done

ELAPSED=$(( $(date +%s) - START ))
echo
echo "========================= OVERALL SUMMARY ========================="
printf '  elapsed: %dh %dm %ds\n' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
[[ ${#PASSED[@]} -gt 0 ]] && echo "  ok     : ${PASSED[*]}"
[[ ${#FAILED[@]} -gt 0 ]] && echo "  failed : ${FAILED[*]}"
echo "  Check judge-failure counts before quoting anything."
echo "==================================================================="

[[ ${#FAILED[@]} -eq 0 ]]
