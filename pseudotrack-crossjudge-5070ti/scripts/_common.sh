#!/bin/bash
# =============================================================================
# Shared config + pipeline body for the cross-judge pseudotrack bundle.
#
# TWO DIRECTIONS, selected by $DIRECTION:
#
#   A : gpt-oss:20b   mines  ->  qwen3.6:27b  judges
#   B : qwen3.6:27b   mines  ->  gpt-oss:20b  judges
#
# Neither model ever grades its own output in either direction.
#
# LOCAL OLLAMA ONLY. No provider switch, no API key, no OpenAI-compatible shim:
# src/ talks to Ollama's native /api/chat through utils/ollama_client.py, whose
# only dependency is `requests`.
#
# >>> READ THIS BEFORE INTERPRETING A CORRECT-ONLY RUN <<<
# In RUN_MODE=correct_only the judge makes ZERO calls, in BOTH directions.
# Ground truth for every bag is NONE, so both scorers decide by rule:
#     compute_eval_metrics_multi.py       -> method "correct_bag_rule"
#     evaluate_single_multi_predictions.py -> method "empty_check"
# A judge is only consulted when there is a ground-truth misconception
# description to compare a prediction against, and here there is none.
# Consequence: A and B differ ONLY in the miner. The judge is configured,
# preflighted and verified, but is never asked anything, and the numbers are
# deterministic given the mined predictions. See README "Why the judge is
# inert".
#
# This file is SOURCED by scripts/run_<arm>.sh and by the entry points.
# Do not run it directly.
# =============================================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# -----------------------------------------------------------------------------
#  Server
# -----------------------------------------------------------------------------
# Exported because the judge client reads it directly (create_judge_client in
# src/compute_eval_metrics_multi.py); the mining steps take it on the command
# line as --ollama-host.
export OLLAMA_HOST_URL="${OLLAMA_HOST_URL:-http://localhost:11434}"

# Request timeout, seconds. The client (utils/ollama_client.py) reads this.
# 900 s is deliberate: qwen3.6:27b partially offloaded to RAM has been measured
# at ~160 s for a judge call and several minutes for a hard one. The earlier
# OpenAI-SDK-based bundle had NO configurable timeout and inherited a 600 s x 3
# retry default, which turned slow calls into 30-minute stalls scored as
# non-matches -- that is the single defect that invalidated the previous qwen
# judge run. Do not lower this without reading README "What went wrong before".
export OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-900}"

# -----------------------------------------------------------------------------
#  Direction -> (miner, judge)
# -----------------------------------------------------------------------------
DIRECTION="${DIRECTION:-A}"
case "${DIRECTION}" in
  A|a)
    DIRECTION="A"
    MODEL="${MODEL:-gpt-oss-mcminer:latest}"
    export JUDGE_MODEL="${JUDGE_MODEL:-qwen36-judge:latest}"
    DIR_LABEL="A: gpt-oss mines -> qwen3.6 judges"
    ;;
  B|b)
    DIRECTION="B"
    MODEL="${MODEL:-qwen3.6-mcminer:latest}"
    export JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss-judge:latest}"
    DIR_LABEL="B: qwen3.6 mines -> gpt-oss judges"
    ;;
  *)
    echo "ERROR: DIRECTION must be A or B (got '${DIRECTION}')"; exit 1 ;;
esac
export DIRECTION

# -----------------------------------------------------------------------------
#  Reasoning control -- decided per model, in one place
# -----------------------------------------------------------------------------
# Both models reason before answering, and those tokens come out of the SAME
# budget as the answer. When the budget runs out the reply is truncated -- and
# the parsers' defaults are not neutral, so a clipped reply silently becomes a
# *score*, or a "no misconception predicted", rather than an error.
#
# The right setting differs by model family, and `think: false` is NOT the
# universal off switch. Measured on Ollama 0.32.14:
#
#   gpt-oss, think=false -> content='', thinking='The user says...',
#                           done_reason='length'. Reasoned until the budget ran
#                           out and returned NOTHING.
#   gpt-oss, think='low' -> answered in 17 tokens, 5.5 s.
#   qwen3.6, think=true  -> >10 min per call; think=false -> ~160 s.
#
# A reasoning-only model needs a LEVEL; qwen3.6 needs the boolean. That mapping
# lives in THINK_BY_MODEL in utils/ollama_client.py and is applied by model
# name, so there is nothing to export here and nothing that can leak from one
# step into the next.
#
# To measure the cost of thinking, override for a whole run:
#   OLLAMA_THINK=medium bash run_correct_only.sh   # force a level
#   OLLAMA_THINK=default bash run_correct_only.sh  # each model's own default
[[ -n "${OLLAMA_THINK:-}" ]] && export OLLAMA_THINK

# Free memory when switching between the two models. Ollama holds a model
# resident for 5 minutes after the last request; 17 GB (qwen) and 13.8 GB
# (gpt-oss) cannot co-reside on a 16 GB card, and leaving the first loaded
# forces the second much further into system RAM than it needs to go.
unload_model() {
  local m="$1"
  [[ "${UNLOAD_BETWEEN:-1}" == "1" ]] || return 0
  if ollama ps 2>/dev/null | grep -q "${m%%:*}"; then
    echo "   unloading ${m} to free VRAM..."
    ollama stop "${m}" >/dev/null 2>&1 || true
    sleep 2
  fi
}

# -----------------------------------------------------------------------------
#  Judge guardrails  (inert in correct-only mode -- see header)
# -----------------------------------------------------------------------------
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-3000}"
export JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
# Stop after this many CONSECUTIVE judge failures instead of writing a
# plausible but fabricated low score. A real run once wrote a 0.00% match rate
# from 106 consecutive API failures, each recorded as match:False, with nothing
# crashing. 0 disables.
#
# KNOWN GAP: this counter is consecutive-only and cannot catch a scattered
# failure mode. The previous qwen judge run failed 31% of its calls in a
# scattered pattern and never tripped it. The real protection here is
# OLLAMA_TIMEOUT above plus the health lines summarize.py prints; check those
# before quoting any number from a run with a non-zero failure count.
export JUDGE_ABORT_AFTER="${JUDGE_ABORT_AFTER:-5}"

# -----------------------------------------------------------------------------
#  Paths
# -----------------------------------------------------------------------------
PYTHON="${PYTHON:-python}"

# The pipeline's status banners contain emoji. On Windows Python defaults stdout
# to the console codepage (cp1252), which cannot encode them, and the very first
# banner raises UnicodeEncodeError before any work starts.
export PYTHONIOENCODING=utf-8

IN="dataset/pseudocode_track/pseudocode_codes"            # 209 corrupted pseudocode
NONE_IN="dataset/pseudocode_track/pseudocode_codes_none"  # 96 correct -> NONE files
MISC="dataset/pseudocode_track/misconceptions_22.json"
PROBLEMS="dataset/pseudocode_track/problems_pseudocode.json"

RAG_SUBMISSION_CSV="${RAG_SUBMISSION_CSV:-dataset/retrival_openai_embedding_large.csv}"
RAG_CORRECT_CSV="${RAG_CORRECT_CSV:-dataset/retrival_correct_codes.csv}"
RAG_TOP_K="${RAG_TOP_K:-3}"
REF_CSV="${REF_CSV:-dataset/Submission_Code_with_reference_from_APR.csv}"
REF_COLUMN="${REF_COLUMN:-Reference_Code}"

# -----------------------------------------------------------------------------
#  Run controls
# -----------------------------------------------------------------------------
SMOKE="${SMOKE:-0}"
SMOKE_LIMIT="${SMOKE_LIMIT:-6}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE="${FORCE:-0}"

# Correct-bag policy. cover-all partitions EVERY correct program into bags, so
# each one is evaluated at least once per pass. CORRECT_PASSES>1 re-shuffles and
# re-partitions that many times, which is the only way to get more than ~4 bags
# out of 19 programs -- see README "Why 96 correct files are 19 programs".
COVER_ALL="${COVER_ALL:-1}"
CORRECT_PASSES="${CORRECT_PASSES:-1}"
CORRECT_RATIO="${CORRECT_RATIO:-0.15}"

# "full" (misconception bags + correct bags) or "correct_only" (correct bags
# only). Set by the entry point, read by run_arm.
RUN_MODE="${RUN_MODE:-correct_only}"

ARMS="${ARMS:-baseline rag ref rag_ref}"

# -----------------------------------------------------------------------------
#  Preflight -- fail fast and legibly instead of 200 confusing errors
# -----------------------------------------------------------------------------
preflight() {
  local missing=0

  for f in "${MISC}" "${PROBLEMS}"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing dataset file: ${f}"; missing=1; }
  done
  for d in "${IN}" "${NONE_IN}"; do
    [[ -d "${d}" ]] || { echo "ERROR: missing dataset dir: ${d}"; missing=1; }
  done

  if [[ "${AID_FLAGS}" == *"--rag-csv"* ]]; then
    [[ -f "${RAG_SUBMISSION_CSV}" ]] || { echo "ERROR: missing RAG CSV: ${RAG_SUBMISSION_CSV}"; missing=1; }
    [[ -f "${RAG_CORRECT_CSV}"    ]] || { echo "ERROR: missing RAG correct CSV: ${RAG_CORRECT_CSV}"; missing=1; }
  fi
  if [[ "${AID_FLAGS}" == *"--ref-csv"* ]]; then
    [[ -f "${REF_CSV}" ]] || { echo "ERROR: missing REF CSV: ${REF_CSV}"; missing=1; }
  fi

  # Is Ollama up, and does it have BOTH models? A missing judge model is worth
  # catching now rather than after several hours of mining -- even though in
  # correct-only mode it will never be called.
  local tmp; tmp="$(mktemp)"
  if ! curl -sf --max-time 5 "${OLLAMA_HOST_URL}/api/tags" -o "${tmp}"; then
    echo "ERROR: no Ollama server reachable at ${OLLAMA_HOST_URL}"
    echo "       Start one with:  ollama serve"
    missing=1
  else
    local m
    for m in "${MODEL}" "${JUDGE_MODEL}"; do
      if ! grep -q "\"${m}\"" "${tmp}"; then
        echo "ERROR: Ollama is running but '${m}' is not built/pulled."
        echo "       Build all four with:  bash scripts/build_models.sh"
        echo "       Installed models:"
        grep -oE '"name":"[^"]+"' "${tmp}" | sed 's/"name":/         /' || true
        missing=1
      fi
    done
  fi
  rm -f "${tmp}"

  [[ "${missing}" == "0" ]] || { echo; echo "Preflight failed -- nothing was run."; exit 1; }
}

# -----------------------------------------------------------------------------
#  The pipeline
# -----------------------------------------------------------------------------
run_arm() {
  local suffix=""
  [[ "${RUN_MODE}" == "correct_only" ]] && suffix="_correctbags"

  # Direction is in the tag so A and B never overwrite each other, even if
  # someone overrides MODEL to the same name in both.
  MODEL_TAG="${MODEL_TAG:-dir${DIRECTION}_${MODEL//[:\/]/-}_${ARM}${suffix}}"
  OUT="results/${MODEL_TAG}"
  EVAL_OUT="results/evaluations/${MODEL_TAG}/single_multi"
  EVAL_OUT_MULTI="results/evaluations/${MODEL_TAG}/multi"

  preflight
  mkdir -p "${OUT}/single" "${OUT}/multi" "${OUT}/single_multi" "${EVAL_OUT}" "${EVAL_OUT_MULTI}"

  local SMOKE_SINGLE="" SMOKE_MULTI=""
  if [[ "${SMOKE}" == "1" ]]; then
    echo "### SMOKE TEST: limiting to ${SMOKE_LIMIT} codes / bags ###"
    SMOKE_SINGLE="--max-files ${SMOKE_LIMIT}"
    SMOKE_MULTI="--max-requests ${SMOKE_LIMIT}"
  fi

  local CORRECT_FLAGS
  if [[ "${COVER_ALL}" == "1" ]]; then
    CORRECT_FLAGS="--correct-bags-cover-all --correct-bags-passes ${CORRECT_PASSES}"
  else
    CORRECT_FLAGS="--correct-bags-ratio ${CORRECT_RATIO}"
  fi
  # Correct-only: emit ONLY correct-only bags -- no misconception bags at all.
  [[ "${RUN_MODE}" == "correct_only" ]] && CORRECT_FLAGS="${CORRECT_FLAGS} --correct-bags-only"

  local LLM_FLAGS=(--ollama-model "${MODEL}"
                   --ollama-host "${OLLAMA_HOST_URL}"
                   --template-dir prompt_templates/mining-pseudocode
                   --problems-file "${PROBLEMS}")

  echo "================================================================"
  echo "  DIRECTION: ${DIR_LABEL}"
  echo "  ARM      : ${ARM}   (RUN_MODE=${RUN_MODE})"
  echo "  MINING   : ${MODEL}"
  echo "  JUDGE    : ${JUDGE_MODEL}$( [[ "${RUN_MODE}" == "correct_only" ]] && echo '   [INERT: 0 calls expected]' )"
  echo "  OLLAMA   : ${OLLAMA_HOST_URL}   (native /api/chat, no API key, timeout ${OLLAMA_TIMEOUT}s)"
  echo "  TEMPLATE : single=${SINGLE_TEMPLATE}  multi=${MULTI_TEMPLATE}"
  echo "  OUTPUT   : ${OUT}"
  echo "  SMOKE=${SMOKE}  RUN_EVAL=${RUN_EVAL}  FORCE=${FORCE}  PASSES=${CORRECT_PASSES}"
  echo "================================================================"

  # ===========================================================================
  #  MINING
  # ===========================================================================
  unload_model "${JUDGE_MODEL}"

  # -- [1] McMiner-M: forms the bags AND mines them ---------------------------
  if [[ -f "${OUT}/multi/multi_predictions.json" && "${FORCE}" != "1" ]]; then
    echo "== [1/5] McMiner-M SKIPPED -- bags exist at ${OUT}/multi/multi_predictions.json (FORCE=1 to rebuild) =="
  else
    if [[ "${RUN_MODE}" == "correct_only" ]]; then
      echo "== [1/5] McMiner-M, CORRECT-ONLY bags (every correct program, partitioned; ${CORRECT_PASSES} pass(es)) =="
    else
      echo "== [1/5] McMiner-M (whole-bag mining; also forms the bags) =="
    fi
    "${PYTHON}" src/run_infer_misc_multi.py \
      "${LLM_FLAGS[@]}" \
      --template "${MULTI_TEMPLATE}" ${AID_FLAGS} \
      --input-dir "${IN}" \
      ${CORRECT_FLAGS} \
      ${SMOKE_MULTI} \
      --output-dir "${OUT}/multi"
  fi

  # -- [2] McMiner-S on the corrupted codes -----------------------------------
  # Skipped in correct-only mode: there are no misconception bags to align them
  # to, so those 209 mining calls per arm would be pure waste.
  if [[ "${RUN_MODE}" == "correct_only" ]]; then
    echo "== [2/5] McMiner-S on corrupted codes SKIPPED (correct-only run has no misconception bags) =="
  else
    echo "== [2/5] McMiner-S (per-code mining) =="
    "${PYTHON}" src/run_infer_misc.py \
      "${LLM_FLAGS[@]}" \
      --template "${SINGLE_TEMPLATE}" ${AID_FLAGS} \
      --input-dir "${IN}" \
      ${SMOKE_SINGLE} \
      --output-dir "${OUT}/single"
  fi

  # -- [2b] McMiner-S on the correct codes -> NONE predictions ----------------
  # These give the per-code false-positive rate, which is the finer-grained
  # half of this experiment: a bag fails if ANY of its five codes draws a
  # spurious misconception, so the per-code rate is what the bag rate is made of.
  local APPEND_FLAG="--append-results"
  [[ "${RUN_MODE}" == "correct_only" ]] && APPEND_FLAG=""
  echo "== [2b/5] McMiner-S on correct (none_inapplicable) codes -> NONE =="
  "${PYTHON}" src/run_infer_misc.py \
    "${LLM_FLAGS[@]}" \
    --template "${SINGLE_TEMPLATE}" ${AID_FLAGS} \
    --input-dir "${NONE_IN}" \
    ${APPEND_FLAG} \
    ${SMOKE_SINGLE} \
    --output-dir "${OUT}/single"

  # -- [3] Align single predictions into the multi bags -----------------------
  echo "== [3/5] Align single predictions into the multi bags =="
  "${PYTHON}" src/create_single_multi_predictions.py \
    --multi-predictions-file "${OUT}/multi/multi_predictions.json" \
    --single-predictions-dir "${OUT}/single" \
    --output-file "${OUT}/single_multi/grouped_predictions.json" \
    --pretty-print

  if [[ "${RUN_EVAL}" != "1" ]]; then
    echo "== [4-5/5] SKIPPED (RUN_EVAL=0). Predictions:"
    echo "     ${OUT}/single/predictions.json               (McMiner-S)"
    echo "     ${OUT}/multi/multi_predictions.json          (McMiner-M)"
    echo "     ${OUT}/single_multi/grouped_predictions.json (aligned)"
    return 0
  fi

  # ===========================================================================
  #  SCORING
  # ===========================================================================
  unload_model "${MODEL}"
  if [[ "${RUN_MODE}" == "correct_only" ]]; then
    echo "-- NOTE: a correct-only run needs ZERO judge calls. Ground truth is NONE,"
    echo "         so both scorers decide by rule (correct_bag_rule / empty_check)."
    echo "         ${JUDGE_MODEL} is configured and verified, but is not asked anything."
    echo "         Directions A and B therefore differ only in the MINER."
  fi

  # -- [4] Evaluate McMiner-S -------------------------------------------------
  echo "== [4/5] Evaluate McMiner-S =="
  "${PYTHON}" src/evaluate_single_multi_predictions.py \
    --grouped-predictions-file "${OUT}/single_multi/grouped_predictions.json" \
    --input-dir "${IN}" \
    --misconceptions-file "${MISC}" \
    --output-dir "${EVAL_OUT}"

  # -- [5] Evaluate McMiner-M -------------------------------------------------
  echo "== [5/5] Evaluate McMiner-M (whole-bag) =="
  "${PYTHON}" src/compute_eval_metrics_multi.py \
    --predictions-file "${OUT}/multi/multi_predictions.json" \
    --misconceptions-file "${MISC}" \
    --input-dir "${IN}" \
    --output-dir "${EVAL_OUT_MULTI}" \
    --judge-model "${JUDGE_MODEL}"

  echo "== DONE (dir ${DIRECTION}, ${ARM}, ${RUN_MODE}) =="
  echo "   McMiner-S metrics: ${EVAL_OUT}/evaluation_metrics.json"
  echo "   McMiner-M metrics: ${EVAL_OUT_MULTI}/evaluation_metrics.json"
}
