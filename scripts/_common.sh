#!/bin/bash
# =============================================================================
# Shared config + pipeline body for the four McMiner pseudotrack arms.
#
# This file is SOURCED by scripts/run_<arm>.sh — do not run it directly.
# Each arm script sets ARM / SINGLE_TEMPLATE / MULTI_TEMPLATE / AID_FLAGS,
# then calls run_arm.
# =============================================================================

# -----------------------------------------------------------------------------
#  Ollama wiring
# -----------------------------------------------------------------------------
# The pipeline reaches Ollama through the `openrouter` provider. That provider is
# nothing but an OpenAI-compatible client with a settable base_url (see
# create_llm_client() in src/run_infer_misc.py and OpenAIClient in
# utils/llm_clients.py), and Ollama serves an OpenAI-compatible API on /v1 — so
# no code changes are needed to point it at localhost.
#
# Do NOT switch this to `--llm vllm`: the scripts build the model flag as
# --${PROVIDER}-model, and passing --vllm-model puts VLLMClient into *offline*
# mode, which imports the real vllm package and loads weights into local GPU
# memory. That is a different thing entirely and will fail here.
PROVIDER="openrouter"

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
MODEL="${MODEL:-qwen3.6:27b}"

# create_llm_client() raises if this is unset. Ollama ignores the value.
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-ollama}"

# No --reasoning flag: that sends OpenRouter's unified `reasoning: {effort}`
# field via extra_body, which is an OpenRouter-gateway concept, not an Ollama
# one. Set REASONING_FLAG="--reasoning" only if you have verified your Ollama
# build accepts it.
REASONING_FLAG="${REASONING_FLAG:-}"

# -----------------------------------------------------------------------------
#  Judge — the same local qwen3.6:27b, via the same Ollama server
# -----------------------------------------------------------------------------
# NOTE: this is self-evaluation. The model grades its own mined output, so these
# numbers are NOT comparable to the o3-mini / gemini / qwen3-14b results in the
# main repo (those used GPT-5 as judge, per the paper's main table). Treat this
# arm as a pipeline check, not as a paper-comparable result. To switch to the
# paper's judge, see "Swapping the judge" in README.md.
export JUDGE_PROVIDER="${JUDGE_PROVIDER:-openrouter}"
export JUDGE_MODEL="${JUDGE_MODEL:-${MODEL}}"
# compute_eval_metrics_multi.py builds its openrouter judge client from this env
# var (the mining steps instead take --openrouter-base-url on the command line).
export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-${OLLAMA_BASE_URL}}"

# -----------------------------------------------------------------------------
#  Paths — all relative to the bundle root (scripts cd there themselves)
# -----------------------------------------------------------------------------
# `python`, not `python3`: on a conda setup the deps live under `python` (the
# system python3 typically lacks tqdm/openai/dotenv). Override if yours differs.
PYTHON="${PYTHON:-python}"

IN="dataset/pseudocode_track/pseudocode_codes"            # corrupted pseudocode
NONE_IN="dataset/pseudocode_track/pseudocode_codes_none"  # correct -> NONE files
MISC="dataset/pseudocode_track/misconceptions_22.json"
PROBLEMS="dataset/pseudocode_track/problems_pseudocode.json"

# RAG / REF sources (bundled copies).
RAG_SUBMISSION_CSV="${RAG_SUBMISSION_CSV:-dataset/retrival_openai_embedding_large.csv}"
RAG_CORRECT_CSV="${RAG_CORRECT_CSV:-dataset/retrival_correct_codes.csv}"
RAG_TOP_K="${RAG_TOP_K:-3}"
REF_CSV="${REF_CSV:-dataset/Submission_Code_with_reference_from_APR.csv}"
REF_COLUMN="${REF_COLUMN:-Reference_Code}"

# -----------------------------------------------------------------------------
#  Run controls
# -----------------------------------------------------------------------------
# SMOKE=1 processes only a handful of codes — always do this first with a new
# model, before committing to the full 209-file run.
SMOKE="${SMOKE:-0}"
SMOKE_LIMIT="${SMOKE_LIMIT:-6}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE="${FORCE:-0}"

# Correct-bag policy (paper default: cover every correct code at least once).
COVER_ALL="${COVER_ALL:-1}"
CORRECT_PASSES="${CORRECT_PASSES:-1}"
CORRECT_RATIO="${CORRECT_RATIO:-0.15}"

# -----------------------------------------------------------------------------
#  Preflight — fail fast and legibly instead of 200 confusing API errors
# -----------------------------------------------------------------------------
preflight() {
  local missing=0

  for f in "${MISC}" "${PROBLEMS}"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing dataset file: ${f}"; missing=1; }
  done
  for d in "${IN}" "${NONE_IN}"; do
    [[ -d "${d}" ]] || { echo "ERROR: missing dataset dir: ${d}"; missing=1; }
  done

  # Arm-specific inputs.
  if [[ "${AID_FLAGS}" == *"--rag-csv"* ]]; then
    [[ -f "${RAG_SUBMISSION_CSV}" ]] || { echo "ERROR: missing RAG CSV: ${RAG_SUBMISSION_CSV}"; missing=1; }
    [[ -f "${RAG_CORRECT_CSV}"    ]] || { echo "ERROR: missing RAG correct CSV: ${RAG_CORRECT_CSV}"; missing=1; }
  fi
  if [[ "${AID_FLAGS}" == *"--ref-csv"* ]]; then
    [[ -f "${REF_CSV}" ]] || { echo "ERROR: missing REF CSV: ${REF_CSV}"; missing=1; }
  fi

  # Is Ollama up, and does it actually have the model?
  local tags_url="${OLLAMA_BASE_URL%/v1}/api/tags"
  if ! curl -sf --max-time 5 "${tags_url}" -o /tmp/_mcminer_tags.json; then
    echo "ERROR: no Ollama server reachable at ${OLLAMA_BASE_URL}"
    echo "       Start one with:  OLLAMA_CONTEXT_LENGTH=32768 ollama serve"
    missing=1
  elif ! grep -q "\"${MODEL}\"" /tmp/_mcminer_tags.json; then
    echo "ERROR: Ollama is running but '${MODEL}' is not pulled."
    echo "       Pull it with:  ollama pull ${MODEL}"
    echo "       Installed models:"
    grep -oE '"name":"[^"]+"' /tmp/_mcminer_tags.json | sed 's/"name":/         /' || true
    missing=1
  fi

  [[ "${missing}" == "0" ]] || { echo; echo "Preflight failed — nothing was run."; exit 1; }
}

# -----------------------------------------------------------------------------
#  The pipeline
# -----------------------------------------------------------------------------
run_arm() {
  MODEL_TAG="${MODEL_TAG:-ollama_${MODEL//[:\/]/-}_${ARM}}"
  OUT="results/${MODEL_TAG}"
  EVAL_OUT="results/evaluations/${MODEL_TAG}/single_multi"
  EVAL_OUT_MULTI="results/evaluations/${MODEL_TAG}/multi"

  preflight
  mkdir -p "${OUT}/single" "${OUT}/multi" "${OUT}/single_multi" "${EVAL_OUT}" "${EVAL_OUT_MULTI}"

  local SMOKE_SINGLE="" SMOKE_MULTI=""
  if [[ "${SMOKE}" == "1" ]]; then
    echo "### SMOKE TEST: limiting to ${SMOKE_LIMIT} codes ###"
    SMOKE_SINGLE="--max-files ${SMOKE_LIMIT}"
    SMOKE_MULTI="--max-requests ${SMOKE_LIMIT}"
  fi

  local CORRECT_FLAGS
  if [[ "${COVER_ALL}" == "1" ]]; then
    CORRECT_FLAGS="--correct-bags-cover-all --correct-bags-passes ${CORRECT_PASSES}"
  else
    CORRECT_FLAGS="--correct-bags-ratio ${CORRECT_RATIO}"
  fi

  # Every mining call shares these.
  local LLM_FLAGS=(--llm "${PROVIDER}"
                   --"${PROVIDER}"-model "${MODEL}"
                   --"${PROVIDER}"-base-url "${OLLAMA_BASE_URL}"
                   --template-dir prompt_templates/mining-pseudocode
                   --problems-file "${PROBLEMS}")

  echo "================================================================"
  echo "  ARM     : ${ARM}"
  echo "  MINING  : provider=${PROVIDER}  model=${MODEL}  endpoint=${OLLAMA_BASE_URL}"
  echo "  JUDGE   : provider=${JUDGE_PROVIDER}  model=${JUDGE_MODEL}  endpoint=${OPENROUTER_BASE_URL}"
  echo "  TEMPLATE: single=${SINGLE_TEMPLATE}  multi=${MULTI_TEMPLATE}"
  echo "  OUTPUT  : ${OUT}"
  echo "  SMOKE=${SMOKE}  RUN_EVAL=${RUN_EVAL}"
  echo "================================================================"

  # -- [1] McMiner-M: forms the bags AND mines them ---------------------------
  if [[ -f "${OUT}/multi/multi_predictions.json" && "${FORCE}" != "1" ]]; then
    echo "== [1/5] McMiner-M SKIPPED — bags exist at ${OUT}/multi/multi_predictions.json (FORCE=1 to rebuild) =="
  else
    echo "== [1/5] McMiner-M (whole-bag mining; also forms the bags) =="
    "${PYTHON}" src/run_infer_misc_multi.py \
      "${LLM_FLAGS[@]}" ${REASONING_FLAG} \
      --template "${MULTI_TEMPLATE}" ${AID_FLAGS} \
      --input-dir "${IN}" \
      ${CORRECT_FLAGS} \
      ${SMOKE_MULTI} \
      --output-dir "${OUT}/multi"
  fi

  # -- [2] McMiner-S on the corrupted codes -----------------------------------
  echo "== [2/5] McMiner-S (per-code mining) =="
  "${PYTHON}" src/run_infer_misc.py \
    "${LLM_FLAGS[@]}" ${REASONING_FLAG} \
    --template "${SINGLE_TEMPLATE}" ${AID_FLAGS} \
    --input-dir "${IN}" \
    ${SMOKE_SINGLE} \
    --output-dir "${OUT}/single"

  # -- [2b] McMiner-S on the correct codes -> NONE predictions ----------------
  # Appended to the same single/predictions.json; the aligner needs these to
  # score the correct-only bags.
  echo "== [2b/5] McMiner-S on correct (none_inapplicable) codes -> NONE =="
  "${PYTHON}" src/run_infer_misc.py \
    "${LLM_FLAGS[@]}" ${REASONING_FLAG} \
    --template "${SINGLE_TEMPLATE}" ${AID_FLAGS} \
    --input-dir "${NONE_IN}" \
    --append-results \
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
    echo "     ${OUT}/single/predictions.json            (McMiner-S)"
    echo "     ${OUT}/multi/multi_predictions.json       (McMiner-M)"
    echo "     ${OUT}/single_multi/grouped_predictions.json (aligned)"
    return 0
  fi

  # -- [4] Evaluate McMiner-S -------------------------------------------------
  echo "== [4/5] Evaluate McMiner-S (LLM-as-judge; novelty-aware) =="
  "${PYTHON}" src/evaluate_single_multi_predictions.py \
    --grouped-predictions-file "${OUT}/single_multi/grouped_predictions.json" \
    --input-dir "${IN}" \
    --misconceptions-file "${MISC}" \
    --output-dir "${EVAL_OUT}"

  # -- [5] Evaluate McMiner-M -------------------------------------------------
  # The main repo's baseline and _ref scripts stop at step 4 and never score
  # McMiner-M. All four arms here run both, so S and M are comparable.
  echo "== [5/5] Evaluate McMiner-M (whole-bag; misconception + correct bags) =="
  "${PYTHON}" src/compute_eval_metrics_multi.py \
    --predictions-file "${OUT}/multi/multi_predictions.json" \
    --misconceptions-file "${MISC}" \
    --input-dir "${IN}" \
    --output-dir "${EVAL_OUT_MULTI}" \
    --judge-provider "${JUDGE_PROVIDER}" --judge-model "${JUDGE_MODEL}"

  echo "== DONE (${ARM}) =="
  echo "   McMiner-S metrics: ${EVAL_OUT}/evaluation_metrics.json"
  echo "   McMiner-M metrics: ${EVAL_OUT_MULTI}/evaluation_metrics.json"
}
