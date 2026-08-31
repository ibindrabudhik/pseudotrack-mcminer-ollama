#!/bin/bash
# =============================================================================
# Shared config + pipeline body for the McMiner pseudotrack, mined by
# qwen3.6:27b and judged by gpt-oss:20b -- two DIFFERENT local models.
#
# This file is SOURCED by scripts/run_<arm>.sh and by the two entry points.
# Do not run it directly.
# =============================================================================

# Resolve the bundle root regardless of where the caller invoked us from.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# -----------------------------------------------------------------------------
#  Ollama wiring
# -----------------------------------------------------------------------------
# The pipeline reaches Ollama through the `openrouter` provider. That provider is
# nothing but an OpenAI-compatible client with a settable base_url (see
# create_llm_client() in src/run_infer_misc.py and OpenAIClient in
# utils/llm_clients.py), and Ollama serves an OpenAI-compatible API on /v1 -- so
# no code changes are needed to point it at localhost.
#
# Do NOT switch this to `--llm vllm`: the scripts build the model flag as
# --${PROVIDER}-model, and passing --vllm-model puts VLLMClient into *offline*
# mode, which imports the real vllm package and loads weights into local GPU
# memory. That is a different thing entirely and will fail here.
PROVIDER="openrouter"

OLLAMA_HOST_URL="${OLLAMA_HOST_URL:-http://localhost:11434}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-${OLLAMA_HOST_URL}/v1}"

# create_llm_client() raises if this is unset. Ollama ignores the value.
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-ollama}"

# -----------------------------------------------------------------------------
#  The two models
# -----------------------------------------------------------------------------
# MINER  -- qwen3.6:27b, 16K context (Modelfile.qwen36-miner)
# JUDGE  -- gpt-oss:20b, 16K context (Modelfile.gpt-oss-judge)
#
# This is the point of the bundle. In the earlier runs the mining model also
# graded its own output, which is self-evaluation bias. Here neither model
# scores what it wrote.
MODEL="${MODEL:-qwen3.6-mcminer:latest}"
export JUDGE_PROVIDER="${JUDGE_PROVIDER:-openrouter}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss-judge:latest}"

# compute_eval_metrics_multi.py builds its openrouter judge client from this env
# var (the mining steps instead take --openrouter-base-url on the command line).
export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-${OLLAMA_BASE_URL}}"

# -----------------------------------------------------------------------------
#  Per-model "stop thinking" switches -- and why they must not leak
# -----------------------------------------------------------------------------
# Both models reason before answering, and those tokens come out of the SAME
# budget as the answer. When the budget runs out the reply is truncated -- and
# the parsers' defaults are not neutral, so a clipped reply silently becomes a
# *score* (or a "no misconception predicted") rather than an error.
#
# There is no single flag for this. Each family exposes a different one:
#
#   qwen3.6  reads `chat_template_kwargs.enable_thinking` (nested in extra_body).
#            With thinking left on, one judge-sized call was measured at >10
#            minutes versus ~160 s with it off, on the same hardware.
#
#   gpt-oss  reads a TOP-LEVEL `reasoning_effort`. Left at its default
#            ("medium") it will spend an entire 4000-token budget reasoning and
#            return EMPTY content -- measured on this dataset, not hypothetical.
#
# Both are applied via env vars read in utils/llm_clients.py. Because the miner
# and the judge are different families HERE, the two switches must be swapped
# between pipeline steps rather than exported once at the top: leaving qwen's
# extra_body set while gpt-oss judges (or vice versa) means one of the two runs
# at its default and fails in the silent way described above.
#
# Every step below is preceded by miner_env or judge_env. Do not remove them.
QWEN_NO_THINK='{"chat_template_kwargs":{"enable_thinking":false}}'

miner_env() {
  unset LLM_REASONING_EFFORT LLM_EXTRA_BODY
  export LLM_EXTRA_BODY="${QWEN_EXTRA_BODY:-${QWEN_NO_THINK}}"
}

judge_env() {
  unset LLM_REASONING_EFFORT LLM_EXTRA_BODY
  export LLM_REASONING_EFFORT="${GPTOSS_REASONING_EFFORT:-low}"
}

# Free VRAM when switching between the two models. Ollama holds a model resident
# for 5 minutes after the last request; 17 GB (miner) and 13 GB (judge) cannot
# co-reside on any consumer card, so leaving the miner loaded pushes the judge
# much further into system RAM than it needs to go -- and on a machine where RAM
# is already the binding constraint, that is the difference between a partial
# offload and a failure to allocate.
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
#  Judge guardrails
# -----------------------------------------------------------------------------
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-3000}"
export JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
# Stop after this many consecutive judge failures instead of writing a plausible
# but fabricated low score. A real run once wrote a 0.00% match rate from 106
# consecutive API failures, each recorded as match:False, with nothing crashing.
# 0 disables.
export JUDGE_ABORT_AFTER="${JUDGE_ABORT_AFTER:-5}"

# -----------------------------------------------------------------------------
#  Paths
# -----------------------------------------------------------------------------
# `python`, not `python3`: on a conda setup the deps live under `python` (the
# system python3 typically lacks tqdm/openai/dotenv). Override if yours differs.
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

# Correct-bag policy (paper default: cover every correct code at least once).
COVER_ALL="${COVER_ALL:-1}"
CORRECT_PASSES="${CORRECT_PASSES:-1}"
CORRECT_RATIO="${CORRECT_RATIO:-0.15}"

# Which run are we in: "full" (misconception bags + correct bags) or
# "correct_only" (correct bags only). Set by the entry point, read by run_arm.
RUN_MODE="${RUN_MODE:-full}"

ARMS="${ARMS:-baseline rag ref rag_ref}"

# -----------------------------------------------------------------------------
#  Preflight -- fail fast and legibly instead of 200 confusing API errors
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
  # catching now rather than after several hours of mining.
  local tags="${OLLAMA_HOST_URL}/api/tags" tmp
  tmp="$(mktemp)"
  if ! curl -sf --max-time 5 "${tags}" -o "${tmp}"; then
    echo "ERROR: no Ollama server reachable at ${OLLAMA_HOST_URL}"
    echo "       Start one with:  ollama serve"
    missing=1
  else
    local m
    for m in "${MODEL}" "${JUDGE_MODEL}"; do
      if ! grep -q "\"${m}\"" "${tmp}"; then
        echo "ERROR: Ollama is running but '${m}' is not built/pulled."
        echo "       Build both with:  bash scripts/build_models.sh"
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

  MODEL_TAG="${MODEL_TAG:-ollama_${MODEL//[:\/]/-}_${ARM}${suffix}}"
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
  # RUN 2: emit ONLY correct-only bags -- no misconception bags at all.
  [[ "${RUN_MODE}" == "correct_only" ]] && CORRECT_FLAGS="${CORRECT_FLAGS} --correct-bags-only"

  local LLM_FLAGS=(--llm "${PROVIDER}"
                   --"${PROVIDER}"-model "${MODEL}"
                   --"${PROVIDER}"-base-url "${OLLAMA_BASE_URL}"
                   --template-dir prompt_templates/mining-pseudocode
                   --problems-file "${PROBLEMS}")

  echo "================================================================"
  echo "  ARM      : ${ARM}   (RUN_MODE=${RUN_MODE})"
  echo "  MINING   : ${MODEL}        via ${OLLAMA_BASE_URL}"
  echo "  JUDGE    : ${JUDGE_MODEL}  via ${OPENROUTER_BASE_URL}"
  echo "  TEMPLATE : single=${SINGLE_TEMPLATE}  multi=${MULTI_TEMPLATE}"
  echo "  OUTPUT   : ${OUT}"
  echo "  SMOKE=${SMOKE}  RUN_EVAL=${RUN_EVAL}  FORCE=${FORCE}"
  echo "================================================================"

  # ===========================================================================
  #  MINING -- qwen3.6:27b, thinking OFF
  # ===========================================================================
  miner_env
  echo "-- miner thinking control: LLM_EXTRA_BODY='${LLM_EXTRA_BODY}'"
  unload_model "${JUDGE_MODEL}"

  # -- [1] McMiner-M: forms the bags AND mines them ---------------------------
  if [[ -f "${OUT}/multi/multi_predictions.json" && "${FORCE}" != "1" ]]; then
    echo "== [1/5] McMiner-M SKIPPED -- bags exist at ${OUT}/multi/multi_predictions.json (FORCE=1 to rebuild) =="
  else
    if [[ "${RUN_MODE}" == "correct_only" ]]; then
      echo "== [1/5] McMiner-M, CORRECT-ONLY bags (all correct codes, partitioned; ${CORRECT_PASSES} pass(es)) =="
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
  # The aligner needs these to score the correct-only bags at the single-code
  # level. In the full run they are APPENDED to the corrupted-code predictions;
  # in the correct-only run they are the whole file.
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
  #  JUDGING -- gpt-oss:20b, reasoning_effort=low
  # ===========================================================================
  unload_model "${MODEL}"
  judge_env
  echo "-- judge thinking control: LLM_REASONING_EFFORT='${LLM_REASONING_EFFORT}'"
  if [[ "${RUN_MODE}" == "correct_only" ]]; then
    echo "-- NOTE: a correct-only run needs ZERO judge calls. Ground truth is NONE,"
    echo "         so both scorers decide by rule (correct_bag_rule / empty_check)."
    echo "         gpt-oss is configured and verified, but will not be asked anything."
  fi

  # -- [4] Evaluate McMiner-S -------------------------------------------------
  echo "== [4/5] Evaluate McMiner-S (LLM-as-judge; novelty-aware) =="
  "${PYTHON}" src/evaluate_single_multi_predictions.py \
    --grouped-predictions-file "${OUT}/single_multi/grouped_predictions.json" \
    --input-dir "${IN}" \
    --misconceptions-file "${MISC}" \
    --output-dir "${EVAL_OUT}"

  # -- [5] Evaluate McMiner-M -------------------------------------------------
  echo "== [5/5] Evaluate McMiner-M (whole-bag; misconception + correct bags) =="
  "${PYTHON}" src/compute_eval_metrics_multi.py \
    --predictions-file "${OUT}/multi/multi_predictions.json" \
    --misconceptions-file "${MISC}" \
    --input-dir "${IN}" \
    --output-dir "${EVAL_OUT_MULTI}" \
    --judge-provider "${JUDGE_PROVIDER}" --judge-model "${JUDGE_MODEL}"

  echo "== DONE (${ARM}, ${RUN_MODE}) =="
  echo "   McMiner-S metrics: ${EVAL_OUT}/evaluation_metrics.json"
  echo "   McMiner-M metrics: ${EVAL_OUT_MULTI}/evaluation_metrics.json"
}
