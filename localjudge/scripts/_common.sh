# =============================================================================
#  Shared config for the local dual-judge bundle. Sourced, not run.
# =============================================================================

# Resolve the bundle root regardless of where the caller invoked us from.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
ARMS="${ARMS:-baseline rag ref rag_ref}"

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

# The judging code speaks OpenAI-compatible HTTP. Ollama serves that at /v1 and
# ignores the key, but the OpenAI SDK refuses to construct a client without one.
export OPENROUTER_BASE_URL="${OLLAMA_BASE_URL}/v1"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-ollama}"

# Emoji in the status banners vs the Windows console codepage (cp1252):
# without this the first banner raises UnicodeEncodeError before any work runs.
export PYTHONIOENCODING=utf-8

# Judge token budget. Both judges here "think" before answering, and those
# tokens come out of the SAME budget as the answer. When it runs out, the
# <evaluation> block is truncated or empty -- and the parser's defaults are not
# neutral, so a clipped reply silently becomes a score rather than an error.
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-3000}"
export JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"

# Stop after this many consecutive judge failures instead of writing a plausible
# but fabricated low score. 0 disables.
export JUDGE_ABORT_AFTER="${JUDGE_ABORT_AFTER:-5}"

DATASET="dataset/pseudocode_track"
IN="${DATASET}/pseudocode_codes"
MISC="${DATASET}/misconceptions_22.json"
PRED_ROOT="${PRED_ROOT:-predictions}"
OUT_ROOT="${OUT_ROOT:-results}"

# -----------------------------------------------------------------------------
#  Per-judge "stop thinking" switches.
#
#  There is no single flag for this. Each family exposes a different one, and
#  getting it wrong is expensive rather than obviously broken:
#
#    gpt-oss  reads a TOP-LEVEL `reasoning_effort`. Left at its default
#             ("medium") it will spend an entire 4000-token budget reasoning and
#             return EMPTY content -- measured, not hypothetical.
#
#    qwen3.6  reads `chat_template_kwargs.enable_thinking`. With thinking left
#             on, a single judge call was measured at >10 minutes versus ~160 s
#             with it off, on the same hardware.
#
#  Both are applied via env vars read in utils/llm_clients.py.
# -----------------------------------------------------------------------------
judge_env() {
  local judge="$1"
  unset LLM_REASONING_EFFORT LLM_EXTRA_BODY
  case "$judge" in
    gpt-oss-judge*)
      export LLM_REASONING_EFFORT="${GPTOSS_REASONING_EFFORT:-low}"
      ;;
    qwen36-judge*)
      export LLM_EXTRA_BODY="${QWEN_EXTRA_BODY:-{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
      ;;
    *)
      echo "  (no thinking-control profile for '${judge}' -- using model defaults)"
      ;;
  esac
}

# Free VRAM before switching judges. Ollama holds a model resident for 5 minutes
# after the last request; with two models of 14 GB and 17 GB on a 16 GB card,
# leaving the first one loaded forces the second to spill much further than it
# needs to.
unload_model() {
  local m="$1"
  if ollama ps 2>/dev/null | grep -q "$m"; then
    echo "  unloading ${m} to free VRAM for the next judge..."
    ollama stop "$m" >/dev/null 2>&1 || true
    sleep 2
  fi
}
