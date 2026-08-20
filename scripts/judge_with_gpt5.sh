#!/bin/bash
# =============================================================================
# Re-judge ALREADY-MINED predictions with GPT-5 via OpenRouter.
#
# Why this exists: the local run judges with the same model that did the mining,
# which is self-evaluation and not comparable to the McMiner paper's main table
# (judged by GPT-5). Mining is the expensive part and is already done, so this
# re-runs ONLY the two evaluation steps against the existing predictions.
#
#   bash scripts/judge_with_gpt5.sh              # preflight + estimate, then ask
#   CONFIRM=1 bash scripts/judge_with_gpt5.sh    # skip the prompt (for CI)
#   DRY_RUN=1 bash scripts/judge_with_gpt5.sh    # preflight + estimate only
#
# THIS SPENDS MONEY. It prints an estimate from OpenRouter's live pricing and
# requires confirmation before the first API call.
#
# Results are written to a SEPARATE tree (results/evaluations_gpt5/) so the
# existing self-judged numbers stay intact for comparison.
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# -----------------------------------------------------------------------------
#  Config
# -----------------------------------------------------------------------------
PYTHON="${PYTHON:-python}"
ARMS="${ARMS:-baseline rag ref rag_ref}"

# Which mined run to re-judge. Must match the results/<tag>_<arm>/ dirs.
TAG_PREFIX="${TAG_PREFIX:-ollama_gpt-oss-mcminer-latest}"

# Provider: `openai` talks to api.openai.com directly (needs OPENAI_API_KEY and
# a bare model id such as `gpt-5`); `openrouter` goes through the gateway (needs
# an sk-or- key and a namespaced id such as `openai/gpt-5`).
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openrouter}"
if [[ "${JUDGE_PROVIDER}" == "openai" ]]; then
  JUDGE_MODEL="${JUDGE_MODEL:-gpt-5}"
  API_BASE="https://api.openai.com/v1"
  KEY_VAR="OPENAI_API_KEY"
else
  JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5}"
  # The bundle's _common.sh points OPENROUTER_BASE_URL at localhost for the local
  # judge. It MUST be overridden here or every judge call would hit Ollama and
  # silently score with the local model instead of GPT-5.
  export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
  API_BASE="${OPENROUTER_BASE_URL}"
  KEY_VAR="OPENROUTER_API_KEY"
fi

# GPT-5 is a reasoning model: it spends part of its token budget thinking before
# it answers. The judge's built-in default is 1500, which is fine for a
# non-reasoning judge and too tight here -- when it runs out, the <evaluation>
# block is truncated or empty and parse_judge_response turns that into a score
# via defaults that are NOT neutral. Measured on gpt-oss: at a 4000 budget,
# reasoning consumed all 4000 and returned zero content.
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4000}"

# Some reasoning endpoints reject `temperature`. Empty string omits it entirely.
export JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"

# Do NOT inherit a local reasoning-effort override; that env var is for Ollama.
unset LLM_REASONING_EFFORT

# Emoji banners vs the Windows console codepage.
export PYTHONIOENCODING=utf-8

export JUDGE_PROVIDER JUDGE_MODEL

EVAL_ROOT="${EVAL_ROOT:-results/evaluations_gpt5}"
MISC="dataset/pseudocode_track/misconceptions_22.json"
IN="dataset/pseudocode_track/pseudocode_codes"
DRY_RUN="${DRY_RUN:-0}"
CONFIRM="${CONFIRM:-0}"

echo "================================================================"
echo "  RE-JUDGE with a hosted model"
echo "  judge     : ${JUDGE_PROVIDER} / ${JUDGE_MODEL}   (key from \$${KEY_VAR})"
echo "  endpoint  : ${API_BASE}"
echo "  mined run : results/${TAG_PREFIX}_<arm>/"
echo "  arms      : ${ARMS}"
echo "  output    : ${EVAL_ROOT}/<arm>/"
echo "  budgets   : max_tokens=${JUDGE_MAX_TOKENS} temperature='${JUDGE_TEMPERATURE}'"
echo "================================================================"

# -----------------------------------------------------------------------------
#  Preflight + cost estimate (no API spend beyond one /models call)
# -----------------------------------------------------------------------------
mkdir -p results
if ! "${PYTHON}" scripts/estimate_judge_cost.py \
      --arms ${ARMS} \
      --tag-prefix "${TAG_PREFIX}" \
      --judge-model "${JUDGE_MODEL}" \
      --provider "${JUDGE_PROVIDER}" \
      --base-url "${API_BASE}" \
      --max-output-tokens "${JUDGE_MAX_TOKENS}" \
      --misconceptions-file "${MISC}" \
      --input-dir "${IN}"; then
  echo
  echo "Preflight failed -- nothing was judged and nothing was charged."
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "DRY_RUN=1 -- stopping before any judge calls."
  exit 0
fi

if [[ "${CONFIRM}" != "1" ]]; then
  echo
  read -r -p "Proceed and spend this? Type 'yes' to continue: " reply
  if [[ "${reply}" != "yes" ]]; then
    echo "Aborted -- nothing was charged."
    exit 1
  fi
fi

# -----------------------------------------------------------------------------
#  Judge each arm: steps 4 and 5 only. Mining is untouched.
# -----------------------------------------------------------------------------
declare -a OK=() BAD=()
START=$(date +%s)

for arm in ${ARMS}; do
  tag="${TAG_PREFIX}_${arm}"
  out="results/${tag}"
  eval_s="${EVAL_ROOT}/${arm}/single_multi"
  eval_m="${EVAL_ROOT}/${arm}/multi"
  mkdir -p "${eval_s}" "${eval_m}"

  echo
  echo "########################################################################"
  echo "#  ARM: ${arm}   ->  ${EVAL_ROOT}/${arm}/"
  echo "########################################################################"

  arm_ok=1

  echo "== [1/2] Judge McMiner-S =="
  if ! "${PYTHON}" src/evaluate_single_multi_predictions.py \
        --grouped-predictions-file "${out}/single_multi/grouped_predictions.json" \
        --input-dir "${IN}" \
        --misconceptions-file "${MISC}" \
        --output-dir "${eval_s}"; then
    echo "!! McMiner-S judging FAILED for ${arm}"
    arm_ok=0
  fi

  echo "== [2/2] Judge McMiner-M =="
  if ! "${PYTHON}" src/compute_eval_metrics_multi.py \
        --predictions-file "${out}/multi/multi_predictions.json" \
        --misconceptions-file "${MISC}" \
        --input-dir "${IN}" \
        --output-dir "${eval_m}" \
        --judge-provider "${JUDGE_PROVIDER}" --judge-model "${JUDGE_MODEL}"; then
    echo "!! McMiner-M judging FAILED for ${arm}"
    arm_ok=0
  fi

  if [[ "${arm_ok}" == "1" ]]; then OK+=("${arm}"); else BAD+=("${arm}"); fi
done

ELAPSED=$(( $(date +%s) - START ))

# -----------------------------------------------------------------------------
#  Compare against the local self-judged numbers
# -----------------------------------------------------------------------------
echo
echo "========================= SUMMARY ========================="
printf '  elapsed: %dh %dm %ds\n' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
[[ ${#OK[@]}  -gt 0 ]] && echo "  ok     : ${OK[*]}"
[[ ${#BAD[@]} -gt 0 ]] && echo "  failed : ${BAD[*]}"
echo "==========================================================="

"${PYTHON}" - "${EVAL_ROOT}" "${TAG_PREFIX}" ${ARMS} <<'PY'
import json, os, sys
eval_root, prefix = sys.argv[1], sys.argv[2]
arms = sys.argv[3:]
def rd(p, *keys):
    if not os.path.exists(p): return None
    d = json.load(open(p, encoding="utf-8"))
    for k in keys: d = d.get(k, {}) if isinstance(d, dict) else {}
    return d if d != {} else None

print(f"\n{'arm':10} {'M std local':>12} {'M std GPT-5':>12} {'delta':>7} "
      f"{'M novel local':>14} {'M novel GPT-5':>14} {'judge fails':>12}")
print("-"*90)
for a in arms:
    loc = rd(f"results/evaluations/{prefix}_{a}/multi/claude_evaluation_results.json", "summary")
    new = rd(f"{eval_root}/{a}/multi/claude_evaluation_results.json", "summary")
    if not new:
        print(f"{a:10} {'(no GPT-5 result)':>50}")
        continue
    l_s = loc["match_rate"] if loc else float("nan")
    l_n = loc["match_with_novel_rate"] if loc else float("nan")
    print(f"{a:10} {l_s:11.2%} {new['match_rate']:11.2%} "
          f"{(new['match_rate']-l_s)*100:+6.1f} "
          f"{l_n:13.2%} {new['match_with_novel_rate']:13.2%} "
          f"{new.get('judge_parse_failures',0):12}")
print("\nlocal = gpt-oss:20b judging its own mining (self-evaluation)")
print("GPT-5 = independent judge, comparable to the paper's main table")
print("\nCheck judge_parse_failures: anything above 0 means some replies had no")
print("parseable <evaluation> block and fell back to defaults. Raise")
print("JUDGE_MAX_TOKENS and re-run those arms if so.")
PY

[[ ${#BAD[@]} -eq 0 ]]
