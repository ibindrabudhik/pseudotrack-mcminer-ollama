#!/bin/bash
# Build both judge models from their Modelfiles, then report how each one
# actually loaded. The report is the point: a model that silently fell back to
# CPU still "works", just 20x slower, and you want to know that now rather than
# four hours in.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for spec in "gpt-oss-judge:Modelfile.gpt-oss-judge:gpt-oss:20b" \
            "qwen36-judge:Modelfile.qwen36-judge:qwen3.6:27b"; do
  name="${spec%%:*}"; rest="${spec#*:}"; file="${rest%%:*}"; base="${rest#*:}"
  echo "=== ${name} (from ${base}) ==="
  if ! ollama list 2>/dev/null | grep -q "^${base%%:*}"; then
    echo "  base model '${base}' not pulled. Run: ollama pull ${base}"
    continue
  fi
  ollama create "${name}" -f "${file}" 2>&1 | tr '\r' '\n' | grep -viE "^\s*$" | tail -2
done

echo
echo "=== load check: each judge, one tiny request ==="
printf '%-16s %-8s %-18s %s\n' MODEL SIZE PROCESSOR CONTEXT
for m in gpt-oss-judge qwen36-judge; do
  curl -s --max-time 900 "${OLLAMA_BASE_URL}/api/chat" \
    -d "{\"model\":\"${m}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false,\"options\":{\"num_predict\":1}}" \
    >/dev/null 2>&1
  ollama ps 2>/dev/null | grep "^${m}" || echo "${m}: did not load"
  ollama stop "${m}" >/dev/null 2>&1 || true
done
echo
echo "Read PROCESSOR carefully:"
echo "  100% GPU        -> ideal"
echo "  ~75%/25% GPU/CPU-> expected for qwen3.6:27b on 16 GB; fine with 32 GB RAM"
echo "  mostly CPU      -> something else is holding VRAM, or num_ctx is too high"
