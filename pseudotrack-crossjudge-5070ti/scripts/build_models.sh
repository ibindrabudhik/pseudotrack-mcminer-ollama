#!/bin/bash
# Build all FOUR models (2 miners + 2 judges) from their Modelfiles, then report
# how each one actually loaded. The report is the point: a model that silently
# fell back to CPU still "works", just 20x slower, and you want to know that now
# rather than four hours in.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for spec in "gpt-oss-mcminer|Modelfile.gpt-oss-miner|gpt-oss:20b" \
            "gpt-oss-judge|Modelfile.gpt-oss-judge|gpt-oss:20b" \
            "qwen3.6-mcminer|Modelfile.qwen36-miner|qwen3.6:27b" \
            "qwen36-judge|Modelfile.qwen36-judge|qwen3.6:27b"; do
  IFS='|' read -r name file base <<< "${spec}"
  echo "=== ${name} (from ${base}) ==="
  if ! ollama list 2>/dev/null | grep -q "^${base%%:*}"; then
    echo "  base model '${base}' not pulled. Run: ollama pull ${base}"
    continue
  fi
  ollama create "${name}" -f "${file}" 2>&1 | tr '\r' '\n' | grep -viE "^\s*$" | tail -2
done

echo
echo "=== load check: each model, one tiny request (loaded then unloaded) ==="
printf '%-22s %s\n' MODEL "SIZE / PROCESSOR / CONTEXT (from ollama ps)"
for m in gpt-oss-mcminer gpt-oss-judge qwen3.6-mcminer qwen36-judge; do
  curl -s --max-time "${OLLAMA_TIMEOUT}" "${OLLAMA_HOST_URL}/api/chat" \
    -d "{\"model\":\"${m}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false,\"options\":{\"num_predict\":1}}" \
    >/dev/null 2>&1
  ollama ps 2>/dev/null | grep "^${m}" || echo "${m}: did not load"
  ollama stop "${m}" >/dev/null 2>&1 || true
done

echo
echo "Read PROCESSOR carefully (RTX 5070 Ti, 16 GB):"
echo "  gpt-oss  (13.8 GB) -> expect 100% GPU"
echo "  qwen3.6  (17 GB)   -> expect ~75%/25% GPU/CPU; fine IF system RAM is free"
echo "  a 0 GB working set -> paging to DISK. Stop and free RAM; see README Hardware."
