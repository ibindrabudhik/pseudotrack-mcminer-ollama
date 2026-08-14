#!/bin/bash
# RAG arm — retrieved similar submissions + retrieved correct codes in the prompt.
#   bash scripts/run_rag.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ARM="rag"
SINGLE_TEMPLATE="zeroshot-rag"
MULTI_TEMPLATE="zeroshot-no-reasoning-multi-rag"

# RAG_* defaults live in _common.sh; sourced first so the flags can use them.
source scripts/_common.sh
AID_FLAGS="--rag-csv ${RAG_SUBMISSION_CSV} --rag-correct-csv ${RAG_CORRECT_CSV} --rag-top-k ${RAG_TOP_K}"

run_arm
