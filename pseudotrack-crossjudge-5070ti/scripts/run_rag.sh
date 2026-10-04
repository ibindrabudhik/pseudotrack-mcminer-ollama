#!/bin/bash
# RAG arm -- top-k retrieved candidate misconceptions injected into the prompt.
# Correct codes get their shortlist from RAG_CORRECT_CSV (keyed by problem_id).
#   DIRECTION=A bash scripts/run_rag.sh
set -euo pipefail
ARM="rag"
SINGLE_TEMPLATE="zeroshot-rag"
MULTI_TEMPLATE="zeroshot-no-reasoning-multi-rag"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
AID_FLAGS="--rag-csv ${RAG_SUBMISSION_CSV} --rag-correct-csv ${RAG_CORRECT_CSV} --rag-top-k ${RAG_TOP_K}"
run_arm
