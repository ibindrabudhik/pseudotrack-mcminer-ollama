#!/bin/bash
# RAG + REF arm — retrieval AND APR reference code together. Longest prompts of
# the four, so this is the arm most likely to hit an undersized context window.
#   bash scripts/run_rag_ref.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ARM="rag_ref"
SINGLE_TEMPLATE="zeroshot-rag-ref"
MULTI_TEMPLATE="zeroshot-no-reasoning-multi-rag-ref"

source scripts/_common.sh
AID_FLAGS="--rag-csv ${RAG_SUBMISSION_CSV} --rag-correct-csv ${RAG_CORRECT_CSV} --rag-top-k ${RAG_TOP_K} --ref-csv ${REF_CSV} --ref-column ${REF_COLUMN}"

run_arm
