#!/bin/bash
# REF arm — APR-repaired reference code in the prompt (no retrieval).
#   bash scripts/run_ref.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ARM="ref"
SINGLE_TEMPLATE="zeroshot-ref"
MULTI_TEMPLATE="zeroshot-no-reasoning-multi-ref"

source scripts/_common.sh
AID_FLAGS="--ref-csv ${REF_CSV} --ref-column ${REF_COLUMN}"

run_arm
