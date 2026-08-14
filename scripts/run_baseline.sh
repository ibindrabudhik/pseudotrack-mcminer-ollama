#!/bin/bash
# BASELINE arm — no retrieval, no APR reference. Problem + student pseudocode only.
#   bash scripts/run_baseline.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ARM="baseline"
SINGLE_TEMPLATE="zeroshot"
MULTI_TEMPLATE="zeroshot-no-reasoning-multi"
AID_FLAGS=""

source scripts/_common.sh
run_arm
