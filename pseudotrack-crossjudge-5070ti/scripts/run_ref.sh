#!/bin/bash
# REF arm -- the APR-retrieved correct reference solution injected into the prompt.
# NOTE: correct-only codes carry misconception_id=None and therefore never join
# the REF table; they always render the NO_REFERENCE placeholder. In a
# correct-only run this arm is consequently near-identical to baseline by
# construction -- see README "What REF does in a correct-only run".
#   DIRECTION=A bash scripts/run_ref.sh
set -euo pipefail
ARM="ref"
SINGLE_TEMPLATE="zeroshot-ref"
MULTI_TEMPLATE="zeroshot-no-reasoning-multi-ref"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
AID_FLAGS="--ref-csv ${REF_CSV} --ref-column ${REF_COLUMN}"
run_arm
