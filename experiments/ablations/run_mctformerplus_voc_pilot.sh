#!/usr/bin/env bash
set -euo pipefail

tgca_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$tgca_repo_root"

tgca_modes=(vanilla split_11 split_05 tgca tgca_bias)
for tgca_mode in "${tgca_modes[@]}"; do
    printf "SUITE_STAGE=%s started=%s\n" "$tgca_mode" "$(date --iso-8601=seconds)"
    bash experiments/ablations/run_mctformerplus_voc_mode.sh "$tgca_mode"
done
printf "SUITE_COMPLETE finished=%s\n" "$(date --iso-8601=seconds)"
