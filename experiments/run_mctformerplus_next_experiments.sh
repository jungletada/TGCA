#!/usr/bin/env bash
set -euo pipefail

tgca_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$tgca_repo_root"

printf "QUEUE_STAGE=attention_scale_diagnostic started=%s\n" "$(date --iso-8601=seconds)"
bash experiments/diagnostics/run_mctformerplus_mechanism.sh
printf "QUEUE_STAGE=normalization_pilot started=%s\n" "$(date --iso-8601=seconds)"
bash experiments/ablations/run_mctformerplus_voc_pilot.sh
printf "QUEUE_COMPLETE finished=%s\n" "$(date --iso-8601=seconds)"
