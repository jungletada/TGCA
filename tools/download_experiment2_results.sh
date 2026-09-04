#!/usr/bin/env bash
set -Eeuo pipefail

# Download the completed Experiment 2 deliverables without checkpoints or the
# per-image signal NPZ files. Run this script on the destination machine.

readonly DEFAULT_REMOTE_ROOT='/home/peng/code/TGCA/results/lazy_assignment/experiment2_semantic_ownership/20260904-exp2-semantic-ownership-voc-val-full-0d47db4-v1'

exp2_remote="${EXP2_REMOTE:-peng@LHR}"
exp2_remote_root="${EXP2_REMOTE_ROOT:-${DEFAULT_REMOTE_ROOT}}"
exp2_destination="${EXP2_DESTINATION:-${PWD}/Experiment2_0d47db4}"
with_rendered=0
with_geometry=0
with_core_tables=0
with_all_tables=0
with_canonical=0
dry_run=0

usage() {
    cat <<'EOF'
Usage: download_experiment2_results.sh [OPTIONS]

Downloads the completed Experiment 2 results from LHR. The default download
is the lightweight report/provenance package (~11 MiB). Signal NPZ files and
checkpoints are never selected.

Options:
  --remote USER@HOST       SSH destination (default: peng@LHR)
  --remote-root PATH       Experiment 2 result root on the server
  --dest PATH              Local destination directory
  --recommended            Add core tables, rendered panels, and geometry
  --with-core-tables       Add the selected core numerical CSVs (~160 MiB)
  --with-rendered          Add 100 rendered Experiment 2 panels (~61 MiB)
  --with-geometry          Add the 20-image geometry verification (~13 MiB)
  --with-canonical         Add all nine canonical Parquet tables (~308 MiB)
  --with-all-tables        Add every analysis CSV (~1.9 GiB); supersedes core
  --dry-run                Print commands without downloading
  -h, --help               Show this help

Examples:
  ./download_experiment2_results.sh
  ./download_experiment2_results.sh --recommended
  ./download_experiment2_results.sh --with-core-tables --with-canonical
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        printf 'missing value for %s\n' "$1" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote)
            require_value "$@"
            exp2_remote="$2"
            shift 2
            ;;
        --remote-root)
            require_value "$@"
            exp2_remote_root="$2"
            shift 2
            ;;
        --dest)
            require_value "$@"
            exp2_destination="$2"
            shift 2
            ;;
        --recommended)
            with_rendered=1
            with_geometry=1
            with_core_tables=1
            shift
            ;;
        --with-core-tables)
            with_core_tables=1
            shift
            ;;
        --with-rendered)
            with_rendered=1
            shift
            ;;
        --with-geometry)
            with_geometry=1
            shift
            ;;
        --with-canonical)
            with_canonical=1
            shift
            ;;
        --with-all-tables)
            with_all_tables=1
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

run_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ ${dry_run} -eq 0 ]]; then
        "$@"
    fi
}

copy_remote_files() {
    local destination="$1"
    local relative_root="$2"
    shift 2
    local remote_sources=()
    local filename
    run_command mkdir -p "${destination}"
    for filename in "$@"; do
        remote_sources+=(
            "${exp2_remote}:${exp2_remote_root}/${relative_root}/${filename}"
        )
    done
    run_command scp -p "${remote_sources[@]}" "${destination}/"
}

copy_remote_directory() {
    local relative_path="$1"
    local destination="$2"
    run_command mkdir -p "${destination}"
    run_command scp -pr \
        "${exp2_remote}:${exp2_remote_root}/${relative_path}" \
        "${destination}/"
}

readonly -a ROOT_FILES=(
    exact_commands.sh
    pipeline_metadata.json
    pipeline_status.json
    pipeline.log
    final_report_sha256.txt
)

readonly -a SIGNAL_METADATA_FILES=(
    completion.json
    metadata.json
    command.txt
    manifest.jsonl
    run.log
    conda_explicit.txt
    pip_freeze.txt
)

readonly -a CORE_TABLES=(
    failure_pattern_summary.csv
    raw_final_cam_miou.csv
    checkpoint_classification_performance.csv
    class_token_similarity_vs_map_overlap.csv
    multiclass_map_diversity.csv
    new_shared_support_l9_l12.csv
    qk_head_region_summary.csv
    last_three_aggregation_analysis.csv
    shared_support_ownership.csv
    patch_norm_joint_control.csv
    feature_attention_cam_linkage.csv
    classification_stratified_results.csv
    probe_validity_raw_norm_qk_attn.csv
    layerwise_region_metrics.csv
    paired_model_deltas.csv
)

printf 'Remote:      %s\n' "${exp2_remote}"
printf 'Result root: %s\n' "${exp2_remote_root}"
printf 'Destination: %s\n' "${exp2_destination}"
printf 'Policy: signal NPZ files and checkpoints are excluded.\n'

# Fail early if the remote pipeline or either requested report is incomplete.
remote_preflight="test -s '${exp2_remote_root}/pipeline_status.json' && grep -q '\"status\":\"complete\"' '${exp2_remote_root}/pipeline_status.json' && grep -q '\"exit_code\":0' '${exp2_remote_root}/pipeline_status.json' && test -s '${exp2_remote_root}/reports/EXPERIMENT2_SEMANTIC_OWNERSHIP_REPORT.md' && test -s '${exp2_remote_root}/reports/NEXT_EXPERIMENT_DECISION.md'"
run_command ssh "${exp2_remote}" "${remote_preflight}"

run_command mkdir -p "${exp2_destination}"
copy_remote_files "${exp2_destination}" '.' "${ROOT_FILES[@]}"

for directory in reports plots examples audit integrity logs; do
    copy_remote_directory "${directory}" "${exp2_destination}"
done

copy_remote_files \
    "${exp2_destination}/analysis" \
    analysis \
    analysis_metadata.json analysis.log command.txt
copy_remote_files \
    "${exp2_destination}/canonical" \
    canonical \
    canonical_metadata.json

for model in mctformer mctformer_plus; do
    copy_remote_files \
        "${exp2_destination}/signals/${model}" \
        "signals/${model}" \
        "${SIGNAL_METADATA_FILES[@]}"
done

if [[ ${with_rendered} -eq 1 ]]; then
    copy_remote_directory rendered_examples "${exp2_destination}"
fi

if [[ ${with_geometry} -eq 1 ]]; then
    copy_remote_directory geometry "${exp2_destination}"
fi

if [[ ${with_canonical} -eq 1 ]]; then
    # This merges the Parquet files into the canonical directory that already
    # contains canonical_metadata.json.
    copy_remote_directory canonical "${exp2_destination}"
fi

if [[ ${with_all_tables} -eq 1 ]]; then
    copy_remote_directory analysis/tables "${exp2_destination}/analysis"
elif [[ ${with_core_tables} -eq 1 ]]; then
    copy_remote_files \
        "${exp2_destination}/analysis/tables" \
        analysis/tables \
        "${CORE_TABLES[@]}"
fi

if [[ ${dry_run} -eq 0 ]]; then
    printf '\nDownload complete.\n'
    du -sh "${exp2_destination}"
else
    printf '\nDry run complete; no local files were written.\n'
fi

