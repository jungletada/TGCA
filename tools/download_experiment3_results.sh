#!/usr/bin/env bash
set -Eeuo pipefail

# Download the completed Experiment 3 deliverables without checkpoints,
# per-image NPZ signals, CAM tensors, smoke outputs, or canonical Parquet
# tables. Run this script on the destination machine.

readonly DEFAULT_REMOTE_ROOT='/home/peng/code/TGCA/results/lazy_assignment/experiment3_three_validations/20260904-exp3-three-validations-voc-val-full-327e24c-v1'

exp3_remote="${EXP3_REMOTE:-peng@LHR}"
exp3_remote_root="${EXP3_REMOTE_ROOT:-${DEFAULT_REMOTE_ROOT}}"
exp3_destination="${EXP3_DESTINATION:-${PWD}/Experiment3_327e24c}"
with_manifests=0
with_detailed_tables=0
dry_run=0

usage() {
    cat <<'EOF'
Usage: download_experiment3_results.sh [OPTIONS]

Downloads the completed Experiment 3 reports, core numerical summaries,
plots, selected examples, logs, and reproducibility metadata from LHR.

The default package is approximately 14 MiB. Checkpoints, per-image NPZ signals,
full-resolution CAM artifacts, smoke outputs, and canonical Parquet tables
are never selected by this script.

Options:
  --remote USER@HOST       SSH destination (default: peng@LHR)
  --remote-root PATH       Experiment 3 result root on the server
  --dest PATH              New/empty local destination directory
  --with-manifests         Add per-image manifests and final source hash list
                           (approximately 11 MiB)
  --with-detailed-tables   Add the remaining detailed analysis CSV tables
                           (approximately 105 MiB)
  --dry-run                Print commands without downloading or writing files
  -h, --help               Show this help

Environment-variable equivalents:
  EXP3_REMOTE, EXP3_REMOTE_ROOT, EXP3_DESTINATION

Examples:
  ./download_experiment3_results.sh
  ./download_experiment3_results.sh --with-manifests
  ./download_experiment3_results.sh \
      --remote peng@LHR --dest "$PWD/Experiment3_results"
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
            exp3_remote="$2"
            shift 2
            ;;
        --remote-root)
            require_value "$@"
            exp3_remote_root="$2"
            shift 2
            ;;
        --dest)
            require_value "$@"
            exp3_destination="$2"
            shift 2
            ;;
        --with-manifests)
            with_manifests=1
            shift
            ;;
        --with-detailed-tables)
            with_detailed_tables=1
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
            "${exp3_remote}:${exp3_remote_root}/${relative_root}/${filename}"
        )
    done
    run_command scp -p "${remote_sources[@]}" "${destination}/"
}

copy_remote_directory() {
    local relative_path="$1"
    local destination="$2"

    run_command mkdir -p "${destination}"
    run_command scp -pr \
        "${exp3_remote}:${exp3_remote_root}/${relative_path}" \
        "${destination}/"
}

readonly -a ROOT_FILES=(
    artifact_manifest.csv
    artifact_manifest.sha256
    exact_commands.sh
    pipeline_metadata.json
    pipeline_status.json
)

readonly -a AUDIT_FILES=(
    INPUT_AUDIT.md
    checkpoint_verification.json
    experiment2_linkage.json
    source_metadata.json
)

readonly -a RUN_METADATA_FILES=(
    completion.json
    command.txt
    conda_explicit.txt
    metadata.json
    pip_freeze.txt
    run.log
)

readonly -a A_ANALYSIS_FILES=(
    analysis.log
    artifact_manifest.csv
    command.txt
    completion.json
    decision_rules.json
    metadata.json
    validation_a_decision.json
)

readonly -a A_CORE_TABLES=(
    presence_axis_pair_metrics.csv
    presence_axis_probe_linkage.csv
    presence_axis_token_metrics.csv
    shared_presence_direction.csv
)

readonly -a A_DETAILED_TABLES=(
    presence_axis_gt_region_metrics.csv
    presence_axis_map_metrics.csv
)

readonly -a B_ANALYSIS_FILES=(
    analysis_metadata.json
    canonical_metadata.json
    command.txt
)

readonly -a B_CORE_TABLES=(
    fixed_t045_metrics.csv
    native_b0_anchor.csv
    normalized_curve_auc.csv
    paired_cam_bootstrap.csv
    threshold_curves.csv
)

readonly -a B_DETAILED_TABLES=(
    class_pair_metric_summary.csv
    paired_class_pair_bootstrap.csv
    paired_region_bootstrap.csv
    paired_stage_transition_bootstrap.csv
    per_class_iou_thresholds.csv
    region_metric_summary.csv
    stage_transition_summary.csv
)

readonly -a C_ANALYSIS_FILES=(
    analysis_metadata.json
    canonical_metadata.json
    command.txt
)

readonly -a C_CORE_TABLES=(
    c2c_mass_summary.csv
    class_pair_metric_summary.csv
    classification_noninferiority.csv
    fixed_t045_metrics.csv
    head_region_summary.csv
    paired_c2c_bootstrap.csv
    paired_cam_bootstrap.csv
    paired_cam_classwise_iou_bootstrap.csv
    paired_classification_bootstrap.csv
    paired_class_pair_bootstrap.csv
    paired_head_region_bootstrap.csv
    paired_positive_recall_bootstrap.csv
    per_class_iou_thresholds.csv
    region_metric_summary.csv
    shared_support_summary.csv
    stage_transition_summary.csv
    threshold_curve_summary.csv
    threshold_curves.csv
)

readonly -a C_DETAILED_TABLES=(
    paired_region_bootstrap.csv
    paired_shared_support_bootstrap.csv
    paired_stage_transition_bootstrap.csv
)

readonly -a RUN_METADATA_ROOTS=(
    presence_axis/mctformer
    presence_axis/mctformer_plus
    cam_layer_intervention/mctformer
    cam_layer_intervention/mctformer_plus
    c2c_intervention/mctformer_plus
)

printf 'Remote:      %s\n' "${exp3_remote}"
printf 'Result root: %s\n' "${exp3_remote_root}"
printf 'Destination: %s\n' "${exp3_destination}"
printf '%s\n' \
    'Policy: checkpoints, NPZ signals, CAM tensors, smoke outputs, and' \
    '        canonical Parquet tables are excluded in every mode.'

if [[ ${dry_run} -eq 0 && -d "${exp3_destination}" ]]; then
    if [[ -n "$(find "${exp3_destination}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        printf 'destination is not empty; choose a new --dest path: %s\n' \
            "${exp3_destination}" >&2
        exit 2
    fi
fi

# Fail early unless the sealed pipeline, final source audit, and all five
# requested reports are complete. The default root contains no shell-special
# characters; custom roots should likewise be ordinary absolute paths.
remote_preflight="test -s '${exp3_remote_root}/pipeline_status.json' && grep -Eq '\"status\"[[:space:]]*:[[:space:]]*\"complete\"' '${exp3_remote_root}/pipeline_status.json' && grep -Eq '\"active_stage\"[[:space:]]*:[[:space:]]*null' '${exp3_remote_root}/pipeline_status.json' && test -s '${exp3_remote_root}/audit/final_immutability/immutability_verification.json' && grep -Eq '\"integrity_passed\"[[:space:]]*:[[:space:]]*true' '${exp3_remote_root}/audit/final_immutability/immutability_verification.json'"
for report in \
    VALIDATION_A_PRESENCE_AXIS.md \
    VALIDATION_B_CAM_LAYER_READOUT.md \
    VALIDATION_C_LATE_C2C_CAUSAL.md \
    EXPERIMENT3_COMBINED_REPORT.md \
    NEXT_METHOD_DECISION.md; do
    remote_preflight+=" && test -s '${exp3_remote_root}/reports/${report}'"
done
run_command ssh "${exp3_remote}" "${remote_preflight}"

run_command mkdir -p "${exp3_destination}"
copy_remote_files "${exp3_destination}" . "${ROOT_FILES[@]}"
copy_remote_directory reports "${exp3_destination}"
copy_remote_directory logs "${exp3_destination}"

copy_remote_files \
    "${exp3_destination}/audit" \
    audit \
    "${AUDIT_FILES[@]}"
copy_remote_files \
    "${exp3_destination}/audit/final_immutability" \
    audit/final_immutability \
    immutability_verification.json

copy_remote_files \
    "${exp3_destination}/presence_axis/analysis" \
    presence_axis/analysis \
    "${A_ANALYSIS_FILES[@]}"
copy_remote_files \
    "${exp3_destination}/presence_axis/analysis/canonical" \
    presence_axis/analysis/canonical \
    canonical_metadata.json
copy_remote_files \
    "${exp3_destination}/presence_axis/analysis/tables" \
    presence_axis/analysis/tables \
    "${A_CORE_TABLES[@]}"
copy_remote_directory \
    presence_axis/analysis/plots \
    "${exp3_destination}/presence_axis/analysis"
copy_remote_directory presence_axis/examples "${exp3_destination}/presence_axis"

copy_remote_files \
    "${exp3_destination}/cam_layer_intervention/analysis" \
    cam_layer_intervention/analysis \
    "${B_ANALYSIS_FILES[@]}" \
    "${B_CORE_TABLES[@]}"
copy_remote_directory \
    cam_layer_intervention/analysis/plots \
    "${exp3_destination}/cam_layer_intervention/analysis"
copy_remote_directory \
    cam_layer_intervention/examples \
    "${exp3_destination}/cam_layer_intervention"

copy_remote_files \
    "${exp3_destination}/c2c_intervention/analysis" \
    c2c_intervention/analysis \
    "${C_ANALYSIS_FILES[@]}" \
    "${C_CORE_TABLES[@]}"
copy_remote_directory \
    c2c_intervention/examples \
    "${exp3_destination}/c2c_intervention"

for relative_root in "${RUN_METADATA_ROOTS[@]}"; do
    copy_remote_files \
        "${exp3_destination}/${relative_root}" \
        "${relative_root}" \
        "${RUN_METADATA_FILES[@]}"
done

if [[ ${with_manifests} -eq 1 ]]; then
    copy_remote_files \
        "${exp3_destination}/audit/final_immutability" \
        audit/final_immutability \
        immutable_manifest_after.csv

    for relative_root in "${RUN_METADATA_ROOTS[@]}"; do
        copy_remote_files \
            "${exp3_destination}/${relative_root}" \
            "${relative_root}" \
            manifest.jsonl
    done

    copy_remote_files \
        "${exp3_destination}/presence_axis/mctformer" \
        presence_axis/mctformer \
        split_manifest.csv
    copy_remote_files \
        "${exp3_destination}/presence_axis/mctformer_plus" \
        presence_axis/mctformer_plus \
        split_manifest.csv
    copy_remote_files \
        "${exp3_destination}/cam_layer_intervention/analysis" \
        cam_layer_intervention/analysis \
        consumed_input_manifest.csv
    copy_remote_files \
        "${exp3_destination}/c2c_intervention/mctformer_plus" \
        c2c_intervention/mctformer_plus \
        structural_records.csv
fi

if [[ ${with_detailed_tables} -eq 1 ]]; then
    copy_remote_files \
        "${exp3_destination}/presence_axis/analysis/tables" \
        presence_axis/analysis/tables \
        "${A_DETAILED_TABLES[@]}"
    copy_remote_files \
        "${exp3_destination}/cam_layer_intervention/analysis" \
        cam_layer_intervention/analysis \
        "${B_DETAILED_TABLES[@]}"
    copy_remote_files \
        "${exp3_destination}/c2c_intervention/analysis" \
        c2c_intervention/analysis \
        "${C_DETAILED_TABLES[@]}"
fi

if [[ ${dry_run} -eq 0 ]]; then
    for report in \
        VALIDATION_A_PRESENCE_AXIS.md \
        VALIDATION_B_CAM_LAYER_READOUT.md \
        VALIDATION_C_LATE_C2C_CAUSAL.md \
        EXPERIMENT3_COMBINED_REPORT.md \
        NEXT_METHOD_DECISION.md; do
        test -s "${exp3_destination}/reports/${report}"
    done

    if command -v python3 >/dev/null 2>&1; then
        python3 - "${exp3_destination}" <<'PY'
import csv
import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(sys.argv[1])
manifest = root / "artifact_manifest.csv"
sidecar = root / "artifact_manifest.sha256"
expected_manifest_sha256 = sidecar.read_text(encoding="utf-8").split()[0]
actual_manifest_sha256 = sha256(manifest)
if actual_manifest_sha256 != expected_manifest_sha256:
    print("Top-level artifact manifest SHA-256 verification failed.", file=sys.stderr)
    raise SystemExit(1)

checked = 0
skipped = 0
mismatches = []
with manifest.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        artifact = root / row["relative_path"]
        if not artifact.is_file():
            skipped += 1
            continue
        checked += 1
        if artifact.stat().st_size != int(row["size_bytes"]):
            mismatches.append(f"size: {row['relative_path']}")
        elif sha256(artifact) != row["sha256"]:
            mismatches.append(f"sha256: {row['relative_path']}")

if mismatches:
    print("Downloaded artifact verification failed:", file=sys.stderr)
    print("\n".join(mismatches), file=sys.stderr)
    raise SystemExit(1)

print(
    f"Verified {checked} downloaded manifest artifacts; "
    f"{skipped} large/unselected manifest artifacts were intentionally skipped."
)
PY
    else
        printf '%s\n' \
            'python3 not found; skipped optional subset SHA-256 verification.'
    fi

    printf '\nDownload complete.\n'
    du -sh "${exp3_destination}"
else
    printf '\nDry run complete; no local files were written.\n'
fi
