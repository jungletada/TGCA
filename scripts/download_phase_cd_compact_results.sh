#!/usr/bin/env bash
# Download only the compact Phase C/D artifacts; never pulls raw NPY caches or PNG panels.
set -euo pipefail

remote_host="${1:-LHR}"
destination_root="${2:-./tgca_phase_cd_compact_results}"
remote_repo="${TGCA_REMOTE_REPO:-/home/peng/code/TGCA}"

phase_c_run="phase_c_graph_stability/20260907-phase-c-full-0d6a1b3"
phase_d_run="phase_d_relevance_stability/20260907-phase-d-full-0d6a1b3"
results_root="${remote_repo}/results/spatial_graph_stability"

phase_c_files=(
  03_GRAPH_STABILITY_DIAGNOSTIC_REPORT.md
  graph_lowpass_fidelity.csv
  graph_energy_metrics.csv
  lambda_sensitivity.csv
  stability_region_metrics.csv
  stability_boundary_metrics.csv
  stability_class_overlap.csv
  stability_negative_class_control.csv
  stability_correlations.csv
  basis_invariance.json
  completion.json
  run_metadata.json
  exact_commands.sh
  conda_explicit.txt
  pip_freeze.txt
  run.log
)

phase_d_files=(
  04_RELEVANCE_STABILITY_ABLATION_REPORT.md
  relevance_stability_metrics.csv
  graph_comparison.csv
  conditional_stability_uplift.csv
  relevance_stability_5x5_grid.csv
  quadrant_region_composition.csv
  false_positive_suppression.csv
  multi_label_overlap.csv
  absent_class_control.csv
  basis_invariance.json
  completion.json
  run_metadata.json
  exact_commands.sh
  conda_explicit.txt
  pip_freeze.txt
  run.log
)

phase_c_destination="${destination_root}/${phase_c_run}"
phase_d_destination="${destination_root}/${phase_d_run}"
mkdir -p "${phase_c_destination}/selected_visualizations" "${phase_d_destination}/selected_visualizations" "${destination_root}/test_logs"

phase_c_sources=()
for file in "${phase_c_files[@]}"; do
  phase_c_sources+=("${remote_host}:${results_root}/${phase_c_run}/${file}")
done
scp -p "${phase_c_sources[@]}" "${phase_c_destination}/"
scp -p "${remote_host}:${results_root}/${phase_c_run}/selected_visualizations/selection_manifest.json" "${phase_c_destination}/selected_visualizations/"

phase_d_sources=()
for file in "${phase_d_files[@]}"; do
  phase_d_sources+=("${remote_host}:${results_root}/${phase_d_run}/${file}")
done
scp -p "${phase_d_sources[@]}" "${phase_d_destination}/"
scp -p "${remote_host}:${results_root}/${phase_d_run}/selected_visualizations/selection_manifest.json" "${phase_d_destination}/selected_visualizations/"
scp -p "${remote_host}:${results_root}/test_logs/phase_cd_0d6a1b3.log" "${destination_root}/test_logs/"

printf 'Downloaded compact Phase C/D artifacts to: %s\n' "${destination_root}"
printf 'Excluded: raw_outputs/*.npy and selected_visualizations/*.png\n'
