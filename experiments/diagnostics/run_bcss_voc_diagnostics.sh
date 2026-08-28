#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

if [[ "${CONDA_DEFAULT_ENV:-}" != "tgca-repro" ]]; then
    echo "Activate the tgca-repro Conda environment before running this script." >&2
    exit 2
fi

run_dir=${1:?Usage: run_bcss_voc_diagnostics.sh RUN_DIR VARIANT}
variant=${2:?Usage: run_bcss_voc_diagnostics.sh RUN_DIR VARIANT}
case "$variant" in
    e0|e1|e2|e4|e4_mass|e5|e6) ;;
    *) echo "Unsupported BCSS variant: $variant" >&2; exit 2 ;;
esac

run_dir=$(realpath "$run_dir")
checkpoint="$run_dir/mctformerplus_final.pth"
voc_root="$repo_root/data/VOCdevkit/VOC2012"
val_list="$voc_root/ImageLists/val_id.txt"
diagnostic_dir="$run_dir/bcss_diagnostics"
input_size=${BCSS_INPUT_SIZE:-448}
attention_images=${BCSS_ATTENTION_IMAGES:-50}

if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 2
fi
if [[ -e "$diagnostic_dir" ]]; then
    echo "Refusing to overwrite diagnostic directory: $diagnostic_dir" >&2
    exit 2
fi
mkdir -p "$diagnostic_dir"

printf "STAGE=unified_map_dump started=%s\n" "$(date --iso-8601=seconds)"
python -m analysis.dump_attention \
    --checkpoint "$checkpoint" \
    --variant "$variant" \
    --voc-root "$voc_root" \
    --id-list "$val_list" \
    --input-size "$input_size" \
    --output-dir "$diagnostic_dir/unified_maps"

printf "STAGE=layer_head_attention_dump started=%s\n" "$(date --iso-8601=seconds)"
python -m analysis.dump_attention \
    --checkpoint "$checkpoint" \
    --variant "$variant" \
    --voc-root "$voc_root" \
    --id-list "$val_list" \
    --input-size "$input_size" \
    --max-images "$attention_images" \
    --save-layer-head \
    --output-dir "$diagnostic_dir/layer_head_attention"

for map_key in patch_cam class_to_patch final_cam; do
    python -m analysis.ownership_metrics \
        --dump-dir "$diagnostic_dir/unified_maps" \
        --voc-root "$voc_root" \
        --map-key "$map_key" \
        --output-dir "$diagnostic_dir/ownership_$map_key"
done

case "$variant" in
    e0)
        python -m analysis.background_metrics \
            --dump-dir "$diagnostic_dir/unified_maps" --voc-root "$voc_root" \
            --map-key cam_complement --threshold 0.5 \
            --output-dir "$diagnostic_dir/background_cam_complement"
        ;;
    e1)
        for direction in register_to_patch patch_to_register; do
            python -m analysis.background_metrics \
                --dump-dir "$diagnostic_dir/unified_maps" --voc-root "$voc_root" \
                --map-key "$direction" --score-transform max --threshold 0.5 \
                --output-dir "$diagnostic_dir/background_$direction"
        done
        ;;
    e2)
        python -m analysis.background_metrics \
            --dump-dir "$diagnostic_dir/unified_maps" --voc-root "$voc_root" \
            --map-key background_attention --score-transform max --threshold 0.5 \
            --output-dir "$diagnostic_dir/background_attention"
        ;;
    e4|e4_mass|e5|e6)
        python -m analysis.background_metrics \
            --dump-dir "$diagnostic_dir/unified_maps" --voc-root "$voc_root" \
            --map-key background_ownership --threshold 0.5 \
            --output-dir "$diagnostic_dir/background_ownership"
        python -m analysis.background_metrics \
            --dump-dir "$diagnostic_dir/unified_maps" --voc-root "$voc_root" \
            --map-key background_raw_score --score-transform sigmoid --threshold 0.5 \
            --output-dir "$diagnostic_dir/background_raw_score"
        ;;
esac

python -m analysis.visualize_slots \
    --dump-dir "$diagnostic_dir/unified_maps" \
    --voc-root "$voc_root" \
    --output-dir "$diagnostic_dir/visualizations" \
    --max-images "${BCSS_VISUALIZATION_IMAGES:-20}"

if [[ "${BCSS_RUN_COUNTERFACTUAL:-0}" == "1" ]]; then
    counterfactual_args=()
    if [[ -n "${BCSS_COUNTERFACTUAL_MAX_IMAGES:-}" ]]; then
        counterfactual_args+=(--max-images "$BCSS_COUNTERFACTUAL_MAX_IMAGES")
    fi
    python -m analysis.counterfactual \
        --checkpoint "$checkpoint" --variant "$variant" \
        --voc-root "$voc_root" --id-list "$val_list" \
        --input-size "$input_size" \
        --output-dir "$diagnostic_dir/counterfactual" \
        "${counterfactual_args[@]}"
fi

printf "DIAGNOSTICS_COMPLETE finished=%s\n" "$(date --iso-8601=seconds)"
