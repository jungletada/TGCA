#!/usr/bin/env bash
set -euo pipefail

tgca_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$tgca_repo_root"

if [[ "${CONDA_DEFAULT_ENV:-}" != "tgca-repro" ]]; then
    echo "Activate the tgca-repro Conda environment before running this script." >&2
    exit 2
fi

tgca_queue_log="$tgca_repo_root/results/queues/22427d6/mctformerplus-next.log"
tgca_post_log="$tgca_repo_root/results/queues/22427d6/mctformerplus-post-pilot.log"
mkdir -p "$(dirname "$tgca_post_log")"
exec > >(tee -a "$tgca_post_log") 2>&1

printf "POST_QUEUE_WATCH started=%s\n" "$(date --iso-8601=seconds)"
while ! grep -q '^QUEUE_COMPLETE ' "$tgca_queue_log"; do
    if ! tmux has-session -t tgca-mctplus-next 2>/dev/null; then
        echo "The primary queue ended without a QUEUE_COMPLETE marker." >&2
        exit 1
    fi
    sleep 60
done
printf "POST_QUEUE_START started=%s\n" "$(date --iso-8601=seconds)"

if [[ "$(git rev-parse HEAD)" != "22427d60bff5d1f6c4cc9c5c33f8912502d5a4b0" ]]; then
    echo "Refusing to mix pilot results across commits." >&2
    exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to analyze from a tracked dirty worktree." >&2
    exit 2
fi

printf "POST_STAGE=cuda_tests started=%s\n" "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
    tests/test_tgca_normalization.py \
    tests/test_tgca_replication.py \
    tests/test_mctformerplus_attention.py

tgca_modes=(vanilla split_11 split_05 tgca tgca_bias)
tgca_run_dirs=()
for tgca_mode in "${tgca_modes[@]}"; do
    tgca_run_dir="$tgca_repo_root/results/mctformerplus/voc/20260826-mctformerplus-voc-${tgca_mode}-s0-22427d6"
    tgca_run_dirs+=("$tgca_run_dir")
    test -f "$tgca_run_dir/metrics.json"
    test -f "$tgca_run_dir/attention_diagnostics/metrics.json"

    printf "POST_STAGE=raw_cam_metrics mode=%s started=%s\n" \
        "$tgca_mode" "$(date --iso-8601=seconds)"
    if [[ ! -f "$tgca_run_dir/raw_cam_diagnostics/metrics.json" ]]; then
        python tools/collect_cam_metrics.py \
            --cam-dir "$tgca_run_dir/cam_train" \
            --voc-root data/VOCdevkit/VOC2012 \
            --id-list data/VOCdevkit/VOC2012/ImageLists/train_id.txt \
            --output-dir "$tgca_run_dir/raw_cam_diagnostics" \
            --threshold 0.45
    fi

    printf "POST_STAGE=efficiency mode=%s started=%s\n" \
        "$tgca_mode" "$(date --iso-8601=seconds)"
    if [[ ! -f "$tgca_run_dir/efficiency/metrics.json" ]]; then
        CUDA_VISIBLE_DEVICES=0 python tools/benchmark_mctformerplus.py \
            --checkpoint "$tgca_run_dir/mctformerplus_final.pth" \
            --mode "$tgca_mode" \
            --output "$tgca_run_dir/efficiency/metrics.json"
    fi

    printf "POST_STAGE=scale_cams mode=%s started=%s\n" \
        "$tgca_mode" "$(date --iso-8601=seconds)"
    if [[ ! -f "$tgca_run_dir/scale_cams/COMPLETE" ]]; then
        CUDA_VISIBLE_DEVICES=0 python tools/generate_mctformerplus_scale_cams.py \
            --checkpoint "$tgca_run_dir/mctformerplus_final.pth" \
            --mode "$tgca_mode" \
            --voc-root data/VOCdevkit/VOC2012 \
            --id-list data/VOCdevkit/VOC2012/ImageLists/train_id.txt \
            --output-dir "$tgca_run_dir/scale_cams" \
            --resolutions 224,320,448,512
    fi

    for tgca_resolution in 224 320 448 512; do
        printf "POST_STAGE=scale_cam_quality mode=%s resolution=%s started=%s\n" \
            "$tgca_mode" "$tgca_resolution" "$(date --iso-8601=seconds)"
        if [[ ! -f "$tgca_run_dir/scale_cam_metrics/$tgca_resolution/metrics.json" ]]; then
            python tools/collect_cam_metrics.py \
                --cam-dir "$tgca_run_dir/scale_cams/$tgca_resolution" \
                --voc-root data/VOCdevkit/VOC2012 \
                --id-list data/VOCdevkit/VOC2012/ImageLists/train_id.txt \
                --output-dir "$tgca_run_dir/scale_cam_metrics/$tgca_resolution" \
                --threshold 0.45
        fi
    done

    printf "POST_STAGE=scale_consistency mode=%s started=%s\n" \
        "$tgca_mode" "$(date --iso-8601=seconds)"
    if [[ ! -f "$tgca_run_dir/scale_consistency/metrics.json" ]]; then
        python tools/evaluate_scale_consistency.py \
            --cam-root "$tgca_run_dir/scale_cams" \
            --id-list data/VOCdevkit/VOC2012/ImageLists/train_id.txt \
            --output-dir "$tgca_run_dir/scale_consistency" \
            --resolutions 224,320,448,512 \
            --reference-resolution 448 \
            --threshold 0.45
    fi
done

tgca_comparison_dir="$tgca_repo_root/results/mctformerplus/voc/comparisons/pilot-s0-22427d6"
printf "POST_STAGE=collect_comparison started=%s\n" "$(date --iso-8601=seconds)"
python tools/collect_mctformerplus_pilot.py \
    --run-dir "${tgca_run_dirs[0]}" \
    --run-dir "${tgca_run_dirs[1]}" \
    --run-dir "${tgca_run_dirs[2]}" \
    --run-dir "${tgca_run_dirs[3]}" \
    --run-dir "${tgca_run_dirs[4]}" \
    --output-dir "$tgca_comparison_dir" \
    --require-all

printf "POST_PILOT_COMPLETE finished=%s comparison=%s\n" \
    "$(date --iso-8601=seconds)" "$tgca_comparison_dir"
