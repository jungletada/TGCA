from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.lazy_assignment.experiment2.generate_experiment2_report import (
    TABLE_FILES,
    _aggregation_coverage_table,
    _class_pair_classification_control_tables,
    _endpoint_association_table,
    _last_three_table,
    _paired_focus_table,
    _patch_norm_joint_table,
    _transition_classwise_delta_table,
    decide_case,
    generate_reports,
)
from analysis.lazy_assignment.experiment2.plot_experiment2 import (
    PLOT_FILES,
    TABLE_FILES as PLOT_TABLE_FILES,
    generate_plots,
)
from analysis.lazy_assignment.experiment2.select_experiment2_examples import (
    NEW_CATEGORIES,
    select_examples,
)


def _aggregate_row(
    metric: str,
    estimate: float,
    *,
    model: str = "mctformer_plus",
    signal: str | None = None,
    stage: str | None = None,
    layer: int | None = None,
    layer_or_stage: int | str | None = None,
    transition: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "model": model,
        "label_stratum": "all",
        "aggregation": "micro",
        "rho": 0.5,
        "topk_ratio": 0.1,
        "metric": metric,
        "estimate": estimate,
        "ci_low": estimate - 0.03,
        "ci_high": estimate + 0.03,
        "num_images": 10,
        "num_rows": 14,
        "bootstrap_repeats": 100,
        "bootstrap_seed": 7,
    }
    optional = {
        "signal": signal,
        "stage": stage,
        "layer": layer,
        "layer_or_stage": layer_or_stage,
        "transition": transition,
    }
    row.update({key: value for key, value in optional.items() if value is not None})
    return row


def test_last_three_table_uses_propagated_native_endpoint() -> None:
    rows = []
    for stage, estimate in (("c2p_cam", 0.11), ("final_cam", 0.77)):
        rows.append(
            _aggregate_row(
                "conditional_bg_mass",
                estimate,
                stage=stage,
            )
        )
    rendered = _last_three_table(pd.DataFrame(rows))
    native_row = next(
        line for line in rendered.splitlines() if "CAM with native last3" in line
    )
    assert "0.770" in native_row
    assert "0.110" not in native_row


def test_paired_focus_retains_rows_where_topk_is_not_applicable() -> None:
    row = _aggregate_row(
        "auc_target_bg",
        0.23,
        signal="feature_post",
        layer=12,
    )
    row.update(
        {
            "source_table": "layer_signal",
            "delta": "mctformer_plus_minus_mctformer",
            "topk_ratio": float("nan"),
        }
    )
    rendered = _paired_focus_table(pd.DataFrame([row]))
    feature_row = next(
        line
        for line in rendered.splitlines()
        if "L12 feature target-vs-BG AUROC" in line
    )
    assert "0.230" in feature_row


def test_patch_norm_report_uses_overall_not_first_classwise_row() -> None:
    overall = _aggregate_row("post_cosine_patch_l2norm_pearson_bg", 0.12, layer=12)
    overall.update(
        {
            "model": "mctformer_plus",
            "model_or_delta": "mctformer_plus",
            "aggregation_scope": "overall",
        }
    )
    classwise = dict(overall)
    classwise.update(
        {
            "estimate": 0.99,
            "ci_low": 0.98,
            "ci_high": 1.0,
            "aggregation_scope": "within_class",
            "class_id": 0,
        }
    )

    rendered = _patch_norm_joint_table(pd.DataFrame([classwise, overall]))
    assert "0.120" in rendered
    assert "0.990" not in rendered


def test_decision_discloses_coexisting_cases_and_uses_locked_precedence() -> None:
    layer = pd.DataFrame(
        [
            _aggregate_row("auc_target_bg", value, signal=signal, layer=12)
            for signal, value in (
                ("feature_post", 0.30),
                ("feature_norm", 0.75),
                ("qk_mean", 0.80),
                ("attn_c2p_conditional", 0.80),
            )
        ]
        + [
            _aggregate_row("bg_tail_enrich_10", 0.80, signal=signal, layer=12)
            for signal in ("feature_post", "attn_c2p_conditional")
        ]
    )
    cam = pd.DataFrame(
        [
            _aggregate_row("auc_target_bg", 0.80, stage="final_cam"),
            _aggregate_row("bg_tail_enrich_10", 0.80, stage="final_cam"),
        ]
    )
    shared = pd.DataFrame(
        [
            _aggregate_row(metric, value, signal="feature_post", layer_or_stage=12)
            for metric, value in (
                ("shared_target_a_fraction", 0.05),
                ("shared_target_b_fraction", 0.05),
                ("shared_other_fg_fraction", 0.05),
                ("shared_background_fraction", 0.70),
                ("shared_pair_target_fraction", 0.10),
                ("shared_dominant_target_fraction", 0.05),
            )
        ]
    )
    decision = decide_case(
        {
            "layerwise_region_metrics.csv": layer,
            "cam_stage_region_metrics.csv": cam,
            "shared_support_ownership.csv": shared,
        }
    )
    assert decision["case"] == "G"
    assert decision["satisfied_cases"] == ["G", "F"]
    assert decision["precedence"][:2] == ["G", "F"]


def test_pair_classification_tables_render_focal_joint_and_structural_na() -> None:
    focal_row = _aggregate_row(
        "shared_own_target_fraction",
        0.31,
        signal="feature_post",
        layer_or_stage=12,
    )
    focal_row.update(
        {
            "source_table": "shared_support",
            "classification_subset": "either_negative",
        }
    )
    joint_row = _aggregate_row("topk_jaccard", 0.42, signal="feature_post", layer=12)
    joint_row.update(
        {
            "source_table": "multiclass_map_diversity",
            "classification_subset": "either_class_negative",
        }
    )
    rendered = _class_pair_classification_control_tables(
        pd.DataFrame([focal_row]), pd.DataFrame([joint_row])
    )
    assert "single-label stratum" in rendered
    assert "either_negative" in rendered
    assert "0.310" in rendered
    assert "either_class_negative" in rendered
    assert "0.420" in rendered


def test_endpoint_association_table_renders_macro_and_class_denominator() -> None:
    row = _aggregate_row(
        "pearson_class_token_cosine_vs_feature_post_top10_jaccard",
        0.27,
        layer=12,
    )
    row.update(
        {
            "aggregation": "macro_class_pearson",
            "num_classes": 17,
            "num_classes_total": 20,
        }
    )
    rendered = _endpoint_association_table(pd.DataFrame([row]))
    assert "0.270" in rendered
    assert "17/20" in rendered


def test_transition_classwise_table_renders_paired_delta_extrema() -> None:
    rows = []
    for class_id, estimate in ((0, -0.15), (2, 0.25)):
        row = _aggregate_row(
            "survive_background",
            estimate,
            transition="feature_post_to_attn",
            layer=12,
        )
        row.update(
            {
                "class_id": class_id,
                "source_table": "stage_transition",
                "model_or_delta": "mctformer_plus_minus_mctformer",
                "delta": "mctformer_plus_minus_mctformer",
            }
        )
        rows.append(row)
    rendered = _transition_classwise_delta_table(pd.DataFrame(rows))
    assert "aeroplane" in rendered
    assert "bird" in rendered
    assert "-0.150" in rendered
    assert "0.250" in rendered


def test_aggregation_coverage_table_distinguishes_structural_na_and_pairing() -> None:
    classification = pd.DataFrame(
        {
            "model_or_delta": ["mctformer", "mctformer_plus"],
            "class_id": [0, 0],
            "label_stratum": ["all", "all"],
        }
    )
    paired_control = pd.DataFrame(
        {
            "model_or_delta": [
                "mctformer",
                "mctformer_plus",
                "mctformer_plus_minus_mctformer",
            ],
            "class_id": [0, 0, 0],
            "label_stratum": ["all", "all", "all"],
        }
    )

    rendered = _aggregation_coverage_table(
        classification, paired_control, paired_control
    )

    classification_line = next(
        line
        for line in rendered.splitlines()
        if "Classification-conditioned layer/CAM" in line
    )
    assert "1 / 1" in classification_line
    assert "N/A: model-specific status" in classification_line
    visible_line = next(
        line for line in rendered.splitlines() if "Target-visible layer/CAM" in line
    )
    assert "| 1 | 1 | all | exact common image_id/class_id |" in visible_line


def _write_analysis_fixture(root: Path) -> None:
    tables = root / "tables"
    audit = root / "audit"
    examples = root / "examples"
    tables.mkdir(parents=True)
    audit.mkdir()
    examples.mkdir()

    layer_rows = []
    for model, shift in (("mctformer", 0.0), ("mctformer_plus", 0.05)):
        for layer in (9, 10, 11, 12):
            for signal in (
                "feature_post",
                "feature_norm",
                "qk_mean",
                "attn_c2p_conditional",
            ):
                values = {
                    "target_hit": 0.55 + shift,
                    "bg_tail_enrich_10": 0.9 + shift,
                    "auc_target_bg": 0.70 + shift,
                    "conditional_bg_mass": 0.30 + shift,
                }
                for metric, estimate in values.items():
                    layer_rows.append(
                        _aggregate_row(
                            metric,
                            estimate,
                            model=model,
                            signal=signal,
                            layer=layer,
                        )
                    )
    layer = pd.DataFrame(layer_rows)

    cam_rows = []
    for stage, bg_mass in (("patch_cam", 0.34), ("c2p_cam", 0.29), ("final_cam", 0.31)):
        for metric, estimate in {
            "conditional_bg_mass": bg_mass,
            "target_hit": 0.62,
            "bg_tail_enrich_10": 0.92,
            "auc_target_bg": 0.78,
        }.items():
            cam_rows.append(_aggregate_row(metric, estimate, stage=stage))
    cam = pd.DataFrame(cam_rows)

    transition_rows = []
    for transition in ("feature_post_to_attn", "c2p_cam_to_final_cam"):
        for metric, estimate in {
            "survive_background": 0.30,
            "introduced_background_fraction": 0.12,
            "survive_target": 0.72,
            "removed_background_fraction": 0.24,
        }.items():
            transition_rows.append(
                _aggregate_row(metric, estimate, transition=transition, layer=12)
            )
    transitions = pd.DataFrame(transition_rows)

    shared_rows = []
    for layer_number in (9, 10, 11, 12):
        metrics = {
            "shared_target_a_fraction": 0.10,
            "shared_target_b_fraction": 0.10,
            "shared_other_fg_fraction": 0.10,
            "shared_background_fraction": 0.60,
            "new_shared_target_a_fraction": 0.10,
            "new_shared_target_b_fraction": 0.10,
            "new_shared_other_fg_fraction": 0.10,
            "new_shared_background_fraction": 0.60,
        }
        for metric, estimate in metrics.items():
            shared_rows.append(
                _aggregate_row(
                    metric,
                    estimate,
                    signal="feature_post",
                    layer_or_stage=layer_number,
                )
            )
    shared = pd.DataFrame(shared_rows)
    new_shared = shared[
        shared["metric"].str.startswith("new_shared_")
        & shared["layer_or_stage"].isin((10, 11, 12))
    ].copy()
    last3 = layer[
        (layer["signal"] == "attn_c2p_conditional")
        & (layer["layer"].isin((10, 11, 12)))
    ].copy()
    classwise = layer[
        (layer["layer"] == 12) & (layer["signal"] == "feature_post")
    ].copy()
    classwise["class_id"] = 2
    paired = layer[(layer["layer"] == 12) & (layer["signal"] == "feature_post")].copy()
    paired["delta"] = "mctformer_plus_minus_mctformer"
    probe = layer.copy()
    token = pd.DataFrame(
        [
            _aggregate_row(metric, value, layer=layer_number)
            for layer_number in (9, 10, 11, 12)
            for metric, value in {
                "class_token_cosine": 0.15,
                "feature_post_top10_jaccard": 0.44,
                "attn_c2p_top10_jaccard": 0.38,
            }.items()
        ]
    )
    classification = layer.head(8).copy()
    classification["classification_subset"] = "both_positive"
    target_visible = pd.concat(
        [layer.assign(table="layer_signal"), cam.assign(table="cam_stage")],
        ignore_index=True,
        sort=False,
    )

    def classwise_control_fixture(source_table: str, *, paired: bool) -> pd.DataFrame:
        rows = []
        for model in ("mctformer", "mctformer_plus"):
            rows.append(
                {
                    **_aggregate_row(
                        "target_hit", 0.55, model=model, signal="feature_post", layer=12
                    ),
                    "class_id": 2,
                    "source_table": source_table,
                    "model_or_delta": model,
                    "aggregation_scope": "within_class",
                    "comparison_policy": (
                        "exact_common_key_paired"
                        if paired
                        else "not_applicable_model_specific_conditioning"
                    ),
                }
            )
        if paired:
            rows.append(
                {
                    **_aggregate_row(
                        "target_hit", 0.05, signal="feature_post", layer=12
                    ),
                    "class_id": 2,
                    "source_table": source_table,
                    "model_or_delta": "mctformer_plus_minus_mctformer",
                    "aggregation_scope": "within_class",
                    "comparison_policy": "exact_common_key_paired",
                    "paired_key_columns": "image_id,class_id",
                }
            )
        return pd.DataFrame(rows)

    classification_classwise = classwise_control_fixture(
        "classification_conditioned", paired=False
    )
    target_visible_classwise = classwise_control_fixture(
        "target_visible_layer_signal", paired=True
    )
    qk_head_classwise = classwise_control_fixture("qk_head_control", paired=True)
    pair_focal_classification = pd.DataFrame(
        [
            {
                **_aggregate_row(
                    "shared_own_target_fraction",
                    0.31,
                    signal="feature_post",
                    layer_or_stage=12,
                ),
                "source_table": "shared_support",
                "classification_subset": "either_negative",
                "classification_scope": "focal_endpoint",
            }
        ]
    )
    pair_joint_classification = pd.DataFrame(
        [
            {
                **_aggregate_row("topk_jaccard", 0.42, signal="feature_post", layer=12),
                "source_table": "multiclass_map_diversity",
                "classification_subset": "either_class_negative",
                "classification_scope": "unordered_pair",
            }
        ]
    )
    endpoint_association = pd.DataFrame(
        [
            {
                **_aggregate_row(
                    "pearson_class_token_cosine_vs_feature_post_top10_jaccard",
                    0.27,
                    layer=12,
                ),
                "aggregation": "macro_class_pearson",
                "num_classes": 17,
                "num_classes_total": 20,
            }
        ]
    )
    transition_classwise = transitions.copy()
    transition_classwise["class_id"] = 0
    transition_classwise["source_table"] = "stage_transition"
    transition_classwise["model_or_delta"] = "mctformer_plus_minus_mctformer"
    transition_classwise["delta"] = "mctformer_plus_minus_mctformer"
    products = {
        "layerwise_region_metrics.csv": layer,
        "cam_stage_region_metrics.csv": cam,
        "target_visible_region_metrics.csv": target_visible,
        "target_visible_classwise_results.csv": target_visible_classwise,
        "stage_transition_metrics.csv": transitions,
        "stage_transition_classwise_results.csv": transition_classwise,
        "shared_support_ownership.csv": shared,
        "shared_support_class_marginals.csv": pd.DataFrame({"status": ["no_rows"]}),
        "new_shared_support_l9_l12.csv": new_shared,
        "last_three_aggregation_analysis.csv": last3,
        "classwise_results.csv": classwise,
        "paired_model_deltas.csv": paired,
        "probe_validity_raw_norm_qk_attn.csv": probe,
        "patch_norm_joint_control.csv": pd.DataFrame(
            [
                {
                    **_aggregate_row(
                        "post_cosine_patch_l2norm_pearson_bg", 0.12, layer=12
                    ),
                    "model": "mctformer_plus",
                    "model_or_delta": "mctformer_plus",
                    "aggregation_scope": "overall",
                }
            ]
        ),
        "class_token_similarity_vs_map_overlap.csv": token,
        "classification_stratified_results.csv": classification,
        "classification_conditioned_classwise_results.csv": classification_classwise,
        "class_pair_focal_classification_stratified_results.csv": pair_focal_classification,
        "class_pair_joint_classification_stratified_results.csv": pair_joint_classification,
        "multiclass_map_diversity.csv": token.copy(),
        "qk_head_region_summary.csv": layer.head(8).copy(),
        "qk_head_classwise_results.csv": qk_head_classwise,
        "per_image_class_failure_patterns.csv": pd.DataFrame(
            [{"model": "mctformer_plus", "image_id": "a", "class_id": 0}]
        ),
        "failure_pattern_summary.csv": pd.DataFrame(
            [_aggregate_row("type_e_full_pipeline", 0.2)]
        ),
        "feature_attention_cam_linkage.csv": transitions.copy(),
        "priority_layer_results.csv": layer.copy(),
        "class_token_map_overlap_association.csv": token.copy(),
        "class_token_map_overlap_endpoint_association.csv": endpoint_association,
        "class_pair_macro_class_results.csv": pd.DataFrame({"status": ["no_rows"]}),
        "class_pair_classwise_results.csv": pd.DataFrame({"status": ["no_rows"]}),
        "checkpoint_classification_performance.csv": pd.DataFrame(
            {"status": ["no_rows"]}
        ),
        "raw_final_cam_miou.csv": pd.DataFrame({"status": ["no_rows"]}),
    }
    assert set(products) == set(TABLE_FILES)
    for filename, frame in products.items():
        frame.to_csv(tables / filename, index=False)

    source_metadata = {
        "integrity_passed": True,
        "sources": {
            "mctformer": {
                "result_root": "/immutable/mctformer",
                "model_cli_name": "mctformerv2",
                "checkpoint": {
                    "path": "/immutable/base_final.pth",
                    "sha256": "basehash",
                },
            },
            "mctformer_plus": {
                "result_root": "/immutable/mctformer_plus",
                "model_cli_name": "mctformerplus",
                "checkpoint": {
                    "path": "/immutable/plus_final.pth",
                    "sha256": "plushash",
                },
            },
        },
        "paired_analysis_root": "/immutable/paired",
        "dataset": {
            "voc_root": "/immutable/VOC2012",
            "list_path": "/immutable/val_id.txt",
            "labels_path": "/immutable/cls_labels.npy",
            "input_size": 448,
            "patch_size": 16,
            "num_images": 1449,
        },
        "before_manifest": {"row_count": 1},
        "gt_summary": {
            "raw_mask_image_label_mismatch_count": 1,
            "positive_pairs_with_no_target_pixels_after_crop": 77,
            "positive_pairs_without_target_dominant_patch_rho05": 116,
        },
    }
    (audit / "source_metadata.json").write_text(
        json.dumps(source_metadata), encoding="utf-8"
    )
    before_manifest = audit / "file_manifest_before.csv"
    pd.DataFrame(
        [{"absolute_path": "/immutable/source", "size_bytes": 1, "sha256": "a"}]
    ).to_csv(before_manifest, index=False)

    integrity = root.parent / "integrity" / "final"
    integrity.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"absolute_path": "/immutable/source"}]).to_csv(
        integrity / "file_manifest_after.csv", index=False
    )
    (integrity / "immutability_verification.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "integrity_passed": True,
                "before_manifest": str(before_manifest.resolve()),
                "files_checked": 1,
                "missing_files": 0,
                "size_changed_files": 0,
                "sha256_changed_files": 0,
            }
        ),
        encoding="utf-8",
    )

    canonical = root / "canonical"
    canonical.mkdir()
    signal_roots = {}
    source_metadata_path = (audit / "source_metadata.json").resolve()
    runtime_relatives = (
        "analysis/lazy_assignment/experiment2/run_experiment2_signals.py",
        "analysis/lazy_assignment/experiment2/evaluation_metrics.py",
    )
    repository_root = Path(__file__).resolve().parents[1]
    runtime_hashes = {
        relative: hashlib.sha256((repository_root / relative).read_bytes()).hexdigest()
        for relative in runtime_relatives
    }
    source_trees = {}
    for model in ("mctformer", "mctformer_plus"):
        signal_root = root / "signals" / model
        signal_root.mkdir(parents=True)
        source = source_metadata["sources"][model]
        common_checks = {
            "experiment1_feature_post_max_abs_diff": 0.0,
            "native_cam_max_abs_diff": 0.0,
            "qk_attention_max_abs_diff": 0.0,
            "attention_row_sum_max_abs_error": 0.0,
            "conditional_attention_row_sum_max_abs_error": 0.0,
        }
        (signal_root / "metadata.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "run_kind": "full",
                    "model": model,
                    "processed_images": 1449,
                    "source_metadata": str(source_metadata_path),
                    "source_metadata_sha256": hashlib.sha256(
                        source_metadata_path.read_bytes()
                    ).hexdigest(),
                    "experiment1_result_root": source["result_root"],
                    "checkpoint": source["checkpoint"],
                    "dataset": {
                        "voc_root": source_metadata["dataset"]["voc_root"],
                        "list_path": source_metadata["dataset"]["list_path"],
                        "input_size": 448,
                        "patch_size": 16,
                        "expected_images": 1449,
                        "transform": "bicubic short-side Resize(512) -> CenterCrop(448) -> ToTensor -> ImageNet Normalize; matched nearest-neighbor semantic-mask geometry",
                    },
                    "git": {
                        "commit": "fixture-commit",
                        "tracked_dirty": False,
                        "runtime_source_tracked": {
                            relative: True for relative in runtime_relatives
                        },
                        "runtime_source_sha256": runtime_hashes,
                    },
                    **common_checks,
                }
            ),
            encoding="utf-8",
        )
        (signal_root / "completion.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "run_kind": "full",
                    "num_images": 1449,
                    **common_checks,
                }
            ),
            encoding="utf-8",
        )
        signal_roots[model] = str(signal_root)
        source_trees[model] = {
            "num_files": 4,
            "tree_sha256_before": f"{model}-tree",
            "tree_sha256_after": f"{model}-tree",
        }
    (canonical / "canonical_metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "source_immutability_verified": True,
                "source_manifests_exact_match": True,
                "num_manifest_images_per_model": 1449,
                "source_roots": signal_roots,
                "source_tree_before_after": source_trees,
            }
        ),
        encoding="utf-8",
    )
    output_files = {
        filename: {
            "sha256": hashlib.sha256((tables / filename).read_bytes()).hexdigest()
        }
        for filename in TABLE_FILES
    }
    analysis_log = root / "analysis.log"
    exact_commands = root.parent / "exact_commands.sh"
    pipeline_metadata = root.parent / "pipeline_metadata.json"
    analysis_log.write_text("fixture analysis log\n", encoding="utf-8")
    exact_commands.write_text("#!/bin/sh\n# fixture command ledger\n", encoding="utf-8")
    pipeline_metadata.write_text(
        json.dumps({"status": "fixture", "run_id": root.name}), encoding="utf-8"
    )
    canonical_metadata_path = canonical / "canonical_metadata.json"
    (root / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "command": "python analyze_experiment2.py --fixture",
                "canonical_dir": str(canonical),
                "canonical_verification": {
                    "metadata_path": str(canonical_metadata_path.resolve()),
                    "metadata_sha256": hashlib.sha256(
                        canonical_metadata_path.read_bytes()
                    ).hexdigest(),
                },
                "output_files": output_files,
                "provenance_files": {
                    "analysis_log": {
                        "path": str(analysis_log.resolve()),
                        "sha256": hashlib.sha256(analysis_log.read_bytes()).hexdigest(),
                    },
                    "exact_commands": {
                        "path": str(exact_commands.resolve()),
                        "sha256": hashlib.sha256(
                            exact_commands.read_bytes()
                        ).hexdigest(),
                    },
                    "pipeline_metadata": {
                        "path": str(pipeline_metadata.resolve()),
                        "sha256": hashlib.sha256(
                            pipeline_metadata.read_bytes()
                        ).hexdigest(),
                    },
                },
                "bootstrap": {
                    "unit": "image_id cluster",
                    "repeats": 5000,
                    "seed": 20260901,
                    "ci": "95% percentile",
                },
            }
        ),
        encoding="utf-8",
    )
    plots = root / "plots"
    plots.mkdir()
    plot_hashes = {}
    plot_paths = []
    for filename in PLOT_FILES:
        plot_path = plots / filename
        plot_path.write_bytes((filename.encode("utf-8") + b"\n") * 8)
        plot_paths.append(str(plot_path.resolve()))
        plot_hashes[filename] = hashlib.sha256(plot_path.read_bytes()).hexdigest()
    (plots / "plot_metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "tables_dir": str(tables.resolve()),
                "input_table_sha256": {
                    filename: hashlib.sha256(
                        (tables / filename).read_bytes()
                    ).hexdigest()
                    for filename in PLOT_TABLE_FILES
                },
                "output_dir": str(plots.resolve()),
                "plots": plot_paths,
                "plot_sha256": plot_hashes,
                "missing_or_empty_tables": {},
                "invented_values": False,
            }
        ),
        encoding="utf-8",
    )

    selection_rows = [
        {
            "case_id": f"new-{index:02d}",
            "category": category,
            "class_id": 0,
            "companion_class_id": 1,
            "class_a": "",
            "class_b": "",
            "positive_class_ids_json": "[0, 1]",
            "num_positive_classes": 2,
        }
        for index, category in enumerate(NEW_CATEGORIES)
    ]
    selection_rows.extend(
        {
            "case_id": f"fixed-{index:02d}",
            "category": "experiment1_fixed::fixture",
            "class_id": "",
            "companion_class_id": "",
            "class_a": "",
            "class_b": "",
            "positive_class_ids_json": "",
            "num_positive_classes": "",
        }
        for index in range(70)
    )
    selection_path = examples / "example_selection.csv"
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
    (examples / "selection_metadata.json").write_text(
        json.dumps(
            {
                "canonical_dir": str(canonical.resolve()),
                "selection_is_deterministic": True,
                "manual_cherry_picking": False,
                "new_categories": list(NEW_CATEGORIES),
                "missing_new_categories": [],
                "category_counts": {
                    **{category: 1 for category in NEW_CATEGORIES},
                    "experiment1_fixed::fixture": 70,
                },
                "experiment1_fixed_rows_retained": 70,
                "total_rows": len(selection_rows),
                "selection_path": str(selection_path.resolve()),
                "selection_sha256": hashlib.sha256(
                    selection_path.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    rendered = root / "rendered_examples"
    panel_dir = rendered / "panels"
    panel_dir.mkdir(parents=True)
    render_rows = []
    for row in selection_rows[: len(NEW_CATEGORIES)]:
        panel_path = panel_dir / f"{row['case_id']}.png"
        panel_path.write_bytes(f"panel {row['case_id']}".encode("utf-8"))
        render_rows.append(
            {
                "case_id": row["case_id"],
                "render_status": "rendered_from_existing_npz",
                "class_ids_json": "[0, 1]",
                "num_class_rows": 2,
                "panel_path": str(panel_path.resolve()),
                "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
                "experiment1_raw_figure": "",
                "experiment1_raw_sha256": "",
                "experiment1_minmax_figure": "",
                "experiment1_minmax_sha256": "",
            }
        )
    links = root / "fixture_links"
    links.mkdir()
    raw_path = links / "raw.png"
    minmax_path = links / "minmax.png"
    raw_path.write_bytes(b"raw fixture")
    minmax_path.write_bytes(b"minmax fixture")
    for row in selection_rows[len(NEW_CATEGORIES) :]:
        render_rows.append(
            {
                "case_id": row["case_id"],
                "render_status": "linked_existing_not_redrawn",
                "class_ids_json": "[]",
                "num_class_rows": 0,
                "panel_path": "",
                "panel_sha256": "",
                "experiment1_raw_figure": str(raw_path.resolve()),
                "experiment1_raw_sha256": hashlib.sha256(
                    raw_path.read_bytes()
                ).hexdigest(),
                "experiment1_minmax_figure": str(minmax_path.resolve()),
                "experiment1_minmax_sha256": hashlib.sha256(
                    minmax_path.read_bytes()
                ).hexdigest(),
            }
        )
    render_manifest = rendered / "render_manifest.csv"
    pd.DataFrame(render_rows).to_csv(render_manifest, index=False)
    (rendered / "render_metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "example_selection": str(selection_path.resolve()),
                "example_selection_sha256": hashlib.sha256(
                    selection_path.read_bytes()
                ).hexdigest(),
                "canonical_metadata": str(
                    (canonical / "canonical_metadata.json").resolve()
                ),
                "canonical_metadata_sha256": hashlib.sha256(
                    (canonical / "canonical_metadata.json").read_bytes()
                ).hexdigest(),
                "source_metadata": str(source_metadata_path),
                "source_metadata_sha256": hashlib.sha256(
                    source_metadata_path.read_bytes()
                ).hexdigest(),
                "output_dir": str(rendered.resolve()),
                "manifest_rows": len(render_rows),
                "rendered_panel_files": len(NEW_CATEGORIES),
                "fixed_experiment1_cases_linked": 70,
                "output_file_count": len(NEW_CATEGORIES) + 2,
                "source_npz_manifest_hashes_verified": True,
                "source_npz_unchanged": True,
                "missing_data_placeholders_generated": False,
                "model_execution": False,
                "model_loaded": False,
                "canonical_is_full_set": True,
            }
        ),
        encoding="utf-8",
    )


def _write_canonical_fixture(root: Path) -> Path:
    canonical = root / "canonical"
    canonical.mkdir(parents=True)
    images = ("img_failure", "img_train", "img_bird")
    classes = {"img_failure": 0, "img_train": 18, "img_bird": 2}
    layer_rows = []
    for image_id in images:
        class_id = classes[image_id]
        for signal in ("feature_post", "attn_c2p_conditional"):
            raw_failure = image_id == "img_failure"
            layer_rows.append(
                {
                    "model": "mctformer_plus",
                    "image_id": image_id,
                    "class_id": class_id,
                    "signal": signal,
                    "layer": 12,
                    "rho": 0.5,
                    "auc_target_bg": 0.3
                    if raw_failure and signal == "feature_post"
                    else 0.8,
                    "bg_tail_enrich_10": 1.4 if raw_failure else 0.8,
                    "score_q95": -0.7 if image_id == "img_bird" else 0.4,
                }
            )
    for index in range(3):
        for class_id in (0, 1):
            for signal in ("feature_post", "attn_c2p_conditional"):
                layer_rows.append(
                    {
                        "model": "mctformer_plus",
                        "image_id": f"shared_{index}",
                        "class_id": class_id,
                        "signal": signal,
                        "layer": 12,
                        "rho": 0.5,
                        "auc_target_bg": 0.8,
                        "bg_tail_enrich_10": 0.8,
                        "score_q95": 0.4,
                    }
                )
    pd.DataFrame(layer_rows).to_parquet(
        canonical / "per_image_class_layer_signal.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "model": "mctformer_plus",
                "image_id": image_id,
                "class_id": classes[image_id],
                "stage": "final_cam",
                "rho": 0.5,
                "auc_target_bg": 0.8,
                "bg_tail_enrich_10": 1.4 if image_id == "img_failure" else 0.8,
            }
            for image_id in images
        ]
    ).to_parquet(canonical / "per_image_class_cam_stage.parquet", index=False)
    pd.DataFrame(
        [
            {
                "model": "mctformer_plus",
                "image_id": "img_failure",
                "class_id": 0,
                "transition": transition,
                "layer": 12,
                "rho": 0.5,
                "topk_ratio": 0.1,
                "removed_background_fraction": 0.7,
                "survive_background": 0.1,
                "introduced_background_fraction": 0.6,
            }
            for transition in ("feature_post_to_attn", "c2p_cam_to_final_cam")
        ]
    ).to_parquet(canonical / "per_image_class_stage_transition.parquet", index=False)
    pd.DataFrame(
        [
            {
                "model": "mctformer_plus",
                "image_id": f"shared_{index}",
                "class_a": 0,
                "class_b": 1,
                "signal": "feature_post",
                "layer_or_stage": 12,
                "rho": 0.5,
                "topk_ratio": 0.1,
                "shared_background_fraction": background,
                "shared_target_a_fraction": target_a,
                "shared_target_b_fraction": target_b,
            }
            for index, (background, target_a, target_b) in enumerate(
                ((0.9, 0.05, 0.05), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8))
            )
        ]
    ).to_parquet(canonical / "per_shared_patch_ownership.parquet", index=False)
    pd.DataFrame().to_parquet(
        canonical / "per_class_token_pair_layer.parquet", index=False
    )
    return canonical


def test_plots_generate_all_registered_files_with_missing_slices(
    tmp_path: Path,
) -> None:
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame(
        [_aggregate_row("target_hit", 0.6, signal="feature_post", layer=12)]
    ).to_csv(tables / "layerwise_region_metrics.csv", index=False)

    result = generate_plots(tables, tmp_path / "plots", dpi=72)

    assert len(result["plots"]) == 13
    assert tuple(Path(path).name for path in result["plots"]) == PLOT_FILES
    assert all(Path(path).stat().st_size > 100 for path in result["plots"])
    assert "cam_stage_region_metrics.csv" in result["missing_or_empty_tables"]


def test_selector_covers_registered_rules_and_retains_fixed_experiment1_rows(
    tmp_path: Path,
) -> None:
    canonical = _write_canonical_fixture(tmp_path)
    exp1 = tmp_path / "experiment1"
    (exp1 / "examples").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "case_id": "case_001",
                "category": "A_fixed",
                "selection_scope_model": "mctformer",
                "selection_rank": 1,
                "image_id": "old_image",
                "class_id": 18,
                "selection_layer": 12,
                "selection_metric": "old_metric",
                "selection_value": 1.0,
                "selection_direction": "largest",
                "raw_cosine_figure": "raw.png",
                "minmax_figure": "minmax.png",
            }
        ]
    ).to_csv(exp1 / "examples/example_selection.csv", index=False)

    metadata = select_examples(
        canonical,
        tmp_path / "examples",
        per_category=2,
        experiment1_analysis_root=exp1,
    )

    selected = pd.read_csv(tmp_path / "examples/example_selection.csv")
    assert set(NEW_CATEGORIES).issubset(set(selected["category"]))
    assert metadata["experiment1_fixed_rows_retained"] == 1
    assert "experiment1_fixed::A_fixed" in set(selected["category"])
    assert selected["case_id"].is_unique
    new_rows = selected[~selected["category"].str.startswith("experiment1_fixed::")]
    multilabel = new_rows[new_rows["num_positive_classes"] >= 2]
    assert not multilabel.empty
    assert (
        (multilabel["class_a"].notna() & multilabel["class_b"].notna())
        | multilabel["companion_class_id"].notna()
    ).all()


def test_report_has_exact_sections_claim_labels_provenance_and_case_f(
    tmp_path: Path,
) -> None:
    analysis = tmp_path / "analysis"
    _write_analysis_fixture(analysis)

    result = generate_reports(analysis, analysis / "reports")

    assert result["selected_case"] == "F"
    report = (analysis / "reports/EXPERIMENT2_SEMANTIC_OWNERSHIP_REPORT.md").read_text()
    decision = (analysis / "reports/NEXT_EXPERIMENT_DECISION.md").read_text()
    headings = [line for line in report.splitlines() if line.startswith("## ")]
    assert len(headings) == 15
    assert headings[0] == "## 1. Data and Integrity"
    assert headings[-1] == "## 15. Decision for the Next Causal Experiment"
    for label in (
        "[Fact]",
        "[Statistical inference]",
        "[Interpretation candidate]",
        "[Unsupported]",
    ):
        assert label in report
    assert "/immutable/plus_final.pth" in report
    assert "plushash" in report
    assert "within-block timing offset" in report
    assert "feature_final_norm" in report
    assert "no target pixels" in report
    assert "image_id cluster" in report
    assert "top-5/top-20 composition and enrichment" in report
    assert "exact-common-`image_id,class_id` paired" in report
    assert "Selected Case: F" in decision
    assert "No intervention" in decision


def test_report_refuses_smoke_canonical_as_scientific_conclusion(
    tmp_path: Path,
) -> None:
    analysis = tmp_path / "analysis"
    _write_analysis_fixture(analysis)
    canonical = tmp_path / "canonical-smoke"
    canonical.mkdir()
    (canonical / "canonical_metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "num_manifest_images_per_model": 50,
            }
        ),
        encoding="utf-8",
    )
    metadata_path = analysis / "analysis_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["canonical_dir"] = str(canonical)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="smoke/incomplete"):
        generate_reports(analysis, analysis / "reports")


def test_report_refuses_analysis_table_changed_after_hash_manifest(
    tmp_path: Path,
) -> None:
    analysis = tmp_path / "analysis"
    _write_analysis_fixture(analysis)
    table = analysis / "tables/layerwise_region_metrics.csv"
    table.write_text(table.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="analysis table SHA-256 mismatch"):
        generate_reports(analysis, analysis / "reports")


def test_report_refuses_missing_recorded_execution_provenance(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    _write_analysis_fixture(analysis)
    (analysis.parent / "exact_commands.sh").unlink()

    with pytest.raises(RuntimeError, match="provenance artifact is missing"):
        generate_reports(analysis, analysis / "reports")


def test_report_refuses_execution_provenance_hash_drift(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    _write_analysis_fixture(analysis)
    pipeline_metadata = analysis.parent / "pipeline_metadata.json"
    pipeline_metadata.write_text(
        pipeline_metadata.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="provenance SHA-256 mismatch"):
        generate_reports(analysis, analysis / "reports")


def test_report_refuses_canonical_metadata_hash_drift(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    _write_analysis_fixture(analysis)
    metadata = json.loads(
        (analysis / "analysis_metadata.json").read_text(encoding="utf-8")
    )
    canonical_metadata = Path(metadata["canonical_dir"]) / "canonical_metadata.json"
    payload = json.loads(canonical_metadata.read_text(encoding="utf-8"))
    payload["post_analysis_mutation"] = True
    canonical_metadata.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed or was relinked after analysis"):
        generate_reports(analysis, analysis / "reports")
