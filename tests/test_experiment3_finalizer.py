from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import analysis.lazy_assignment.experiment3.generate_experiment3_report as finalizer_module
from analysis.lazy_assignment.experiment3.common import json_dump, sha256_file
from analysis.lazy_assignment.experiment3.generate_experiment3_report import (
    ALLOWED_CLAIM_LABELS,
    finalize,
    validate_analysis_roots,
)
from analysis.lazy_assignment.experiment3.render_experiment3_examples import (
    per_image_confusion_delta,
    render_compact_panel,
    select_extreme_image_deltas,
)


REPORT_FILENAMES_FOR_TEST = {
    "A": "VALIDATION_A_PRESENCE_AXIS.md",
    "B": "VALIDATION_B_CAM_LAYER_READOUT.md",
    "C": "VALIDATION_C_LATE_C2C_CAUSAL.md",
}


def test_extreme_selection_uses_lower_median_and_lexical_ties() -> None:
    frame = pd.DataFrame(
        {
            "image_id": ["z", "b", "a", "m", "q"],
            "delta": [3.0, 3.0, -2.0, 0.0, 1.0],
        }
    )
    selected = select_extreme_image_deltas(frame).set_index("rank_role")
    assert selected.loc["maximum", "image_id"] == "b"
    assert selected.loc["lower_median", "image_id"] == "q"
    assert selected.loc["minimum", "image_id"] == "a"


def _encoded_confusion(diagonal: list[int], errors: int = 0) -> bytes:
    matrix = np.zeros((21, 21), dtype="<i8")
    for class_id, value in enumerate(diagonal):
        matrix[class_id, class_id] = value
    matrix[1, 0] = errors
    return matrix.tobytes(order="C")


def test_per_image_confusion_delta_is_comparison_minus_baseline() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_id": "x",
                "variant_code": "B0",
                "confusion": _encoded_confusion([10, 5], errors=5),
            },
            {
                "image_id": "x",
                "variant_code": "B1",
                "confusion": _encoded_confusion([10, 10], errors=0),
            },
        ]
    )
    result = per_image_confusion_delta(frame, baseline="B0", comparison="B1")
    assert result.loc[0, "image_id"] == "x"
    assert result.loc[0, "delta"] > 0.0


def test_compact_panel_writes_a_png(tmp_path: Path) -> None:
    destination = tmp_path / "panel.png"
    rgb = np.full((448, 448, 3), 0.5, dtype=np.float32)
    mask = np.zeros((448, 448), dtype=np.uint8)
    mask[100:300, 100:300] = 1
    maps = [
        (f"map {index}", np.full((28, 28), index / 5, dtype=np.float32), kind)
        for index, kind in enumerate(
            ("signed", "positive", "positive", "difference", "difference")
        )
    ]
    render_compact_panel(
        rgb=rgb,
        mask=mask,
        focal_class_id=0,
        maps=maps,
        title="synthetic",
        destination=destination,
        dpi=72,
    )
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    plt.close("all")


def _report(title: str) -> str:
    return "\n".join(
        [
            f"# {title}",
            "",
            *[
                f"- {label} Synthetic categorized statement."
                for label in ALLOWED_CLAIM_LABELS
            ],
            "",
        ]
    )


def _bootstrap_table(path: Path, records: list[dict[str, object]]) -> None:
    frame = pd.DataFrame.from_records(records)
    frame["bootstrap_repeats"] = 5000
    frame["bootstrap_unit"] = "image"
    frame.to_csv(path, index=False)


def _series_record(
    *,
    series: str,
    metric: str,
    estimate: float,
    low: float,
    high: float,
    **identity: object,
) -> dict[str, object]:
    return {
        "series": series,
        "metric": metric,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        **identity,
    }


def _write_generated_inventory(root: Path, names: list[str]) -> dict[str, object]:
    return {
        name: {
            "bytes": (root / name).stat().st_size,
            "sha256": sha256_file(root / name),
        }
        for name in names
    }


def _build_full_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    run = tmp_path / "full_run"
    a = run / "presence_axis" / "analysis"
    b = run / "cam_layer_intervention" / "analysis"
    c = run / "c2c_intervention" / "analysis"
    for root in (a, b, c):
        root.mkdir(parents=True)

    (a / "VALIDATION_A_PRESENCE_AXIS.md").write_text(
        _report("Validation A"), encoding="utf-8"
    )
    a_decision = {
        "decision": "strong_support",
        "primary_statistics": {
            "token_pair_residual_minus_raw": {
                "estimate": -0.2,
                "ci_low": -0.3,
                "ci_high": -0.1,
            },
            "map_top10_both_removed_minus_raw": {
                "estimate": -0.1,
                "ci_low": -0.2,
                "ci_high": -0.05,
            },
            "perp_norm_minus_raw_norm_spearman": {
                "estimate": 0.2,
                "ci_low": 0.1,
                "ci_high": 0.3,
            },
            "l12_oof_projection_auroc": {
                "estimate": 0.8,
                "ci_low": 0.7,
                "ci_high": 0.9,
            },
            "l12_min_signed_alignment_across_fit_folds": 0.95,
        },
    }
    json_dump(a / "validation_a_decision.json", a_decision)
    _bootstrap_table(
        a / "paired_bootstrap.csv",
        [{"metric": "delta", "estimate": -0.1}],
    )
    a_manifest_rows = []
    for name in (
        "VALIDATION_A_PRESENCE_AXIS.md",
        "validation_a_decision.json",
        "paired_bootstrap.csv",
    ):
        path = a / name
        a_manifest_rows.append(
            {
                "relative_path": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    with (a / "artifact_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("relative_path", "size_bytes", "sha256")
        )
        writer.writeheader()
        writer.writerows(a_manifest_rows)
    json_dump(
        a / "metadata.json",
        {
            "status": "complete",
            "run_kind": "full",
            "processed_images": 1449,
            "input_runs": {"mctformer": "synthetic", "mctformer_plus": "synthetic"},
            "bootstrap": {"repeats": 5000, "unit": "image"},
            "source_immutability_verified": True,
            "artifact_manifest": {
                "path": str(a / "artifact_manifest.csv"),
                "rows": len(a_manifest_rows),
                "sha256": sha256_file(a / "artifact_manifest.csv"),
            },
        },
    )
    json_dump(
        a / "completion.json",
        {
            "status": "complete",
            "run_kind": "full",
            "num_images": 1449,
            "bootstrap_repeats": 5000,
            "source_immutability_verified": True,
        },
    )

    (b / "VALIDATION_B_CAM_LAYER_READOUT.md").write_text(
        _report("Validation B"), encoding="utf-8"
    )
    _bootstrap_table(
        b / "paired_cam_bootstrap.csv",
        [
            _series_record(
                series="B1_minus_B0",
                metric="mean_iou",
                estimate=0.01,
                low=0.002,
                high=0.02,
                model="mctformer_plus",
                label_stratum="all",
                threshold=0.45,
            ),
            _series_record(
                series="B1_minus_B0",
                metric="binary_foreground_recall",
                estimate=0.0,
                low=-0.001,
                high=0.002,
                model="mctformer_plus",
                label_stratum="all",
                threshold=0.45,
            ),
        ],
    )
    _bootstrap_table(
        b / "paired_region_bootstrap.csv",
        [
            _series_record(
                series="B1_minus_B0",
                metric="target_other_auroc",
                estimate=0.05,
                low=0.02,
                high=0.08,
                aggregation="micro",
                model="mctformer_plus",
                label_stratum="all",
                stage="attention",
                rho=0.5,
            )
        ],
    )
    _bootstrap_table(
        b / "paired_class_pair_bootstrap.csv",
        [
            _series_record(
                series="B1_minus_B0",
                metric="top10_jaccard",
                estimate=-0.05,
                low=-0.08,
                high=-0.02,
                aggregation="micro",
                model="mctformer_plus",
                label_stratum="all",
                stage="attention",
                rho=0.5,
            )
        ],
    )
    b_names = [
        "VALIDATION_B_CAM_LAYER_READOUT.md",
        "paired_cam_bootstrap.csv",
        "paired_region_bootstrap.csv",
        "paired_class_pair_bootstrap.csv",
    ]
    json_dump(
        b / "analysis_metadata.json",
        {
            "status": "complete",
            "run_kind": "full",
            "num_images": 1449,
            "positive_image_class_pairs": 2147,
            "multilabel_images": 522,
            "models": ["mctformer", "mctformer_plus"],
            "variants": ["B0", "B1", "B2", "B3", "B4", "B5"],
            "bootstrap": {
                "repeats": 5000,
                "unit": "whole image multinomial multiplicity",
                "same_draws_reused_for_paired_variants_and_hosts": True,
            },
            "input_hashes_before_and_after_equal": True,
            "source_immutability_verified": True,
            "generated_files": _write_generated_inventory(b, b_names),
        },
    )

    (c / "VALIDATION_C_LATE_C2C_CAUSAL.md").write_text(
        _report("Validation C"), encoding="utf-8"
    )
    _bootstrap_table(
        c / "paired_cam_bootstrap.csv",
        [
            _series_record(
                series="C4_minus_C0",
                metric="mean_iou",
                estimate=0.01,
                low=0.002,
                high=0.02,
                model="mctformer_plus",
                label_stratum="all",
                threshold=0.45,
            )
        ],
    )
    _bootstrap_table(
        c / "paired_region_bootstrap.csv",
        [
            _series_record(
                series="C4_minus_C0",
                metric=metric,
                estimate=estimate,
                low=low,
                high=high,
                aggregation="micro",
                model="mctformer_plus",
                summary_stratum="all",
                map_family="attention_conditional",
                layer=12,
                rho_name="rho05",
            )
            for metric, estimate, low, high in (
                ("auc_target_other", 0.03, 0.01, 0.05),
                ("conditional_bg_mass", -0.01, -0.02, -0.001),
            )
        ],
    )
    _bootstrap_table(
        c / "paired_class_pair_bootstrap.csv",
        [
            _series_record(
                series="C4_minus_C0",
                metric="attention_top10_jaccard",
                estimate=-0.05,
                low=-0.08,
                high=-0.01,
                aggregation="micro",
                model="mctformer_plus",
                summary_stratum="all",
                layer=12,
            )
        ],
    )
    pd.DataFrame(
        [
            {
                "model": "mctformer_plus",
                "comparison": "C4_minus_C0",
                "logit_source": source,
                "delta_map": 0.001,
                "ci_low": 0.0001,
                "ci_high": 0.002,
                "noninferiority_pass": True,
            }
            for source in ("class_token", "patch_head")
        ]
    ).to_csv(c / "classification_noninferiority.csv", index=False)
    c_names = [
        "VALIDATION_C_LATE_C2C_CAUSAL.md",
        "paired_cam_bootstrap.csv",
        "paired_region_bootstrap.csv",
        "paired_class_pair_bootstrap.csv",
        "classification_noninferiority.csv",
    ]
    json_dump(
        c / "analysis_metadata.json",
        {
            "status": "complete",
            "run_kind": "full",
            "num_images": 1449,
            "positive_image_class_pairs": 2147,
            "multilabel_images": 522,
            "models": ["mctformer_plus"],
            "variants": ["C0", "C1", "C2", "C3", "C4", "C5"],
            "bootstrap": {
                "repeats": 5000,
                "unit": "whole image multinomial multiplicity",
                "same_draws_reused_across_variants_metrics_and_models": True,
            },
            "input_hashes_before_and_after_equal": True,
            "verified_experiment2_sources_before_after": 1449,
            "verified_input_artifacts_before_after": 6 * 1449,
            "generated_files": _write_generated_inventory(c, c_names),
        },
    )

    for validation, parent in (
        ("A", run / "presence_axis"),
        ("B", run / "cam_layer_intervention"),
        ("C", run / "c2c_intervention"),
    ):
        examples = parent / "examples"
        examples.mkdir()
        rows = []
        for index, role in enumerate(("maximum", "lower_median", "minimum")):
            panel = examples / f"{role}.png"
            panel.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")
            rows.append(
                {
                    "rank_role": role,
                    "image_id": f"image-{index}",
                    "selection_was_manual": False,
                    "panel_path": str(panel),
                    "panel_sha256": sha256_file(panel),
                }
            )
        selection_path = examples / "selection.csv"
        pd.DataFrame(rows).to_csv(selection_path, index=False)
        json_dump(
            examples / "render_metadata.json",
            {
                "status": "complete",
                "run_kind": "full",
                "validation": validation,
                "selected_examples": 3,
                "selection_was_manual": False,
                "selection_sha256": sha256_file(selection_path),
                "input_hashes": {
                    str(
                        parent / "analysis" / REPORT_FILENAMES_FOR_TEST[validation]
                    ): sha256_file(
                        parent / "analysis" / REPORT_FILENAMES_FOR_TEST[validation]
                    )
                },
            },
        )

    audit = run / "audit"
    verification = audit / "verification"
    verification.mkdir(parents=True)
    source_file = audit / "immutable-source.bin"
    source_file.write_bytes(b"source")
    before = audit / "immutable_manifest_before.csv"
    pd.DataFrame(
        [
            {
                "absolute_path": str(source_file),
                "size_bytes": source_file.stat().st_size,
                "sha256": sha256_file(source_file),
            }
        ]
    ).to_csv(before, index=False)
    pd.DataFrame(
        [
            {
                "absolute_path": str(source_file),
                "size_bytes": source_file.stat().st_size,
                "sha256": sha256_file(source_file),
                "exists_after": True,
                "size_unchanged": True,
                "sha256_unchanged": True,
                "size_bytes_after": source_file.stat().st_size,
                "sha256_after": sha256_file(source_file),
            }
        ]
    ).to_csv(verification / "immutable_manifest_after.csv", index=False)
    json_dump(
        audit / "source_metadata.json",
        {
            "status": "complete",
            "integrity_passed": True,
            "checkpoints": {
                model: {
                    "passed": True,
                    "actual_sha256": "a" * 64,
                    "expected_sha256": "a" * 64,
                }
                for model in ("mctformer", "mctformer_plus")
            },
            "dataset": {
                "num_images": 1449,
                "positive_image_class_pairs": 2147,
                "multilabel_images": 522,
            },
            "immutable_manifest": {
                "path": str(before),
                "rows": 1,
                "sha256": sha256_file(before),
            },
        },
    )
    verification_json = verification / "immutability_verification.json"
    json_dump(
        verification_json,
        {
            "status": "complete",
            "integrity_passed": True,
            "before_manifest": str(before),
            "files_checked": 1,
            "bytes_checked": source_file.stat().st_size,
            "missing_files": 0,
            "size_changed_files": 0,
            "sha256_changed_files": 0,
        },
    )
    (run / "exact_commands.sh").write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
    commands = run / "exact_commands.sh"
    json_dump(
        run / "pipeline_metadata.json",
        {
            "status": "running",
            "pipeline": "synthetic_experiment3",
            "run_root": str(run),
            "stage_history": ["audit", "A", "B", "C"],
            "stage_order": ["input_audit", "final_reports"],
            "evaluation": {
                "images": 1449,
                "bootstrap_repeats": 5000,
                "bootstrap_unit": "image cluster",
            },
            "exact_commands": {
                "path": str(commands),
                "sha256": sha256_file(commands),
            },
        },
    )
    json_dump(
        run / "pipeline_status.json",
        {
            "status": "running",
            "pipeline": "synthetic_experiment3",
            "run_root": str(run),
            "active_stage": "final_reports",
            "stage_history": ["audit", "A", "B", "C"],
            "stages": {
                "input_audit": {"status": "complete"},
                "final_reports": {"status": "running", "started_at": "fixed"},
            },
        },
    )
    return run, a, b, c, verification_json


def test_finalizer_preserves_history_and_writes_a_hashed_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "tgca-repro")
    run, a, b, c, verification = _build_full_bundle(tmp_path)
    command_hash = sha256_file(run / "exact_commands.sh")
    result = finalize(
        run_root=run,
        validation_a_root=a,
        validation_b_root=b,
        validation_c_root=c,
        source_verification=verification,
    )
    assert result["status"] == "complete"
    assert sha256_file(run / "exact_commands.sh") == command_hash
    for name in (
        "VALIDATION_A_PRESENCE_AXIS.md",
        "VALIDATION_B_CAM_LAYER_READOUT.md",
        "VALIDATION_C_LATE_C2C_CAUSAL.md",
        "EXPERIMENT3_COMBINED_REPORT.md",
        "NEXT_METHOD_DECISION.md",
    ):
        assert (run / "reports" / name).is_file()
    pipeline = json.loads((run / "pipeline_status.json").read_text())
    assert pipeline["status"] == "complete"
    assert pipeline["stage_history"] == ["audit", "A", "B", "C"]
    assert pipeline["stages"]["final_reports"]["status"] == "complete"
    assert pipeline["stages"]["final_reports"]["started_at"] == "fixed"
    manifest = pd.read_csv(run / "artifact_manifest.csv")
    assert not manifest.empty
    assert (
        sha256_file(run / "artifact_manifest.csv")
        in (run / "artifact_manifest.sha256").read_text()
    )
    combined = (run / "reports" / "EXPERIMENT3_COMBINED_REPORT.md").read_text()
    assert all(label in combined for label in ALLOWED_CLAIM_LABELS)


def test_production_validator_rejects_smoke_metadata(tmp_path: Path) -> None:
    run, a, b, c, _ = _build_full_bundle(tmp_path)
    metadata = json.loads((b / "analysis_metadata.json").read_text())
    metadata["run_kind"] = "smoke"
    json_dump(b / "analysis_metadata.json", metadata)
    with pytest.raises(RuntimeError, match="not a full run"):
        validate_analysis_roots(run, a, b, c)


def test_mixed_presence_decision_is_accepted_but_not_mapped_to_outcome3(
    tmp_path: Path,
) -> None:
    run, a, b, c, _ = _build_full_bundle(tmp_path)
    analyses = validate_analysis_roots(run, a, b, c)
    decision_path = a / "validation_a_decision.json"
    decision = json.loads(decision_path.read_text())
    decision["decision"] = "mixed_or_indeterminate"
    json_dump(decision_path, decision)
    rows = finalizer_module._primary_rows(analyses)
    flags = finalizer_module._decision_flags(rows)
    outcomes = finalizer_module._matched_outcomes(flags)
    assert flags["presence_decision"] == "mixed_or_indeterminate"
    assert "Outcome 3" not in outcomes


def test_noninferiority_failure_is_not_classification_harm_or_outcome7() -> None:
    rows = {
        "a": {"decision": "mixed_or_indeterminate"},
        "b_auc": pd.Series({"ci_low": -0.1, "ci_high": 0.1}),
        "b_jaccard": pd.Series({"ci_low": -0.1, "ci_high": 0.1}),
        "b_cam": pd.Series({"ci_low": -0.1, "ci_high": 0.1}),
        "b_recall": pd.Series({"ci_low": -0.1, "ci_high": 0.1}),
        "c_auc": pd.Series({"ci_low": -0.1, "ci_high": 0.1}),
        "c_jaccard": pd.Series({"ci_low": -0.1, "ci_high": 0.1}),
        "c_cam": pd.Series({"ci_low": -0.03, "ci_high": -0.01}),
        "c_bg": pd.Series({"ci_low": -0.1, "ci_high": 0.1}),
        "c_noninferiority": pd.DataFrame(
            [
                {
                    "logit_source": source,
                    "delta_map": -0.002,
                    "ci_low": -0.005,
                    "ci_high": 0.001,
                    "noninferiority_pass": False,
                }
                for source in ("class_token", "patch_head")
            ]
        ),
    }
    flags = finalizer_module._decision_flags(rows)
    assert flags["c_classification_noninferiority_pass"] is False
    assert flags["c_classification_harmed"] is False
    assert flags["c_joint_benefit_gate_passed"] is False
    assert "Outcome 7" not in finalizer_module._matched_outcomes(flags)
    assert "inconclusive" in finalizer_module._decision_report(flags)


def test_outcome7_requires_both_classification_sources_and_cam_harm() -> None:
    rows = {
        "a": {"decision": "mixed_or_indeterminate"},
        "b_auc": pd.Series({"ci_low": -0.1, "ci_high": 0.1}),
        "b_jaccard": pd.Series({"ci_low": -0.1, "ci_high": 0.1}),
        "b_cam": pd.Series({"ci_low": -0.1, "ci_high": 0.1}),
        "b_recall": pd.Series({"ci_low": -0.1, "ci_high": 0.1}),
        "c_auc": pd.Series({"ci_low": -0.1, "ci_high": 0.1}),
        "c_jaccard": pd.Series({"ci_low": -0.1, "ci_high": 0.1}),
        "c_cam": pd.Series({"ci_low": -0.03, "ci_high": -0.01}),
        "c_bg": pd.Series({"ci_low": -0.1, "ci_high": 0.1}),
        "c_noninferiority": pd.DataFrame(
            [
                {
                    "logit_source": source,
                    "delta_map": -0.02,
                    "ci_low": -0.03,
                    "ci_high": -0.01,
                    "noninferiority_pass": False,
                }
                for source in ("class_token", "patch_head")
            ]
        ),
    }
    flags = finalizer_module._decision_flags(rows)
    assert flags["c_classification_harmed"] is True
    assert flags["c_cam_harmed"] is True
    assert "Outcome 7" in finalizer_module._matched_outcomes(flags)


def test_finalizer_rolls_back_complete_state_if_manifest_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "tgca-repro")
    run, a, b, c, verification = _build_full_bundle(tmp_path)
    metadata_before = (run / "pipeline_metadata.json").read_bytes()
    status_before = (run / "pipeline_status.json").read_bytes()

    def fail_manifest(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic manifest failure")

    monkeypatch.setattr(finalizer_module, "_write_artifact_manifest", fail_manifest)
    with pytest.raises(RuntimeError, match="synthetic manifest failure"):
        finalize(
            run_root=run,
            validation_a_root=a,
            validation_b_root=b,
            validation_c_root=c,
            source_verification=verification,
        )
    assert (run / "pipeline_metadata.json").read_bytes() == metadata_before
    assert (run / "pipeline_status.json").read_bytes() == status_before
    assert not (run / "reports").exists()
    assert not (run / "artifact_manifest.csv").exists()
