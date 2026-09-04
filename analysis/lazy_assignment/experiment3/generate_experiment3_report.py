#!/usr/bin/env python3
"""Finalize the three immutable Experiment 3 validation analyses.

This is deliberately a *finalizer*, not another analysis pass.  It accepts the
completed A/B/C analysis directories, verifies their production scope and
bootstrap contract, verifies the post-run source-immutability audit, copies the
three independent reports, and writes the combined decision artifacts.  The
existing top-level pipeline JSON files are merged atomically so orchestration
history is retained; ``exact_commands.sh`` is never modified.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment3.common import (  # noqa: E402
    BOOTSTRAP_REPEATS,
    EXPECTED_IMAGES,
    EXPECTED_MULTILABEL_IMAGES,
    EXPECTED_POSITIVE_PAIRS,
    json_dump,
    read_json,
    require_tgca_repro,
    sha256_file,
    timestamp,
)


REPORT_FILENAMES = {
    "A": "VALIDATION_A_PRESENCE_AXIS.md",
    "B": "VALIDATION_B_CAM_LAYER_READOUT.md",
    "C": "VALIDATION_C_LATE_C2C_CAUSAL.md",
}
ALLOWED_CLAIM_LABELS = (
    "[Fact]",
    "[Statistical inference]",
    "[Mechanistic interpretation]",
    "[Unsupported]",
)
BOOTSTRAP_UNITS = {"image", "whole image multinomial multiplicity"}


@dataclass(frozen=True)
class ValidatedAnalyses:
    run_root: Path
    roots: Mapping[str, Path]
    metadata: Mapping[str, Mapping[str, object]]
    reports: Mapping[str, Path]
    control_hashes: Mapping[Path, str]


def _inside(root: Path, candidate: Path, *, label: str) -> Path:
    root = root.expanduser().resolve()
    candidate = candidate.expanduser().resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"{label} must be inside run root {root}: {candidate}")
    return candidate


def _require_complete_full(metadata: Mapping[str, object], *, validation: str) -> None:
    if metadata.get("status") != "complete":
        raise RuntimeError(f"Validation {validation} analysis is not complete")
    if metadata.get("run_kind") != "full":
        raise RuntimeError(f"Validation {validation} is not a full run")
    image_key = "processed_images" if validation == "A" else "num_images"
    if int(metadata.get(image_key, -1)) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Validation {validation} must contain {EXPECTED_IMAGES} images"
        )
    bootstrap = metadata.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise TypeError(f"Validation {validation} lacks bootstrap metadata")
    if int(bootstrap.get("repeats", -1)) != BOOTSTRAP_REPEATS:
        raise RuntimeError(
            f"Validation {validation} must use {BOOTSTRAP_REPEATS} bootstraps"
        )
    if str(bootstrap.get("unit")) not in BOOTSTRAP_UNITS:
        raise RuntimeError(
            f"Validation {validation} bootstrap unit is not the whole image"
        )


def _verify_bootstrap_csvs(root: Path) -> None:
    paths = sorted(root.rglob("*.csv"))
    checked = 0
    for path in paths:
        header = pd.read_csv(path, nrows=0)
        if "bootstrap_repeats" not in header.columns:
            continue
        frame = pd.read_csv(
            path,
            usecols=[
                column
                for column in ("bootstrap_repeats", "bootstrap_unit")
                if column in header.columns
            ],
        )
        if frame.empty:
            raise RuntimeError(f"empty bootstrap table: {path}")
        repeats = pd.to_numeric(frame["bootstrap_repeats"], errors="raise").dropna()
        # Some compact tables deliberately mix fixed descriptive rows (no CI,
        # hence a blank repeat count) with inferential rows.  Every inferential
        # row must still use the production count.
        if repeats.empty or not (repeats == BOOTSTRAP_REPEATS).all():
            raise RuntimeError(f"non-production bootstrap count in {path}")
        if "bootstrap_unit" in frame:
            units = set(frame["bootstrap_unit"].dropna().astype(str))
            if units != {"image"}:
                raise RuntimeError(f"non-image bootstrap unit in {path}: {units}")
        checked += 1
    if checked == 0:
        raise RuntimeError(f"no machine-readable bootstrap tables found in {root}")


def _verify_generated_files(root: Path, metadata: Mapping[str, object]) -> None:
    generated = metadata.get("generated_files")
    if generated is None:
        return
    if not isinstance(generated, Mapping) or not generated:
        raise TypeError(f"invalid generated_files inventory in {root}")
    for relative, record in generated.items():
        if not isinstance(relative, str) or not isinstance(record, Mapping):
            raise TypeError(f"invalid generated_files record in {root}")
        path = _inside(root, root / relative, label="generated artifact")
        if not path.is_file():
            raise FileNotFoundError(path)
        if int(record.get("bytes", -1)) != path.stat().st_size:
            raise RuntimeError(f"generated artifact size mismatch: {path}")
        if str(record.get("sha256", "")) != sha256_file(path):
            raise RuntimeError(f"generated artifact hash mismatch: {path}")


def _verify_a_manifest(root: Path, metadata: Mapping[str, object]) -> None:
    manifest_record = metadata.get("artifact_manifest")
    if not isinstance(manifest_record, Mapping):
        raise TypeError("Validation A lacks artifact_manifest metadata")
    path = _inside(
        root,
        Path(str(manifest_record.get("path", ""))),
        label="Validation A artifact manifest",
    )
    if not path.is_file() or sha256_file(path) != str(
        manifest_record.get("sha256", "")
    ):
        raise RuntimeError("Validation A artifact manifest hash mismatch")
    frame = pd.read_csv(path)
    required = {"relative_path", "size_bytes", "sha256"}
    if set(frame.columns) != required:
        raise RuntimeError("Validation A artifact manifest schema mismatch")
    if len(frame) != int(manifest_record.get("rows", -1)):
        raise RuntimeError("Validation A artifact manifest row-count mismatch")
    for row in frame.to_dict(orient="records"):
        artifact = _inside(root, root / str(row["relative_path"]), label="A artifact")
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        if artifact.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"Validation A artifact size mismatch: {artifact}")
        if sha256_file(artifact) != str(row["sha256"]):
            raise RuntimeError(f"Validation A artifact hash mismatch: {artifact}")


def _require_report_labels(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    bracketed = all(label in text for label in ALLOWED_CLAIM_LABELS)
    headed = all(
        heading in text
        for heading in (
            "## Fact",
            "## Statistical inference",
            "## Mechanistic interpretation",
            "## Unsupported",
        )
    )
    if not bracketed and not headed:
        raise RuntimeError(f"report lacks claim-category labels: {path}")


def validate_analysis_roots(
    run_root: Path,
    validation_a_root: Path,
    validation_b_root: Path,
    validation_c_root: Path,
) -> ValidatedAnalyses:
    """Validate production A/B/C outputs without writing anything."""

    run_root = run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    roots = {
        "A": _inside(run_root, validation_a_root, label="Validation A root"),
        "B": _inside(run_root, validation_b_root, label="Validation B root"),
        "C": _inside(run_root, validation_c_root, label="Validation C root"),
    }
    for name, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(root)

    metadata_paths = {
        "A": roots["A"] / "metadata.json",
        "B": roots["B"] / "analysis_metadata.json",
        "C": roots["C"] / "analysis_metadata.json",
    }
    metadata = {name: read_json(path) for name, path in metadata_paths.items()}
    for name, value in metadata.items():
        _require_complete_full(value, validation=name)

    a_completion = read_json(roots["A"] / "completion.json")
    if (
        a_completion.get("status") != "complete"
        or a_completion.get("run_kind") != "full"
        or int(a_completion.get("num_images", -1)) != EXPECTED_IMAGES
        or int(a_completion.get("bootstrap_repeats", -1)) != BOOTSTRAP_REPEATS
        or a_completion.get("source_immutability_verified") is not True
    ):
        raise RuntimeError("Validation A completion gate failed")
    _verify_a_manifest(roots["A"], metadata["A"])
    a_inputs = metadata["A"].get("input_runs")
    if (
        metadata["A"].get("source_immutability_verified") is not True
        or not isinstance(a_inputs, Mapping)
        or set(a_inputs) != {"mctformer", "mctformer_plus"}
    ):
        raise RuntimeError("Validation A paired-host/source integrity gate failed")

    for name in ("B", "C"):
        value = metadata[name]
        if int(value.get("positive_image_class_pairs", -1)) != EXPECTED_POSITIVE_PAIRS:
            raise RuntimeError(f"Validation {name} positive-pair count mismatch")
        if int(value.get("multilabel_images", -1)) != EXPECTED_MULTILABEL_IMAGES:
            raise RuntimeError(f"Validation {name} multi-label count mismatch")
        if value.get("input_hashes_before_and_after_equal") is not True:
            raise RuntimeError(f"Validation {name} input immutability gate failed")
        _verify_generated_files(roots[name], value)
    if (
        set(metadata["B"].get("models", ())) != {"mctformer", "mctformer_plus"}
        or list(metadata["B"].get("variants", ()))
        != ["B0", "B1", "B2", "B3", "B4", "B5"]
        or metadata["B"].get("source_immutability_verified") is not True
        or metadata["B"]["bootstrap"].get(  # type: ignore[index,union-attr]
            "same_draws_reused_for_paired_variants_and_hosts"
        )
        is not True
    ):
        raise RuntimeError("Validation B host/variant/pairing gate failed")
    c_models = set(metadata["C"].get("models", ()))
    if (
        "mctformer_plus" not in c_models
        or not c_models.issubset({"mctformer", "mctformer_plus"})
        or list(metadata["C"].get("variants", ()))
        != ["C0", "C1", "C2", "C3", "C4", "C5"]
        or metadata["C"]["bootstrap"].get(  # type: ignore[index,union-attr]
            "same_draws_reused_across_variants_metrics_and_models"
        )
        is not True
        or int(metadata["C"].get("verified_experiment2_sources_before_after", -1))
        < EXPECTED_IMAGES
        or int(metadata["C"].get("verified_input_artifacts_before_after", -1))
        < 6 * EXPECTED_IMAGES
    ):
        raise RuntimeError("Validation C host/variant/source-pairing gate failed")

    reports = {
        name: roots[name] / filename for name, filename in REPORT_FILENAMES.items()
    }
    for path in reports.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        _require_report_labels(path)
    for root in roots.values():
        _verify_bootstrap_csvs(root)

    controls = {
        path: sha256_file(path)
        for path in [
            *metadata_paths.values(),
            roots["A"] / "completion.json",
            *reports.values(),
        ]
    }
    return ValidatedAnalyses(run_root, roots, metadata, reports, controls)


def validate_source_verification(
    source_verification: Path, *, run_root: Path
) -> Mapping[str, object]:
    path = _inside(run_root, source_verification, label="source verification")
    payload = read_json(path)
    if (
        payload.get("status") != "complete"
        or payload.get("integrity_passed") is not True
        or int(payload.get("missing_files", -1)) != 0
        or int(payload.get("size_changed_files", -1)) != 0
        or int(payload.get("sha256_changed_files", -1)) != 0
        or int(payload.get("files_checked", 0)) <= 0
        or int(payload.get("bytes_checked", 0)) <= 0
    ):
        raise RuntimeError("post-run source immutability verification is not a PASS")
    before = _inside(
        run_root,
        Path(str(payload.get("before_manifest", ""))),
        label="immutable before-manifest",
    )
    if not before.is_file():
        raise FileNotFoundError(before)
    before_frame = pd.read_csv(before)
    required_before = {"absolute_path", "size_bytes", "sha256"}
    if not required_before.issubset(before_frame.columns):
        raise RuntimeError("immutable before-manifest schema mismatch")
    if len(before_frame) != int(payload["files_checked"]):
        raise RuntimeError("source verification/before-manifest row mismatch")
    after_path = path.parent / "immutable_manifest_after.csv"
    if not after_path.is_file():
        raise FileNotFoundError(after_path)
    after = pd.read_csv(after_path)
    required_after = {
        "exists_after",
        "size_unchanged",
        "sha256_unchanged",
        "size_bytes_after",
        "sha256_after",
    }
    if not required_after.issubset(after.columns) or len(after) != len(before_frame):
        raise RuntimeError("immutable after-manifest schema/count mismatch")
    for column in ("exists_after", "size_unchanged", "sha256_unchanged"):
        if not after[column].map(_as_bool).all():
            raise RuntimeError(f"immutable after-manifest contains failed {column}")
    if not np_array_equal_text(
        before_frame["absolute_path"], after["absolute_path"]
    ) or not np_array_equal_text(before_frame["sha256"], after["sha256_after"]):
        raise RuntimeError("immutable before/after paths or hashes differ")
    if not (
        pd.to_numeric(before_frame["size_bytes"], errors="raise").to_numpy()
        == pd.to_numeric(after["size_bytes_after"], errors="raise").to_numpy()
    ).all():
        raise RuntimeError("immutable before/after sizes differ")
    source_metadata_path = run_root / "audit" / "source_metadata.json"
    source_metadata = read_json(source_metadata_path)
    manifest = source_metadata.get("immutable_manifest")
    dataset = source_metadata.get("dataset")
    checkpoints = source_metadata.get("checkpoints")
    checkpoints_pass = (
        isinstance(checkpoints, Mapping)
        and set(checkpoints) == {"mctformer", "mctformer_plus"}
        and all(
            isinstance(record, Mapping)
            and record.get("passed") is True
            and record.get("actual_sha256") == record.get("expected_sha256")
            and len(str(record.get("actual_sha256", ""))) == 64
            for record in checkpoints.values()
        )
    )
    if (
        source_metadata.get("status") != "complete"
        or source_metadata.get("integrity_passed") is not True
        or not isinstance(manifest, Mapping)
        or not isinstance(dataset, Mapping)
        or not checkpoints_pass
        or Path(str(manifest.get("path", ""))).resolve() != before
        or int(manifest.get("rows", -1)) != len(before_frame)
        or sha256_file(before) != str(manifest.get("sha256", ""))
        or int(dataset.get("num_images", -1)) != EXPECTED_IMAGES
        or int(dataset.get("positive_image_class_pairs", -1)) != EXPECTED_POSITIVE_PAIRS
        or int(dataset.get("multilabel_images", -1)) != EXPECTED_MULTILABEL_IMAGES
    ):
        raise RuntimeError("source audit linkage/count gate failed")
    return payload


def np_array_equal_text(left: pd.Series, right: pd.Series) -> bool:
    """Exact ordered text comparison without NumPy as a report dependency."""

    return left.astype(str).tolist() == right.astype(str).tolist()


def _single_row(frame: pd.DataFrame, **filters: object) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        if column not in selected:
            raise KeyError(f"missing required column {column!r}")
        if isinstance(value, float):
            selected = selected[
                pd.to_numeric(selected[column], errors="coerce").sub(value).abs() < 1e-9
            ]
        else:
            selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise RuntimeError(f"expected one row for {filters}, found {len(selected)}")
    row = selected.iloc[0]
    values = [float(row[name]) for name in ("estimate", "ci_low", "ci_high")]
    if not all(math.isfinite(value) for value in values) or values[1] > values[2]:
        raise RuntimeError(f"invalid estimate/CI for {filters}")
    return row


def _format_ci(row: pd.Series) -> str:
    return (
        f"{float(row['estimate']):+.4f} "
        f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}]"
    )


def _primary_rows(analyses: ValidatedAnalyses) -> Mapping[str, object]:
    a_decision = read_json(analyses.roots["A"] / "validation_a_decision.json")
    if a_decision.get("decision") not in {
        "strong_support",
        "partial_support_token_geometry_only",
        "not_supported",
        "mixed_or_indeterminate",
    }:
        raise RuntimeError("Validation A lacks an evaluable full-run decision")
    b_cam = pd.read_csv(analyses.roots["B"] / "paired_cam_bootstrap.csv")
    b_region = pd.read_csv(analyses.roots["B"] / "paired_region_bootstrap.csv")
    b_pair = pd.read_csv(analyses.roots["B"] / "paired_class_pair_bootstrap.csv")
    c_cam = pd.read_csv(analyses.roots["C"] / "paired_cam_bootstrap.csv")
    c_region = pd.read_csv(analyses.roots["C"] / "paired_region_bootstrap.csv")
    c_pair = pd.read_csv(analyses.roots["C"] / "paired_class_pair_bootstrap.csv")
    c_noninferiority = pd.read_csv(
        analyses.roots["C"] / "classification_noninferiority.csv"
    )
    common = {"aggregation": "micro", "model": "mctformer_plus"}
    cam_common = {"model": "mctformer_plus"}
    return {
        "a": a_decision,
        "b_cam": _single_row(
            b_cam,
            series="B1_minus_B0",
            metric="mean_iou",
            label_stratum="all",
            threshold=0.45,
            **cam_common,
        ),
        "b_recall": _single_row(
            b_cam,
            series="B1_minus_B0",
            metric="binary_foreground_recall",
            label_stratum="all",
            threshold=0.45,
            **cam_common,
        ),
        "b_auc": _single_row(
            b_region,
            series="B1_minus_B0",
            metric="target_other_auroc",
            label_stratum="all",
            stage="attention",
            rho=0.5,
            **common,
        ),
        "b_jaccard": _single_row(
            b_pair,
            series="B1_minus_B0",
            metric="top10_jaccard",
            label_stratum="all",
            stage="attention",
            rho=0.5,
            **common,
        ),
        "c_cam": _single_row(
            c_cam,
            series="C4_minus_C0",
            metric="mean_iou",
            label_stratum="all",
            threshold=0.45,
            **cam_common,
        ),
        "c_auc": _single_row(
            c_region,
            series="C4_minus_C0",
            metric="auc_target_other",
            summary_stratum="all",
            map_family="attention_conditional",
            layer=12,
            rho_name="rho05",
            **common,
        ),
        "c_bg": _single_row(
            c_region,
            series="C4_minus_C0",
            metric="conditional_bg_mass",
            summary_stratum="all",
            map_family="attention_conditional",
            layer=12,
            rho_name="rho05",
            **common,
        ),
        "c_jaccard": _single_row(
            c_pair,
            series="C4_minus_C0",
            metric="attention_top10_jaccard",
            summary_stratum="all",
            layer=12,
            **common,
        ),
        "c_noninferiority": c_noninferiority[
            (c_noninferiority["model"] == "mctformer_plus")
            & (c_noninferiority["comparison"] == "C4_minus_C0")
        ].copy(),
    }


def _decision_flags(rows: Mapping[str, object]) -> Mapping[str, object]:
    def positive(key: str) -> bool:
        row = rows[key]
        assert isinstance(row, pd.Series)
        return float(row["ci_low"]) > 0.0

    def negative(key: str) -> bool:
        row = rows[key]
        assert isinstance(row, pd.Series)
        return float(row["ci_high"]) < 0.0

    def overlaps_zero(key: str) -> bool:
        row = rows[key]
        assert isinstance(row, pd.Series)
        return float(row["ci_low"]) <= 0.0 <= float(row["ci_high"])

    ni = rows["c_noninferiority"]
    assert isinstance(ni, pd.DataFrame)
    required_sources = {"class_token", "patch_head"}
    observed_sources = set(ni["logit_source"].astype(str))
    if observed_sources != required_sources or len(ni) != 2:
        raise RuntimeError("C4 classification non-inferiority rows are incomplete")
    required_ni_columns = {"delta_map", "ci_low", "ci_high"}
    if not required_ni_columns.issubset(ni.columns):
        raise RuntimeError(
            "C4 classification rows lack explicit delta-mAP confidence intervals"
        )
    ni_intervals = ni.loc[:, ["delta_map", "ci_low", "ci_high"]].apply(
        pd.to_numeric, errors="raise"
    )
    if not all(
        math.isfinite(float(value)) for value in ni_intervals.to_numpy().ravel()
    ):
        raise RuntimeError("C4 classification confidence intervals are not finite")
    ni_pass = bool(ni["noninferiority_pass"].map(_as_bool).all())
    classification_harmed = bool((ni_intervals["ci_high"] < 0.0).all())
    a_value = str(rows["a"].get("decision", ""))  # type: ignore[union-attr]
    b_attention = positive("b_auc") and negative("b_jaccard")
    c_routing = positive("c_auc") and negative("c_jaccard")
    c_joint_benefit = c_routing and positive("c_cam") and ni_pass
    return {
        "presence_decision": a_value,
        "presence_strong_support": a_value == "strong_support",
        "presence_partial_support": a_value == "partial_support_token_geometry_only",
        "b_attention_improves": b_attention,
        "b_cam_improves": positive("b_cam"),
        "b_cam_near_null": overlaps_zero("b_cam"),
        "b_recall_not_significantly_lower": not negative("b_recall"),
        "c_routing_improves": c_routing,
        "c_cam_improves": positive("c_cam"),
        "c_cam_harmed": negative("c_cam"),
        "c_classification_noninferiority_pass": ni_pass,
        "c_classification_harmed": classification_harmed,
        "c_background_mass_increases": positive("c_bg"),
        "c_joint_benefit_gate_passed": c_joint_benefit,
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"not a boolean value: {value!r}")


def _matched_outcomes(flags: Mapping[str, object]) -> list[str]:
    outcomes: list[str] = []
    presence = bool(flags["presence_strong_support"])
    presence_not_supported = flags["presence_decision"] == "not_supported"
    joint_benefit = bool(flags["c_joint_benefit_gate_passed"])
    if presence and joint_benefit:
        outcomes.append("Outcome 1")
    elif presence_not_supported and joint_benefit:
        outcomes.append("Outcome 3")
    if (
        bool(flags["b_attention_improves"])
        and bool(flags["b_cam_improves"])
        and bool(flags["b_recall_not_significantly_lower"])
        and joint_benefit
    ):
        outcomes.append("Outcome 4")
    if bool(flags["b_attention_improves"]) and bool(flags["b_cam_near_null"]):
        outcomes.append("Outcome 5")
    if bool(flags["c_routing_improves"]) and bool(flags["c_background_mass_increases"]):
        outcomes.append("Outcome 6")
    if bool(flags["c_cam_harmed"]) and bool(flags["c_classification_harmed"]):
        outcomes.append("Outcome 7")
    return outcomes or [
        "No exact pre-registered outcome pattern (joint benefit gate did not pass; "
        "inconclusive)"
    ]


def _combined_report(
    analyses: ValidatedAnalyses,
    source_verification: Mapping[str, object],
    rows: Mapping[str, object],
    flags: Mapping[str, object],
) -> str:
    a = rows["a"]
    assert isinstance(a, Mapping)
    ni = rows["c_noninferiority"]
    assert isinstance(ni, pd.DataFrame)
    ni_text = ", ".join(
        f"{row.logit_source}: delta mAP={float(row.delta_map):+.4f} "
        f"[{float(row.ci_low):+.4f}, {float(row.ci_high):+.4f}], "
        f"NI pass={_as_bool(row.noninferiority_pass)}"
        for row in ni.sort_values("logit_source").itertuples()
    )
    outcomes = ", ".join(_matched_outcomes(flags))
    lines = [
        "# Experiment 3 Combined Report",
        "",
        "## Audited scope",
        "",
        f"- [Fact] Validations A, B, and C each contain the complete deterministic {EXPECTED_IMAGES}-image VOC val set and use exactly {BOOTSTRAP_REPEATS:,} whole-image clustered bootstrap draws.",
        f"- [Fact] The post-run source audit passed for {int(source_verification['files_checked']):,} immutable files; no source file was missing, resized, or re-hashed differently.",
        "- [Fact] All three validations are frozen-model inference/analysis operations; no checkpoint was trained or modified.",
        "",
        "## Validation A — Presence axis",
        "",
        f"- [Fact] The production-frozen operational Validation A decision is `{a.get('decision')}`; the plan pre-registered the qualitative logic, not the later numerical CI/alignment cutoffs.",
        f"- [Statistical inference] L10–L12 positive-token-pair residual-minus-raw cosine: {_format_mapping_stat(a, 'token_pair_residual_minus_raw')}.",
        f"- [Statistical inference] L10–L12 both-axis-removed-minus-raw positive-map top-10% Jaccard: {_format_mapping_stat(a, 'map_top10_both_removed_minus_raw')}.",
        f"- [Statistical inference] L12 out-of-fold presence-projection AUROC: {_format_mapping_stat(a, 'l12_oof_projection_auroc')}.",
        "- [Mechanistic interpretation] Validation A isolates representation geometry. Even strong support would not by itself identify attention or CAM causality.",
        "",
        "## Validation B — Native CAM layer readout",
        "",
        f"- [Statistical inference] MCTformer+ B1 (L10-only) minus B0 (native last-three) raw-CAM mIoU at 0.45: {_format_ci(rows['b_cam'])}.",  # type: ignore[arg-type]
        f"- [Statistical inference] The paired B1-B0 attention target-vs-other AUROC delta is {_format_ci(rows['b_auc'])}; positive-class-pair top-10% Jaccard delta is {_format_ci(rows['b_jaccard'])}.",  # type: ignore[arg-type]
        f"- [Statistical inference] The B1-B0 binary-foreground recall delta is {_format_ci(rows['b_recall'])}.",  # type: ignore[arg-type]
        "- [Mechanistic interpretation] A routing-level change without a fixed-threshold CAM gain is compatible with downstream compensation, but does not identify the compensating component.",
        "",
        "## Validation C — Late C2C self-reroute",
        "",
        f"- [Statistical inference] MCTformer+ C4 (L10–L11 reroute) minus C0 L12 target-vs-other AUROC: {_format_ci(rows['c_auc'])}; attention top-10% Jaccard: {_format_ci(rows['c_jaccard'])}.",  # type: ignore[arg-type]
        f"- [Statistical inference] C4-C0 final raw-CAM mIoU at 0.45: {_format_ci(rows['c_cam'])}; L12 conditional background mass: {_format_ci(rows['c_bg'])}.",  # type: ignore[arg-type]
        f"- [Statistical inference] The C4-C0 classification checks are: {ni_text}. Classification harm requires both upper confidence bounds to be below zero; harm={bool(flags['c_classification_harmed'])}.",
        "- [Mechanistic interpretation] A C4-C0 effect is causal only for the exact mass-preserving frozen-model reroute operator; it does not establish off-diagonal C2C mixing as a unique natural cause.",
        "",
        "## Cross-validation decision",
        "",
        f"- [Fact] Deterministic application of the plan's decision matrix matches: {outcomes}.",
        f"- [Fact] Machine-readable decision flags are recorded in `NEXT_METHOD_DECISION.md` and `pipeline_metadata.json`: `{json.dumps(flags, sort_keys=True)}`.",
        "- [Unsupported] Experiment 3 does not validate a proposed method, justify a best-layer deployment rule, or support claims of background leakage, lazy semantic assignment, a causal shortcut, or trained-model improvement.",
        "",
        "## Artifact map",
        "",
        "- [Fact] The three independent reports are copied byte-for-byte beside this report; source hashes and copied hashes are recorded in the top-level artifact manifest.",
        "- [Fact] Rule-selected examples use a frozen maximum/lower-median/minimum per-image paired-delta rule with lexical image-ID tie-breaking; they are diagnostic illustrations, not evidence selected by hand.",
        "",
    ]
    return "\n".join(lines)


def _format_mapping_stat(decision: Mapping[str, object], key: str) -> str:
    stats = decision.get("primary_statistics")
    if not isinstance(stats, Mapping) or not isinstance(stats.get(key), Mapping):
        raise RuntimeError(f"Validation A decision lacks primary statistic {key}")
    row = stats[key]
    assert isinstance(row, Mapping)
    values = [row.get(name) for name in ("estimate", "ci_low", "ci_high")]
    if any(value is None for value in values):
        return "not evaluable"
    return f"{float(values[0]):+.4f} [{float(values[1]):+.4f}, {float(values[2]):+.4f}]"


def _decision_report(flags: Mapping[str, object]) -> str:
    outcomes = _matched_outcomes(flags)
    joint_benefit = bool(flags["c_joint_benefit_gate_passed"])
    if joint_benefit:
        decision = (
            "mechanism evidence passes the joint C-routing/CAM/classification "
            "benefit gate; "
            "any method-design phase requires a separate, explicitly approved task"
        )
    else:
        decision = (
            "the joint C-routing/CAM/classification benefit gate did not pass; the "
            "result is inconclusive and does not establish that the intervention is "
            "ineffective or harmful"
        )
    return "\n".join(
        [
            "# Next Method Decision",
            "",
            "## Decision",
            "",
            f"- [Fact] Matched pre-registered outcome pattern(s): {', '.join(outcomes)}.",
            f"- [Statistical inference] The frozen joint decision flags are `{json.dumps(flags, sort_keys=True)}`.",
            f"- [Mechanistic interpretation] Decision: {decision}.",
            "- [Unsupported] No new module, loss, token, head-selection rule, training recipe, or claimed method is introduced by this decision.",
            "",
            "## Stop boundary",
            "",
            "- [Fact] Work stops at the three completed validations, their combined synthesis, and this decision record.",
            "- [Unsupported] The present results are not evidence for retraining, a production deployment change, or a paper-level method claim without a separately designed and matched experiment.",
            "",
        ]
    )


def _verify_example_metadata(run_root: Path) -> Mapping[str, Path]:
    result: dict[str, Path] = {}
    directories = {
        "A": run_root / "presence_axis" / "examples",
        "B": run_root / "cam_layer_intervention" / "examples",
        "C": run_root / "c2c_intervention" / "examples",
    }
    for name, directory in directories.items():
        metadata_path = directory / "render_metadata.json"
        metadata = read_json(metadata_path)
        if (
            metadata.get("status") != "complete"
            or metadata.get("run_kind") != "full"
            or metadata.get("validation") != name
            or int(metadata.get("selected_examples", -1)) != 3
            or metadata.get("selection_was_manual") is not False
        ):
            raise RuntimeError(f"Validation {name} example-rendering gate failed")
        selection = directory / "selection.csv"
        if sha256_file(selection) != str(metadata.get("selection_sha256", "")):
            raise RuntimeError(f"Validation {name} selection hash mismatch")
        frame = pd.read_csv(selection)
        if len(frame) != 3 or set(frame["rank_role"]) != {
            "maximum",
            "lower_median",
            "minimum",
        }:
            raise RuntimeError(f"Validation {name} selection table is incomplete")
        if frame["image_id"].astype(str).duplicated().any():
            raise RuntimeError(f"Validation {name} selected duplicate images")
        if (
            "selection_was_manual" not in frame
            or frame["selection_was_manual"].map(_as_bool).any()
        ):
            raise RuntimeError(f"Validation {name} selection was not rule-only")
        for row in frame.to_dict(orient="records"):
            panel = _inside(
                run_root, Path(str(row["panel_path"])), label="example panel"
            )
            if panel.parent != directory.resolve():
                raise RuntimeError(
                    f"example panel is outside its validation directory: {panel}"
                )
            if (
                not panel.is_file()
                or not panel.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
                or sha256_file(panel) != str(row["panel_sha256"])
            ):
                raise RuntimeError(f"example panel hash mismatch: {panel}")
        input_hashes = metadata.get("input_hashes")
        if not isinstance(input_hashes, Mapping) or not input_hashes:
            raise RuntimeError(f"Validation {name} lacks renderer input hashes")
        for path_text, expected in input_hashes.items():
            path = Path(str(path_text)).expanduser().resolve()
            if not path.is_file() or sha256_file(path) != str(expected):
                raise RuntimeError(f"Validation {name} renderer input changed: {path}")
        result[name] = metadata_path
    return result


def _load_pipeline_controls(run_root: Path) -> tuple[Path, Path, Path, dict, dict]:
    commands = run_root / "exact_commands.sh"
    metadata_path = run_root / "pipeline_metadata.json"
    status_path = run_root / "pipeline_status.json"
    for path in (commands, metadata_path, status_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if commands.stat().st_size == 0:
        raise RuntimeError("exact_commands.sh is empty")
    metadata = read_json(metadata_path)
    status = read_json(status_path)
    if metadata.get("status") != "running" or status.get("status") != "running":
        raise RuntimeError("pipeline controls must be in the running state")
    if (
        Path(str(metadata.get("run_root", ""))).resolve() != run_root
        or Path(str(status.get("run_root", ""))).resolve() != run_root
        or not metadata.get("pipeline")
        or metadata.get("pipeline") != status.get("pipeline")
    ):
        raise RuntimeError("pipeline controls do not identify this run root")
    exact = metadata.get("exact_commands")
    evaluation = metadata.get("evaluation")
    if (
        not isinstance(exact, Mapping)
        or Path(str(exact.get("path", ""))).resolve() != commands
        or str(exact.get("sha256", "")) != sha256_file(commands)
        or not isinstance(evaluation, Mapping)
        or int(evaluation.get("images", -1)) != EXPECTED_IMAGES
        or int(evaluation.get("bootstrap_repeats", -1)) != BOOTSTRAP_REPEATS
        or evaluation.get("bootstrap_unit") != "image cluster"
    ):
        raise RuntimeError("pipeline command/evaluation metadata gate failed")
    stages = status.get("stages")
    stage_order = metadata.get("stage_order")
    if (
        not isinstance(stages, Mapping)
        or not isinstance(stage_order, list)
        or set(stage_order) != set(stages)
        or "final_reports" not in stages
        or status.get("active_stage") != "final_reports"
    ):
        raise RuntimeError("pipeline stage ledger is incomplete")
    for name, record in stages.items():
        if not isinstance(record, Mapping):
            raise TypeError(f"pipeline stage {name} is not an object")
        expected = "running" if name == "final_reports" else "complete"
        if record.get("status") != expected:
            raise RuntimeError(
                f"pipeline stage {name} must be {expected} before finalization"
            )
    return commands, metadata_path, status_path, metadata, status


def _close_final_report_stage(payload: dict[str, object], completed_at: str) -> None:
    """Close an orchestrator-owned final stage without replacing its history."""

    stages = payload.get("stages")
    if not isinstance(stages, dict) or "final_reports" not in stages:
        return
    record = stages["final_reports"]
    if not isinstance(record, dict):
        raise TypeError("pipeline stages.final_reports must be an object")
    if record.get("status") == "failed":
        raise RuntimeError("pipeline stages.final_reports is already failed")
    record["status"] = "complete"
    record["returncode"] = 0
    record["completed_at"] = completed_at


def _atomic_reports(
    run_root: Path,
    analyses: ValidatedAnalyses,
    source_verification: Mapping[str, object],
    rows: Mapping[str, object],
    flags: Mapping[str, object],
) -> Path:
    destination = run_root / "reports"
    if destination.exists():
        raise FileExistsError(destination)
    temporary = Path(
        tempfile.mkdtemp(prefix=".reports.tmp-", dir=str(run_root))
    ).resolve()
    try:
        for name, filename in REPORT_FILENAMES.items():
            shutil.copyfile(analyses.reports[name], temporary / filename)
        (temporary / "EXPERIMENT3_COMBINED_REPORT.md").write_text(
            _combined_report(analyses, source_verification, rows, flags),
            encoding="utf-8",
        )
        (temporary / "NEXT_METHOD_DECISION.md").write_text(
            _decision_report(flags), encoding="utf-8"
        )
        for path in temporary.glob("*.md"):
            _require_report_labels(path)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _artifact_paths(
    analyses: ValidatedAnalyses,
    examples: Mapping[str, Path],
    reports: Path,
    source_verification_path: Path,
    controls: Sequence[Path],
) -> Iterable[tuple[str, Path]]:
    seen: set[Path] = set()
    groups = [
        *(
            ("validation_" + name.lower(), root)
            for name, root in analyses.roots.items()
        ),
        *(("examples_" + name.lower(), path.parent) for name, path in examples.items()),
        ("reports", reports),
        ("source_verification", source_verification_path.parent),
    ]
    for group, root in groups:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield group, resolved
    for path in controls:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield "pipeline_control", resolved


def _write_artifact_manifest(
    run_root: Path,
    analyses: ValidatedAnalyses,
    examples: Mapping[str, Path],
    reports: Path,
    source_verification_path: Path,
    controls: Sequence[Path],
) -> tuple[Path, str, int]:
    destination = run_root / "artifact_manifest.csv"
    digest_path = run_root / "artifact_manifest.sha256"
    if destination.exists() or digest_path.exists():
        raise FileExistsError(destination if destination.exists() else digest_path)
    rows = []
    for group, path in _artifact_paths(
        analyses, examples, reports, source_verification_path, controls
    ):
        rows.append(
            {
                "artifact_group": group,
                "relative_path": str(path.relative_to(run_root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    with destination.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("artifact_group", "relative_path", "size_bytes", "sha256"),
        )
        writer.writeheader()
        writer.writerows(rows)
    digest = sha256_file(destination)
    digest_path.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return destination, digest, len(rows)


def _restore_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".finalizer-rollback.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _assert_analysis_controls_unchanged(analyses: ValidatedAnalyses) -> None:
    for path, digest in analyses.control_hashes.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"analysis input changed during finalization: {path}")


def finalize(
    *,
    run_root: Path,
    validation_a_root: Path,
    validation_b_root: Path,
    validation_c_root: Path,
    source_verification: Path,
) -> Mapping[str, object]:
    require_tgca_repro()
    analyses = validate_analysis_roots(
        run_root, validation_a_root, validation_b_root, validation_c_root
    )
    source_verification_path = _inside(
        analyses.run_root, source_verification, label="source verification"
    )
    source = validate_source_verification(
        source_verification_path, run_root=analyses.run_root
    )
    examples = _verify_example_metadata(analyses.run_root)
    commands, metadata_path, status_path, pipeline_metadata, pipeline_status = (
        _load_pipeline_controls(analyses.run_root)
    )
    reports_target = analyses.run_root / "reports"
    manifest_target = analyses.run_root / "artifact_manifest.csv"
    manifest_digest_target = analyses.run_root / "artifact_manifest.sha256"
    for path in (reports_target, manifest_target, manifest_digest_target):
        if path.exists():
            raise FileExistsError(path)
    commands_hash = sha256_file(commands)
    metadata_before = metadata_path.read_bytes()
    status_before = status_path.read_bytes()
    rows = _primary_rows(analyses)
    flags = _decision_flags(rows)
    reports: Optional[Path] = None
    try:
        reports = _atomic_reports(analyses.run_root, analyses, source, rows, flags)
        report_records = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(reports.glob("*.md"))
        }
        finalization = {
            "status": "complete",
            "completed_at": timestamp(),
            "full_images": EXPECTED_IMAGES,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_unit": "image cluster",
            "source_immutability_passed": True,
            "source_files_checked": int(source["files_checked"]),
            "analysis_roots": {
                name: str(path) for name, path in analyses.roots.items()
            },
            "example_metadata": {name: str(path) for name, path in examples.items()},
            "reports": report_records,
            "decision_flags": dict(flags),
            "matched_outcomes": _matched_outcomes(flags),
            "artifact_manifest": {
                "path": "artifact_manifest.csv",
                "sha256_sidecar": "artifact_manifest.sha256",
                "self_excluded": True,
            },
        }
        new_metadata = copy.deepcopy(pipeline_metadata)
        new_status = copy.deepcopy(pipeline_status)
        new_metadata["finalization"] = finalization
        new_status["finalization"] = finalization
        new_metadata["status"] = "complete"
        new_status["status"] = "complete"
        new_status["active_stage"] = None
        _close_final_report_stage(new_metadata, str(finalization["completed_at"]))
        _close_final_report_stage(new_status, str(finalization["completed_at"]))
        json_dump(metadata_path, new_metadata)
        json_dump(status_path, new_status)
        if sha256_file(commands) != commands_hash:
            raise RuntimeError("exact_commands.sh changed during finalization")
        _assert_analysis_controls_unchanged(analyses)
        manifest_path, manifest_hash, manifest_rows = _write_artifact_manifest(
            analyses.run_root,
            analyses,
            examples,
            reports,
            source_verification_path,
            (commands, metadata_path, status_path),
        )
    except Exception:
        _restore_bytes(metadata_path, metadata_before)
        _restore_bytes(status_path, status_before)
        if reports is not None and reports == reports_target:
            shutil.rmtree(reports, ignore_errors=True)
        for path in (manifest_target, manifest_digest_target):
            if path.is_file():
                path.unlink()
        raise
    return {
        "status": "complete",
        "reports_dir": str(reports),
        "artifact_manifest": str(manifest_path),
        "artifact_manifest_sha256": manifest_hash,
        "artifact_manifest_rows": manifest_rows,
        "matched_outcomes": _matched_outcomes(flags),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--validation-a-root", type=Path, required=True)
    parser.add_argument("--validation-b-root", type=Path, required=True)
    parser.add_argument("--validation-c-root", type=Path, required=True)
    parser.add_argument("--source-verification", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = finalize(
        run_root=args.run_root,
        validation_a_root=args.validation_a_root,
        validation_b_root=args.validation_b_root,
        validation_c_root=args.validation_c_root,
        source_verification=args.source_verification,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
