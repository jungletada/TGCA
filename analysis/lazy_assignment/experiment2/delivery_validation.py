"""Fail-closed validation for Experiment 2 plots and selected examples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON metadata must be an object: {path}")
    return value


def _same_path(left: object, right: Path) -> bool:
    return Path(str(left)).expanduser().resolve() == right.resolve()


def _verify_plots(
    plots_dir: Path,
    tables_dir: Path,
    required_plot_files: Sequence[str],
    required_table_files: Sequence[str],
) -> dict[str, object]:
    metadata_path = plots_dir / "plot_metadata.json"
    metadata = _read_json(metadata_path)
    if (
        metadata.get("status") != "complete"
        or metadata.get("invented_values") is not False
    ):
        raise RuntimeError("plot metadata is not a completed evidence-only run")
    if not _same_path(metadata.get("tables_dir", ""), tables_dir):
        raise RuntimeError("plots were not generated from the reported analysis tables")
    if metadata.get("missing_or_empty_tables") not in ({}, None):
        raise RuntimeError(
            f"full report refuses plots with missing inputs: "
            f"{metadata.get('missing_or_empty_tables')}"
        )
    input_hashes = metadata.get("input_table_sha256")
    if not isinstance(input_hashes, dict) or set(input_hashes) != set(
        required_table_files
    ):
        raise RuntimeError("plot metadata lacks the exact input-table hash inventory")
    for filename in required_table_files:
        path = tables_dir / filename
        if not path.is_file() or input_hashes.get(filename) != _sha256(path):
            raise RuntimeError(f"plot input-table hash mismatch: {path}")
    expected_names = tuple(required_plot_files)
    recorded_paths = tuple(Path(str(value)).name for value in metadata.get("plots", []))
    hashes = metadata.get("plot_sha256")
    if recorded_paths != expected_names or not isinstance(hashes, dict):
        raise RuntimeError("plot inventory does not match the pre-specified figure set")
    for filename in expected_names:
        path = plots_dir / filename
        if not path.is_file() or path.stat().st_size <= 100:
            raise RuntimeError(f"missing or empty diagnostic plot: {path}")
        if hashes.get(filename) != _sha256(path):
            raise RuntimeError(f"diagnostic plot hash mismatch: {path}")
    return {
        "metadata": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "plot_count": len(expected_names),
    }


def _verify_selection(
    examples_dir: Path,
    canonical_metadata_path: Path,
    required_categories: Sequence[str],
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    metadata_path = examples_dir / "selection_metadata.json"
    selection_path = examples_dir / "example_selection.csv"
    metadata = _read_json(metadata_path)
    if (
        metadata.get("selection_is_deterministic") is not True
        or metadata.get("manual_cherry_picking") is not False
        or not _same_path(
            metadata.get("canonical_dir", ""), canonical_metadata_path.parent
        )
        or not selection_path.is_file()
        or metadata.get("selection_sha256") != _sha256(selection_path)
    ):
        raise RuntimeError("example selection provenance is incomplete or inconsistent")
    selection = pd.read_csv(selection_path, dtype=str, keep_default_na=False)
    if "category" not in selection or "case_id" not in selection:
        raise RuntimeError("example selection lacks case/category identity")
    if selection["case_id"].duplicated().any():
        raise RuntimeError("example selection contains duplicate case IDs")
    required_context = {
        "class_id",
        "companion_class_id",
        "class_a",
        "class_b",
        "positive_class_ids_json",
        "num_positive_classes",
    }
    if not required_context.issubset(selection.columns):
        raise RuntimeError("example selection lacks positive-class panel provenance")
    if int(metadata.get("total_rows", -1)) != len(selection):
        raise RuntimeError("example selection row count disagrees with metadata")
    expected_categories = tuple(required_categories)
    if tuple(metadata.get("new_categories", [])) != expected_categories:
        raise RuntimeError("example selection category registry changed")
    counts = selection["category"].value_counts().to_dict()
    recorded_counts = {
        str(key): int(value)
        for key, value in metadata.get("category_counts", {}).items()
    }
    if counts != recorded_counts:
        raise RuntimeError("example category counts disagree with selection CSV")
    missing = tuple(name for name in expected_categories if name not in counts)
    if tuple(metadata.get("missing_new_categories", [])) != missing:
        raise RuntimeError("missing example categories are not explicitly recorded")
    fixed = selection["category"].str.startswith("experiment1_fixed::").sum()
    if fixed != 70 or int(metadata.get("experiment1_fixed_rows_retained", -1)) != 70:
        raise RuntimeError("final report requires exactly 70 fixed Experiment 1 cases")
    new_selection = selection[
        ~selection["category"].str.startswith("experiment1_fixed::")
    ]
    for index, row in new_selection.iterrows():
        try:
            positive = json.loads(row["positive_class_ids_json"])
            count = int(row["num_positive_classes"])
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"invalid positive-class panel provenance in selection row {index}"
            ) from error
        if (
            not isinstance(positive, list)
            or count != len(positive)
            or len(set(positive)) != len(positive)
        ):
            raise RuntimeError(
                f"inconsistent positive-class panel provenance in selection row {index}"
            )
    return (
        metadata,
        selection,
        {
            "metadata": str(metadata_path),
            "metadata_sha256": _sha256(metadata_path),
            "selection": str(selection_path),
            "selection_sha256": _sha256(selection_path),
            "rows": len(selection),
            "missing_new_categories": list(missing),
        },
    )


def _verify_render(
    render_dir: Path,
    selection: pd.DataFrame,
    selection_path: Path,
    canonical_metadata_path: Path,
    source_metadata_path: Path,
) -> dict[str, object]:
    metadata_path = render_dir / "render_metadata.json"
    manifest_path = render_dir / "render_manifest.csv"
    metadata = _read_json(metadata_path)
    required_true = (
        "source_npz_manifest_hashes_verified",
        "source_npz_unchanged",
        "canonical_is_full_set",
    )
    if (
        metadata.get("status") != "complete"
        or not all(metadata.get(key) is True for key in required_true)
        or metadata.get("missing_data_placeholders_generated") is not False
        or metadata.get("model_execution") is not False
        or metadata.get("model_loaded") is not False
        or not _same_path(metadata.get("output_dir", ""), render_dir)
        or not _same_path(metadata.get("example_selection", ""), selection_path)
        or metadata.get("example_selection_sha256") != _sha256(selection_path)
        or not _same_path(
            metadata.get("canonical_metadata", ""), canonical_metadata_path
        )
        or metadata.get("canonical_metadata_sha256") != _sha256(canonical_metadata_path)
        or not _same_path(metadata.get("source_metadata", ""), source_metadata_path)
        or metadata.get("source_metadata_sha256") != _sha256(source_metadata_path)
    ):
        raise RuntimeError("render metadata is incomplete, non-full, or mislinked")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if (
        int(metadata.get("manifest_rows", -1)) != len(manifest)
        or len(manifest) != len(selection)
        or "case_id" not in manifest
        or manifest["case_id"].tolist() != selection["case_id"].tolist()
    ):
        raise RuntimeError("render manifest does not exactly match selected cases")
    fixed_mask = manifest["render_status"].eq("linked_existing_not_redrawn")
    rendered_mask = manifest["render_status"].eq("rendered_from_existing_npz")
    if not (fixed_mask | rendered_mask).all():
        raise RuntimeError("render manifest contains an unsupported render status")
    if (
        int(fixed_mask.sum()) != int(metadata.get("fixed_experiment1_cases_linked", -1))
        or int(rendered_mask.sum()) != int(metadata.get("rendered_panel_files", -1))
        or int(fixed_mask.sum()) != 70
    ):
        raise RuntimeError("rendered/fixed example counts disagree with metadata")
    selection_by_case = selection.set_index("case_id", verify_integrity=True)
    for _, row in manifest[rendered_mask].iterrows():
        selected_row = selection_by_case.loc[row["case_id"]]
        try:
            positive = json.loads(selected_row["positive_class_ids_json"])
            displayed = json.loads(row["class_ids_json"])
            recorded_rows = int(row["num_class_rows"])
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"invalid rendered class provenance for {row['case_id']}"
            ) from error
        if (
            not isinstance(displayed, list)
            or recorded_rows != len(displayed)
            or any(value not in positive for value in displayed)
            or (len(positive) >= 2 and len(displayed) < 2)
        ):
            raise RuntimeError(
                f"multi-label panel class contract failed for {row['case_id']}"
            )
    panel_paths: list[Path] = []
    for _, row in manifest[rendered_mask].iterrows():
        path = Path(row["panel_path"]).expanduser().resolve()
        if not path.is_file() or _sha256(path) != row["panel_sha256"]:
            raise RuntimeError(f"rendered example panel hash mismatch: {path}")
        panel_paths.append(path)
    if len(set(panel_paths)) != len(panel_paths):
        raise RuntimeError("multiple selected examples unexpectedly share one panel")
    for _, row in manifest[fixed_mask].iterrows():
        for prefix in ("experiment1_raw", "experiment1_minmax"):
            path = Path(row[f"{prefix}_figure"]).expanduser().resolve()
            if not path.is_file() or _sha256(path) != row[f"{prefix}_sha256"]:
                raise RuntimeError(f"linked Experiment 1 figure hash mismatch: {path}")
    actual_output_files = sum(path.is_file() for path in render_dir.rglob("*"))
    if actual_output_files != int(metadata.get("output_file_count", -1)):
        raise RuntimeError("render directory file count disagrees with metadata")
    return {
        "metadata": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "rows": len(manifest),
        "rendered_panels": len(panel_paths),
        "fixed_links": int(fixed_mask.sum()),
    }


def verify_visual_deliverables(
    *,
    tables_dir: Path,
    plots_dir: Path,
    examples_dir: Path,
    render_dir: Path,
    canonical_metadata_path: Path,
    source_metadata_path: Path,
    required_plot_files: Sequence[str],
    required_plot_input_files: Sequence[str],
    required_categories: Sequence[str],
) -> dict[str, object]:
    """Verify full-set plot, selection, and render products without mutation."""

    tables_dir = tables_dir.resolve()
    plots_dir = plots_dir.resolve()
    examples_dir = examples_dir.resolve()
    render_dir = render_dir.resolve()
    selection_metadata, selection, selection_summary = _verify_selection(
        examples_dir, canonical_metadata_path, required_categories
    )
    del selection_metadata
    return {
        "plots": _verify_plots(
            plots_dir,
            tables_dir,
            required_plot_files,
            required_plot_input_files,
        ),
        "selection": selection_summary,
        "render": _verify_render(
            render_dir,
            selection,
            examples_dir / "example_selection.csv",
            canonical_metadata_path,
            source_metadata_path,
        ),
    }
