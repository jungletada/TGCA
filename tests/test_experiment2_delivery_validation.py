from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.lazy_assignment.experiment2.delivery_validation import (
    verify_visual_deliverables,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    tables = tmp_path / "tables"
    plots = tmp_path / "plots"
    examples = tmp_path / "examples"
    rendered = tmp_path / "rendered"
    canonical = tmp_path / "canonical"
    audit = tmp_path / "audit"
    for path in (tables, plots, examples, rendered, canonical, audit):
        path.mkdir()
    canonical_metadata = canonical / "canonical_metadata.json"
    source_metadata = audit / "source_metadata.json"
    _write_json(canonical_metadata, {"status": "complete"})
    _write_json(source_metadata, {"integrity_passed": True})
    plot_input = tables / "metric.csv"
    plot_input.write_text("metric,value\nx,1\n", encoding="utf-8")

    plot = plots / "diagnostic.png"
    plot.write_bytes(b"valid plot" * 20)
    _write_json(
        plots / "plot_metadata.json",
        {
            "status": "complete",
            "invented_values": False,
            "tables_dir": str(tables.resolve()),
            "plots": [str(plot.resolve())],
            "plot_sha256": {plot.name: _sha(plot)},
            "input_table_sha256": {plot_input.name: _sha(plot_input)},
            "missing_or_empty_tables": {},
        },
    )

    selection_rows = [
        {
            "case_id": "new-1",
            "category": "new_category",
            "class_id": 2,
            "companion_class_id": 18,
            "class_a": "",
            "class_b": "",
            "positive_class_ids_json": "[2, 18]",
            "num_positive_classes": 2,
        },
        *[
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
        ],
    ]
    selection = examples / "example_selection.csv"
    pd.DataFrame(selection_rows).to_csv(selection, index=False)
    _write_json(
        examples / "selection_metadata.json",
        {
            "selection_is_deterministic": True,
            "manual_cherry_picking": False,
            "canonical_dir": str(canonical.resolve()),
            "selection_path": str(selection.resolve()),
            "selection_sha256": _sha(selection),
            "total_rows": len(selection_rows),
            "new_categories": ["new_category"],
            "missing_new_categories": [],
            "category_counts": {
                "new_category": 1,
                "experiment1_fixed::fixture": 70,
            },
            "experiment1_fixed_rows_retained": 70,
        },
    )

    panel = rendered / "panel.png"
    panel.write_bytes(b"panel bytes")
    raw = tmp_path / "raw.png"
    minmax = tmp_path / "minmax.png"
    raw.write_bytes(b"raw")
    minmax.write_bytes(b"minmax")
    manifest_rows = [
        {
            "case_id": "new-1",
            "render_status": "rendered_from_existing_npz",
            "class_ids_json": "[2, 18]",
            "num_class_rows": 2,
            "panel_path": str(panel.resolve()),
            "panel_sha256": _sha(panel),
            "experiment1_raw_figure": "",
            "experiment1_raw_sha256": "",
            "experiment1_minmax_figure": "",
            "experiment1_minmax_sha256": "",
        }
    ]
    manifest_rows.extend(
        {
            "case_id": f"fixed-{index:02d}",
            "render_status": "linked_existing_not_redrawn",
            "class_ids_json": "[]",
            "num_class_rows": 0,
            "panel_path": "",
            "panel_sha256": "",
            "experiment1_raw_figure": str(raw.resolve()),
            "experiment1_raw_sha256": _sha(raw),
            "experiment1_minmax_figure": str(minmax.resolve()),
            "experiment1_minmax_sha256": _sha(minmax),
        }
        for index in range(70)
    )
    manifest = rendered / "render_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    _write_json(
        rendered / "render_metadata.json",
        {
            "status": "complete",
            "example_selection": str(selection.resolve()),
            "example_selection_sha256": _sha(selection),
            "canonical_metadata": str(canonical_metadata.resolve()),
            "canonical_metadata_sha256": _sha(canonical_metadata),
            "source_metadata": str(source_metadata.resolve()),
            "source_metadata_sha256": _sha(source_metadata),
            "output_dir": str(rendered.resolve()),
            "manifest_rows": len(manifest_rows),
            "rendered_panel_files": 1,
            "fixed_experiment1_cases_linked": 70,
            "output_file_count": 3,
            "source_npz_manifest_hashes_verified": True,
            "source_npz_unchanged": True,
            "missing_data_placeholders_generated": False,
            "model_execution": False,
            "model_loaded": False,
            "canonical_is_full_set": True,
        },
    )
    return {
        "tables": tables,
        "plots": plots,
        "examples": examples,
        "rendered": rendered,
        "canonical": canonical_metadata,
        "source": source_metadata,
        "plot": plot,
    }


def _verify(paths: dict[str, Path]) -> dict[str, object]:
    return verify_visual_deliverables(
        tables_dir=paths["tables"],
        plots_dir=paths["plots"],
        examples_dir=paths["examples"],
        render_dir=paths["rendered"],
        canonical_metadata_path=paths["canonical"],
        source_metadata_path=paths["source"],
        required_plot_files=("diagnostic.png",),
        required_plot_input_files=("metric.csv",),
        required_categories=("new_category",),
    )


def test_visual_deliverables_are_cryptographically_linked(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = _verify(paths)
    assert result["plots"]["plot_count"] == 1
    assert result["selection"]["rows"] == 71
    assert result["render"]["fixed_links"] == 70


def test_visual_deliverable_tampering_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["plot"].write_bytes(b"tampered" * 20)
    with pytest.raises(RuntimeError, match="plot hash mismatch"):
        _verify(paths)


def test_visual_delivery_refuses_single_class_multilabel_panel(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    manifest_path = paths["rendered"] / "render_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    manifest.loc[manifest["case_id"] == "new-1", "class_ids_json"] = "[2]"
    manifest.loc[manifest["case_id"] == "new-1", "num_class_rows"] = "1"
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(RuntimeError, match="multi-label panel class contract"):
        _verify(paths)
