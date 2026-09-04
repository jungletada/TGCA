"""Atomic, compressed, round-trip-verified Parquet output for Experiment 2."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = "experiment2-canonical-v2-patch-norm-joint"

TABLE_FILENAMES: Mapping[str, str] = {
    "per_image_class_layer_signal": "per_image_class_layer_signal.parquet",
    "per_image_class_cam_stage": "per_image_class_cam_stage.parquet",
    "per_image_class_stage_transition": "per_image_class_stage_transition.parquet",
    "per_multilabel_class_pair_layer_signal": "per_multilabel_class_pair_layer_signal.parquet",
    "per_shared_patch_ownership": "per_shared_patch_ownership.parquet",
    "per_class_token_pair_layer": "per_class_token_pair_layer.parquet",
    "per_image_classification": "per_image_classification.parquet",
    "per_image_cam_confusion": "per_image_cam_confusion.parquet",
    "source_index": "source_index.parquet",
}

TABLE_REQUIRED_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "per_image_class_layer_signal": (
        "model",
        "image_id",
        "class_id",
        "layer",
        "signal",
        "rho",
        "num_target",
        "num_other_fg",
        "num_bg",
        "num_mixed",
        "num_void",
        "top1_region",
        "auc_target_bg",
        "ap_target_bg",
        "auc_target_other",
        "ap_target_other",
        "conditional_bg_mass",
        "classification_status",
    ),
    "per_image_class_cam_stage": (
        "model",
        "image_id",
        "class_id",
        "stage",
        "rho",
        "num_target",
        "num_other_fg",
        "num_bg",
        "num_mixed",
        "num_void",
        "top1_region",
        "auc_target_bg",
        "ap_target_bg",
        "conditional_bg_mass",
        "classification_status",
    ),
    "per_image_class_stage_transition": (
        "model",
        "image_id",
        "class_id",
        "transition",
        "layer",
        "rho",
        "topk_ratio",
        "spearman",
        "topk_jaccard",
        "topk_overlap_coefficient",
        "introduced_size",
        "removed_size",
    ),
    "per_multilabel_class_pair_layer_signal": (
        "model",
        "image_id",
        "class_a",
        "class_b",
        "layer",
        "signal",
        "topk_ratio",
        "spearman",
        "topk_jaccard",
        "topk_overlap_coefficient",
    ),
    "per_shared_patch_ownership": (
        "model",
        "image_id",
        "class_a",
        "class_b",
        "layer_or_stage",
        "signal",
        "rho",
        "topk_ratio",
        "shared_set_size",
        "shared_target_a_fraction",
        "shared_target_b_fraction",
        "shared_other_fg_fraction",
        "shared_bg_fraction",
        "shared_mixed_void_fraction",
        "new_shared_from_previous_layer",
    ),
    "per_class_token_pair_layer": (
        "model",
        "image_id",
        "class_a",
        "class_b",
        "layer",
        "class_token_cosine",
        "feature_post_spearman",
    ),
    "per_image_classification": (
        "model",
        "image_id",
        "class_id",
        "target",
        "class_logit",
        "patch_class_logit",
        "num_positive_classes",
        "label_stratum",
    ),
    "per_image_cam_confusion": (
        "model",
        "image_id",
        "gt_class_id",
        "pred_class_id",
        "pixel_count",
        "background_threshold",
        "num_positive_classes",
        "label_stratum",
    ),
    "source_index": (
        "model",
        "image_id",
        "signal_root",
        "manifest_path",
        "artifact_path",
        "artifact_sha256",
        "manifest_artifact_sha256",
        "hash_verified",
        "source_unchanged",
    ),
}


def _logical_arrow_hash(table: pa.Table) -> str:
    """Hash logical Arrow values, independent of Parquet's physical encoding."""

    normalized = table.combine_chunks().replace_schema_metadata(None)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, normalized.schema) as writer:
        writer.write_table(normalized)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_with_metadata(schema: pa.Schema, table_name: str) -> pa.Schema:
    metadata = dict(schema.metadata or {})
    metadata.update(
        {
            b"experiment": b"Experiment 2 Semantic Ownership",
            b"schema_version": SCHEMA_VERSION.encode("utf-8"),
            b"table_name": table_name.encode("utf-8"),
            b"compression": b"zstd",
        }
    )
    return schema.with_metadata(metadata)


def _validate_rows(table_name: str, rows: Sequence[Mapping[str, object]]) -> None:
    if table_name not in TABLE_FILENAMES:
        raise KeyError(f"unknown canonical table {table_name!r}")
    required = set(TABLE_REQUIRED_COLUMNS[table_name])
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise ValueError(
                f"{table_name} row {index} misses required columns {sorted(missing)}"
            )


class StreamingCanonicalWriter:
    """Write bounded row chunks and verify every Parquet row group on close."""

    def __init__(self, output_dir: Path, flush_rows: int = 10_000) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        if self.output_dir.exists():
            raise FileExistsError(self.output_dir)
        if int(flush_rows) < 1:
            raise ValueError("flush_rows must be positive")
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.flush_rows = int(flush_rows)
        self._buffers: dict[str, list[Mapping[str, object]]] = {
            name: [] for name in TABLE_FILENAMES
        }
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._schemas: dict[str, pa.Schema] = {}
        self._row_group_hashes: dict[str, list[str]] = {
            name: [] for name in TABLE_FILENAMES
        }
        self._row_counts: dict[str, int] = {name: 0 for name in TABLE_FILENAMES}
        self._closed = False

    def append(self, table_name: str, rows: Iterable[Mapping[str, object]]) -> None:
        if self._closed:
            raise RuntimeError("canonical writer is already closed")
        materialized = list(rows)
        if not materialized:
            return
        _validate_rows(table_name, materialized)
        if table_name in self._schemas:
            expected = set(self._schemas[table_name].names)
            for index, row in enumerate(materialized):
                unexpected = set(row).difference(expected)
                if unexpected:
                    raise ValueError(
                        f"{table_name} row {index} introduces columns after schema "
                        f"creation: {sorted(unexpected)}"
                    )
        self._buffers[table_name].extend(materialized)
        if len(self._buffers[table_name]) >= self.flush_rows:
            self._flush(table_name)

    def _flush(self, table_name: str) -> None:
        rows = self._buffers[table_name]
        if not rows:
            return
        if table_name not in self._schemas:
            table = pa.Table.from_pylist(rows)
            schema = _schema_with_metadata(table.schema, table_name)
            table = table.cast(schema)
            path = self.output_dir / (TABLE_FILENAMES[table_name] + ".tmp")
            self._schemas[table_name] = schema
            self._writers[table_name] = pq.ParquetWriter(
                path,
                schema,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
        else:
            schema = self._schemas[table_name]
            expected = set(schema.names)
            for index, row in enumerate(rows):
                missing = expected.difference(row)
                if missing:
                    raise ValueError(
                        f"{table_name} row {index} misses established columns "
                        f"{sorted(missing)}"
                    )
            table = pa.Table.from_pylist(rows, schema=schema)
        self._writers[table_name].write_table(table, row_group_size=len(rows))
        self._row_group_hashes[table_name].append(_logical_arrow_hash(table))
        self._row_counts[table_name] += len(rows)
        rows.clear()

    def _write_empty(self, table_name: str) -> None:
        fields = [
            pa.field(column, pa.string())
            for column in TABLE_REQUIRED_COLUMNS[table_name]
        ]
        schema = _schema_with_metadata(pa.schema(fields), table_name)
        path = self.output_dir / (TABLE_FILENAMES[table_name] + ".tmp")
        writer = pq.ParquetWriter(path, schema, compression="zstd")
        writer.close()
        self._schemas[table_name] = schema

    def close(self) -> dict[str, dict[str, object]]:
        if self._closed:
            raise RuntimeError("canonical writer is already closed")
        for table_name in TABLE_FILENAMES:
            self._flush(table_name)
        for writer in self._writers.values():
            writer.close()
        for table_name in TABLE_FILENAMES:
            if table_name not in self._schemas:
                self._write_empty(table_name)

        summaries: dict[str, dict[str, object]] = {}
        for table_name, filename in TABLE_FILENAMES.items():
            temporary = self.output_dir / (filename + ".tmp")
            parquet = pq.ParquetFile(temporary)
            if parquet.metadata.num_rows != self._row_counts[table_name]:
                raise AssertionError(f"row-count round trip failed for {table_name}")
            expected_hashes = self._row_group_hashes[table_name]
            if parquet.num_row_groups != len(expected_hashes):
                raise AssertionError(f"row-group count changed for {table_name}")
            for index, expected_hash in enumerate(expected_hashes):
                restored = parquet.read_row_group(index)
                if _logical_arrow_hash(restored) != expected_hash:
                    raise AssertionError(
                        f"logical Parquet round trip failed for {table_name} row group {index}"
                    )
            compressions = sorted(
                {
                    parquet.metadata.row_group(row_group).column(column).compression
                    for row_group in range(parquet.num_row_groups)
                    for column in range(parquet.metadata.num_columns)
                }
            )
            if compressions and compressions != ["ZSTD"]:
                raise AssertionError(
                    f"unexpected compression for {table_name}: {compressions}"
                )
            final = self.output_dir / filename
            summaries[table_name] = {
                "path": str(final),
                "rows": int(parquet.metadata.num_rows),
                "columns": list(parquet.schema_arrow.names),
                "row_groups": int(parquet.num_row_groups),
                "compression": compressions or ["ZSTD"],
                "sha256": _sha256_file(temporary),
                "roundtrip_verified": True,
            }
        # Only expose final filenames after every table has passed logical
        # round-trip, row-count, schema, and compression checks.
        for table_name, filename in TABLE_FILENAMES.items():
            temporary = self.output_dir / (filename + ".tmp")
            temporary.replace(self.output_dir / filename)
        self._closed = True
        return summaries

    def abort(self) -> None:
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception:
                pass
        for filename in TABLE_FILENAMES.values():
            temporary = self.output_dir / (filename + ".tmp")
            if temporary.is_file():
                temporary.unlink()
        self._closed = True


def write_parquet_roundtrip(
    frame: pd.DataFrame,
    path: Path,
    *,
    table_name: str,
) -> dict[str, object]:
    """Small-table convenience wrapper used by focused tests and utilities."""

    if table_name not in TABLE_FILENAMES:
        raise KeyError(f"unknown canonical table {table_name!r}")
    missing = set(TABLE_REQUIRED_COLUMNS[table_name]).difference(frame.columns)
    if missing:
        raise ValueError(f"{table_name} frame misses columns {sorted(missing)}")
    requested = Path(path).expanduser().resolve()
    if requested.exists():
        raise FileExistsError(requested)
    requested.parent.mkdir(parents=True, exist_ok=True)
    rows = frame.to_dict(orient="records")
    _validate_rows(table_name, rows)
    table = pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)
    table = table.cast(_schema_with_metadata(table.schema, table_name))
    expected_hash = _logical_arrow_hash(table)
    temporary = requested.with_suffix(requested.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    restored = pq.read_table(temporary)
    if _logical_arrow_hash(restored) != expected_hash:
        raise AssertionError(f"logical Parquet round trip failed for {table_name}")
    temporary.replace(requested)
    parquet = pq.ParquetFile(requested)
    return {
        "path": str(requested),
        "rows": int(parquet.metadata.num_rows),
        "columns": list(parquet.schema_arrow.names),
        "row_groups": int(parquet.num_row_groups),
        "compression": ["ZSTD"],
        "sha256": _sha256_file(requested),
        "roundtrip_verified": True,
    }
