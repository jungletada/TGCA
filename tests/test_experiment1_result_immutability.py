"""Tests for immutable source-result verification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.lazy_assignment.experiment1_analysis_common import sha256_file
from analysis.lazy_assignment.verify_experiment1_immutability import verify_manifest


def _manifest(path: Path) -> pd.DataFrame:
    stat = path.stat()
    return pd.DataFrame(
        [
            {
                "model": "fixture",
                "result_root": str(path.parent),
                "relative_path": path.name,
                "absolute_path": str(path),
                "kind": "score_npz",
                "size_bytes": stat.st_size,
                "mtime_ns_before": stat.st_mtime_ns,
                "sha256_before": sha256_file(path),
            }
        ]
    )


def test_immutability_verifier_accepts_unchanged_file(tmp_path: Path) -> None:
    source = tmp_path / "sample.npz"
    source.write_bytes(b"immutable fixture")

    after, changes = verify_manifest(_manifest(source))

    assert not changes
    assert bool(after.iloc[0]["content_unchanged"])
    assert bool(after.iloc[0]["mtime_unchanged"])


def test_immutability_verifier_detects_content_change(tmp_path: Path) -> None:
    source = tmp_path / "sample.npz"
    source.write_bytes(b"before")
    before = _manifest(source)
    source.write_bytes(b"after")

    _, changes = verify_manifest(before)

    assert len(changes) == 1
    assert changes[0]["issue"] == "source_content_size_or_mtime_changed"

