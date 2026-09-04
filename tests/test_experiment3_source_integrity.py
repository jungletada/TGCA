from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from analysis.lazy_assignment.experiment3.common import (
    enforce_production_source,
    sha256_file,
)
from analysis.lazy_assignment.experiment3.runtime import (
    reload_npz_checked,
    save_npz_atomic,
)
from analysis.lazy_assignment.experiment3.verify_source_immutability import verify


def _before_manifest(path: Path, sources: list[Path]) -> Path:
    manifest = path / "before.csv"
    with manifest.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("absolute_path", "size_bytes", "sha256")
        )
        writer.writeheader()
        for source in sources:
            writer.writerow(
                {
                    "absolute_path": source,
                    "size_bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                }
            )
    return manifest


def test_source_manifest_verifier_passes_and_detects_later_mutation(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"immutable-a")
    second.write_bytes(b"immutable-b")
    before = _before_manifest(tmp_path, [first, second])

    passed = verify(before, tmp_path / "pass")

    assert passed["integrity_passed"] is True
    assert passed["files_checked"] == 2
    first.write_bytes(b"immutable-c")
    failed = verify(before, tmp_path / "fail")
    assert failed["integrity_passed"] is False
    assert failed["sha256_changed_files"] == 1


def test_derived_npz_reload_checks_hash_identity_and_finite_values(tmp_path):
    path = tmp_path / "artifact.npz"
    digest = save_npz_atomic(
        path,
        {
            "image_id": np.asarray("sample"),
            "values": np.asarray([1.0, 2.0], dtype=np.float32),
        },
    )

    payload = reload_npz_checked(
        path, expected_sha256=digest, expected_image_id="sample"
    )

    np.testing.assert_array_equal(payload["values"], [1.0, 2.0])
    with pytest.raises(RuntimeError, match="hash mismatch"):
        reload_npz_checked(path, expected_sha256="0" * 64, expected_image_id="sample")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        reload_npz_checked(path, expected_sha256=digest, expected_image_id="other")

    bad = tmp_path / "bad.npz"
    bad_digest = save_npz_atomic(
        bad,
        {
            "image_id": np.asarray("sample"),
            "values": np.asarray([np.inf], dtype=np.float32),
        },
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        reload_npz_checked(bad, expected_sha256=bad_digest, expected_image_id="sample")


def test_full_runs_reject_dirty_or_untracked_runtime_sources():
    clean = {
        "tracked_dirty": False,
        "runtime_source_tracked": {"runner.py": True},
    }
    enforce_production_source(clean, allow_uncommitted=False, limit=0)

    dirty = {**clean, "tracked_dirty": True}
    with pytest.raises(RuntimeError, match="clean, tracked"):
        enforce_production_source(dirty, allow_uncommitted=False, limit=0)
    with pytest.raises(ValueError, match="smoke-only"):
        enforce_production_source(dirty, allow_uncommitted=True, limit=0)
    enforce_production_source(dirty, allow_uncommitted=True, limit=2)

    untracked = {
        "tracked_dirty": False,
        "runtime_source_tracked": {"runner.py": False},
    }
    with pytest.raises(RuntimeError, match="untracked"):
        enforce_production_source(untracked, allow_uncommitted=False, limit=0)
