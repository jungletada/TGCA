import hashlib

import pandas as pd

from analysis.lazy_assignment.experiment2.verify_experiment2_immutability import verify


def test_immutability_verifier_detects_no_change(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "before.csv"
    pd.DataFrame(
        [
            {
                "absolute_path": str(source),
                "size_bytes": source.stat().st_size,
                "sha256": digest,
            }
        ]
    ).to_csv(manifest, index=False)
    result = verify(manifest, tmp_path / "out")
    assert result["integrity_passed"]
    assert result["files_checked"] == 1


def test_immutability_verifier_detects_change(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"before")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "before.csv"
    pd.DataFrame(
        [
            {
                "absolute_path": str(source),
                "size_bytes": source.stat().st_size,
                "sha256": digest,
            }
        ]
    ).to_csv(manifest, index=False)
    source.write_bytes(b"after!")
    result = verify(manifest, tmp_path / "out")
    assert not result["integrity_passed"]
    assert result["sha256_changed_files"] == 1
