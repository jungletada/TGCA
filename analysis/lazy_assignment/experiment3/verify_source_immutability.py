#!/usr/bin/env python3
"""Re-hash the Experiment 3 immutable source manifest after all validations."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment3.common import (
    json_dump,
    sha256_file,
    timestamp,
)  # noqa: E402


def verify(before_manifest: Path, output_dir: Path) -> dict[str, object]:
    before_manifest = before_manifest.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    before = pd.read_csv(before_manifest)
    required = {"absolute_path", "size_bytes", "sha256"}
    if not required.issubset(before.columns):
        raise ValueError(
            f"manifest missing columns {sorted(required - set(before.columns))}"
        )
    records = []
    for row in before.to_dict(orient="records"):
        path = Path(str(row["absolute_path"]))
        exists = path.is_file()
        size = path.stat().st_size if exists else -1
        digest = sha256_file(path) if exists else ""
        records.append(
            {
                **row,
                "exists_after": exists,
                "size_bytes_after": int(size),
                "sha256_after": digest,
                "size_unchanged": exists and int(size) == int(row["size_bytes"]),
                "sha256_unchanged": exists and digest == str(row["sha256"]),
            }
        )
    after = pd.DataFrame(records)
    after.to_csv(output_dir / "immutable_manifest_after.csv", index=False)
    missing = int((~after["exists_after"]).sum())
    size_changed = int((~after["size_unchanged"]).sum())
    hash_changed = int((~after["sha256_unchanged"]).sum())
    passed = missing == size_changed == hash_changed == 0
    result = {
        "status": "complete" if passed else "failed",
        "integrity_passed": passed,
        "before_manifest": str(before_manifest),
        "files_checked": int(len(after)),
        "bytes_checked": int(
            after.loc[after["exists_after"], "size_bytes_after"].sum()
        ),
        "missing_files": missing,
        "size_changed_files": size_changed,
        "sha256_changed_files": hash_changed,
        "completed_at": timestamp(),
    }
    json_dump(output_dir / "immutability_verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.before_manifest, args.output_dir)
    print(json.dumps(result, sort_keys=True))
    if not result["integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
