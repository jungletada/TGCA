#!/usr/bin/env python3
"""Re-hash Experiment 2 immutable inputs and compare with the audit baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment2.common import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def verify(before_manifest: Path, output_dir: Path) -> dict[str, object]:
    before_manifest = before_manifest.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    before = pd.read_csv(before_manifest)
    required = {"absolute_path", "size_bytes", "sha256"}
    missing_columns = required.difference(before.columns)
    if missing_columns:
        raise ValueError(f"before manifest lacks columns: {sorted(missing_columns)}")
    rows: list[dict[str, object]] = []
    for record in before.to_dict(orient="records"):
        path = Path(str(record["absolute_path"]))
        exists = path.is_file()
        size = path.stat().st_size if exists else -1
        digest = sha256_file(path) if exists else ""
        rows.append(
            {
                **record,
                "exists_after": exists,
                "size_bytes_after": int(size),
                "sha256_after": digest,
                "size_unchanged": exists and int(size) == int(record["size_bytes"]),
                "sha256_unchanged": exists and digest == str(record["sha256"]),
            }
        )
    after = pd.DataFrame(rows)
    after.to_csv(output_dir / "file_manifest_after.csv", index=False)
    missing_count = int((~after["exists_after"]).sum())
    size_changed = int((~after["size_unchanged"]).sum())
    hash_changed = int((~after["sha256_unchanged"]).sum())
    passed = missing_count == 0 and size_changed == 0 and hash_changed == 0
    result = {
        "status": "complete" if passed else "failed",
        "integrity_passed": passed,
        "before_manifest": str(before_manifest),
        "files_checked": len(after),
        "bytes_checked": int(
            after.loc[after["exists_after"], "size_bytes_after"].sum()
        ),
        "missing_files": missing_count,
        "size_changed_files": size_changed,
        "sha256_changed_files": hash_changed,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (output_dir / "immutability_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    result = verify(args.before_manifest, args.output_dir)
    print(json.dumps(result, sort_keys=True))
    if not result["integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
