#!/usr/bin/env python3
"""Verify that every audited Experiment 1 source file is unchanged after analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.lazy_assignment.experiment1_analysis_common import (
    json_dump,
    sha256_file,
    timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-manifest", type=Path, required=True)
    parser.add_argument("--after-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify_manifest(before: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    changes: list[dict[str, object]] = []
    expected_paths = {Path(value).resolve() for value in before["absolute_path"]}
    roots = {Path(value).resolve() for value in before["result_root"]}
    live_paths = {
        path.resolve()
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }
    for extra in sorted(live_paths - expected_paths):
        changes.append({"path": str(extra), "issue": "new_source_file_after_audit"})
    for missing in sorted(expected_paths - live_paths):
        changes.append({"path": str(missing), "issue": "source_file_missing_after_audit"})

    for row in before.itertuples(index=False):
        path = Path(row.absolute_path)
        if not path.is_file():
            continue
        stat = path.stat()
        current_hash = sha256_file(path)
        item = {
            "model": row.model,
            "result_root": row.result_root,
            "relative_path": row.relative_path,
            "absolute_path": row.absolute_path,
            "kind": row.kind,
            "size_bytes_before": int(row.size_bytes),
            "size_bytes_after": int(stat.st_size),
            "mtime_ns_before": int(row.mtime_ns_before),
            "mtime_ns_after": int(stat.st_mtime_ns),
            "sha256_before": row.sha256_before,
            "sha256_after": current_hash,
            "content_unchanged": bool(
                current_hash == row.sha256_before and stat.st_size == row.size_bytes
            ),
            "mtime_unchanged": bool(stat.st_mtime_ns == row.mtime_ns_before),
        }
        rows.append(item)
        if not item["content_unchanged"] or not item["mtime_unchanged"]:
            changes.append(
                {
                    "path": str(path),
                    "issue": "source_content_size_or_mtime_changed",
                    "sha256_before": row.sha256_before,
                    "sha256_after": current_hash,
                    "size_before": int(row.size_bytes),
                    "size_after": int(stat.st_size),
                    "mtime_ns_before": int(row.mtime_ns_before),
                    "mtime_ns_after": int(stat.st_mtime_ns),
                }
            )
    return pd.DataFrame(rows), changes


def verify(args: argparse.Namespace) -> dict[str, object]:
    before = pd.read_csv(args.before_manifest.resolve())
    after, changes = verify_manifest(before)
    after.to_csv(args.after_manifest.resolve(), index=False)
    npz = after[after["kind"] == "score_npz"]
    report: dict[str, object] = {
        "generated_at": timestamp(),
        "passed": not changes and len(after) == len(before),
        "before_manifest": str(args.before_manifest.resolve()),
        "after_manifest": str(args.after_manifest.resolve()),
        "source_files_checked": len(after),
        "source_npz_checked": len(npz),
        "source_npz_content_hashes_unchanged": bool(npz["content_unchanged"].all()),
        "all_content_hashes_unchanged": bool(after["content_unchanged"].all()),
        "all_mtimes_unchanged": bool(after["mtime_unchanged"].all()),
        "changes": changes,
    }
    json_dump(args.output.resolve(), report)
    if not report["passed"]:
        raise RuntimeError(f"source immutability check failed: {changes[:3]}")
    return report


def main() -> None:
    verify(parse_args())


if __name__ == "__main__":
    main()
