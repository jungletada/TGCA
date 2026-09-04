"""Shared, side-effect-conscious helpers for Experiment 2.

The source Experiment 1 result trees, checkpoints, and VOC data are inputs.  The
helpers in this module never create files below those inputs and deliberately
require unambiguous discovery of completed upstream runs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


VOC_CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

MODEL_METADATA_NAMES = {
    "mctformer": "mctformerv2",
    "mctformer_plus": "mctformerplus",
}

LOW_LEVEL_SOURCE_PATHS = (
    "datasets_cam.py",
    "utils.py",
    "models/mctformer.py",
    "models/mctformer_plus.py",
    "models/vit.py",
)


def timestamp() -> str:
    """Return a timezone-aware, filesystem-friendly provenance timestamp."""

    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: Path) -> str:
    """Stream a file into SHA-256 without changing its metadata."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}, got {type(value).__name__}")
    return value


def git_blob(repo_root: Path, commit: str, relative_path: str) -> Optional[bytes]:
    """Read a historical Git blob, returning ``None`` when unavailable."""

    if not commit:
        return None
    process = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.stdout if process.returncode == 0 else None


def git_metadata(repo_root: Path) -> dict[str, object]:
    """Record live Git provenance without requiring a clean worktree."""

    def run(*arguments: str) -> str:
        process = subprocess.run(
            list(arguments),
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return process.stdout.strip() if process.returncode == 0 else ""

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "status_short": run("git", "status", "--short").splitlines(),
        "repository_url": run("git", "config", "--get", "remote.origin.url"),
        "host": os.uname().nodename,
    }


def json_dump(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write JSON in the analysis output tree."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def csv_dump(
    path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]
) -> None:
    """Atomically write a deterministic CSV file."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@dataclass(frozen=True)
class DiscoveryResult:
    root: Path
    inspected: tuple[str, ...]


def _experiment1_candidate(candidate: Path, expected_model: str) -> tuple[bool, str]:
    try:
        completion = read_json(candidate / "completion.json")
        metadata = read_json(candidate / "metadata.json")
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as error:
        return False, f"{candidate}: unreadable ({error})"
    actual_model = metadata.get("model", {}).get("name")
    status = completion.get("status")
    run_kind = completion.get("run_kind")
    valid = (
        actual_model == MODEL_METADATA_NAMES[expected_model]
        and status == "complete"
        and run_kind == "full"
    )
    return (
        valid,
        f"{candidate}: status={status}, run_kind={run_kind}, model={actual_model}",
    )


def resolve_completed_experiment1_root(
    search_root: Path,
    expected_model: str,
    override: Optional[Path] = None,
) -> DiscoveryResult:
    """Resolve exactly one completed full Experiment 1 model result.

    An override is still validated; it does not bypass completion/model checks.
    """

    if expected_model not in MODEL_METADATA_NAMES:
        raise ValueError(f"unsupported model key: {expected_model}")
    search_root = search_root.expanduser().resolve()
    if override is not None:
        candidates = [override.expanduser().resolve()]
    elif (search_root / "completion.json").is_file():
        candidates = [search_root]
    elif search_root.is_dir():
        candidates = sorted(
            {path.parent.resolve() for path in search_root.rglob("completion.json")}
        )
    else:
        candidates = []

    valid: list[Path] = []
    inspected: list[str] = []
    for candidate in candidates:
        is_valid, description = _experiment1_candidate(candidate, expected_model)
        inspected.append(description)
        if is_valid:
            valid.append(candidate)
    if len(valid) != 1:
        raise RuntimeError(
            f"expected exactly one completed full {expected_model} result; "
            f"search_root={search_root}, override={override}, valid={valid}, "
            f"inspected={inspected}"
        )
    return DiscoveryResult(valid[0], tuple(inspected))


def _normal_path(value: object) -> Optional[Path]:
    if not isinstance(value, str) or not value:
        return None
    return Path(value).expanduser().resolve()


def resolve_completed_paired_analysis_root(
    search_root: Path,
    source_roots: Mapping[str, Path],
    override: Optional[Path] = None,
) -> DiscoveryResult:
    """Resolve one complete paired analysis linked to the selected sources."""

    search_root = search_root.expanduser().resolve()
    if override is not None:
        candidates = [override.expanduser().resolve()]
    elif (search_root / "run_metadata.json").is_file():
        candidates = [search_root]
    elif search_root.is_dir():
        candidates = sorted(
            {path.parent.resolve() for path in search_root.rglob("run_metadata.json")}
        )
    else:
        candidates = []

    expected = {key: value.resolve() for key, value in source_roots.items()}
    valid: list[Path] = []
    inspected: list[str] = []
    for candidate in candidates:
        try:
            metadata = read_json(candidate / "run_metadata.json")
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as error:
            inspected.append(f"{candidate}: unreadable ({error})")
            continue
        recorded = {
            key: _normal_path(metadata.get("source_roots", {}).get(key))
            for key in expected
        }
        linked = all(recorded[key] == expected[key] for key in expected)
        reports_exist = all(
            (candidate / "reports" / name).is_file()
            for name in ("EXPERIMENT1_ANALYSIS_REPORT.md", "EXPERIMENT2_READINESS.md")
        )
        status = metadata.get("status")
        inspected.append(
            f"{candidate}: status={status}, linked={linked}, reports={reports_exist}"
        )
        if status == "complete" and linked and reports_exist:
            valid.append(candidate)
    if len(valid) != 1:
        raise RuntimeError(
            "expected exactly one completed paired Experiment 1 analysis linked to "
            f"{expected}; search_root={search_root}, override={override}, "
            f"valid={valid}, inspected={inspected}"
        )
    return DiscoveryResult(valid[0], tuple(inspected))


def assert_output_outside_inputs(output: Path, inputs: Sequence[Path]) -> None:
    """Reject an output directory nested below any immutable input directory."""

    output = output.expanduser().resolve()
    for value in inputs:
        source = value.expanduser().resolve()
        if source.is_file():
            source = source.parent
        try:
            output.relative_to(source)
        except ValueError:
            continue
        raise ValueError(
            f"analysis output {output} must not be inside immutable input {source}"
        )
