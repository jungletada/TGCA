"""Small provenance and immutable-input helpers for the Phase A/B analyses."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence


EXPECTED_ENVIRONMENT = "tgca-repro"
EXPECTED_VOC_IMAGES = 1449
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260901


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_environment() -> None:
    observed = os.environ.get("CONDA_DEFAULT_ENV")
    if observed != EXPECTED_ENVIRONMENT:
        raise RuntimeError(
            f"This analysis requires Conda environment {EXPECTED_ENVIRONMENT!r}; "
            f"observed {observed!r}"
        )


def _git(repo_root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def git_metadata(repo_root: Path) -> dict[str, object]:
    return {
        "commit": _git(repo_root, "rev-parse", "HEAD"),
        "branch": _git(repo_root, "branch", "--show-current"),
        "repository_url": _git(repo_root, "config", "--get", "remote.origin.url"),
        "status_short": _git(repo_root, "status", "--short").splitlines(),
        "tracked_diff": _git(repo_root, "diff", "--name-only").splitlines(),
        "staged_diff": _git(repo_root, "diff", "--cached", "--name-only").splitlines(),
        "host": os.uname().nodename,
    }


def require_clean_tracked(repo_root: Path) -> dict[str, object]:
    metadata = git_metadata(repo_root)
    if metadata["tracked_diff"] or metadata["staged_diff"]:
        raise RuntimeError(
            "Refusing to run from tracked dirty state; untracked user files are "
            f"preserved and allowed. status={metadata['status_short']}"
        )
    return metadata


def command_line() -> str:
    return shlex.join([sys.executable, *sys.argv])


def create_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def json_dump(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def csv_dump(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fields: Sequence[str] | None = None,
) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty CSV {path}")
    columns = list(fields or materialized[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def text_dump(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_environment_manifests(output_dir: Path) -> dict[str, object]:
    commands = {
        "pip_freeze.txt": [sys.executable, "-m", "pip", "freeze"],
        "conda_explicit.txt": ["conda", "list", "--explicit"],
    }
    records: dict[str, object] = {}
    for name, command in commands.items():
        process = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        destination = output_dir / name
        text_dump(destination, process.stdout)
        records[name] = {
            "command": shlex.join(command),
            "returncode": process.returncode,
            "sha256": sha256_file(destination),
        }
    return records


class RunLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __call__(self, message: str) -> None:
        line = f"[{timestamp()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
