"""Shared I/O and provenance helpers for Experiment 1 result analysis."""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
import torch


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
MODEL_ORDER = ("mctformer", "mctformer_plus")


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def git_metadata(repo_root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=repo_root, text=True).strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "status_short": run("git", "status", "--short").splitlines(),
        "repository_url": subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            check=False,
        ).stdout.strip(),
    }


def environment_metadata() -> dict[str, object]:
    return {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pyarrow": pyarrow.__version__,
        "matplotlib": matplotlib.__version__,
    }


def resolve_completed_result_root(path: Path, expected_model: str) -> tuple[Path, list[str]]:
    path = path.resolve()
    candidates: list[Path] = []
    if (path / "completion.json").is_file():
        candidates.append(path)
    elif path.is_dir():
        candidates.extend(parent for parent in path.rglob("completion.json") if parent.is_file())
        candidates = [item.parent for item in candidates]
    valid: list[Path] = []
    inspected: list[str] = []
    for candidate in sorted(set(candidates)):
        try:
            completion = json.loads((candidate / "completion.json").read_text())
            metadata = json.loads((candidate / "metadata.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            inspected.append(f"{candidate}: unreadable ({error})")
            continue
        status = completion.get("status")
        model = metadata.get("model", {}).get("name")
        run_kind = completion.get("run_kind")
        inspected.append(
            f"{candidate}: status={status}, model={model}, run_kind={run_kind}"
        )
        model_matches = (
            expected_model == "mctformer" and model == "mctformerv2"
        ) or (expected_model == "mctformer_plus" and model == "mctformerplus")
        if status == "complete" and run_kind == "full" and model_matches:
            valid.append(candidate)
    if len(valid) != 1:
        raise RuntimeError(
            f"expected exactly one completed {expected_model} full result under {path}; "
            f"found {valid}; inspected={inspected}"
        )
    return valid[0], inspected


def assert_output_outside_sources(output: Path, sources: Sequence[Path]) -> None:
    output = output.resolve()
    for source in sources:
        source = source.resolve()
        try:
            output.relative_to(source)
        except ValueError:
            continue
        raise ValueError(f"analysis output {output} must not be inside source result {source}")


class AnalysisLog:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str) -> None:
        line = f"[{timestamp()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

