"""Shared, side-effect-conscious helpers for Experiment 3.

Experiment 1/2 results, checkpoints, and VOC data are immutable inputs.  The
helpers here deliberately keep every derived artifact below a new Experiment 3
run root and make provenance failures explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_KEYS = ("mctformer_plus", "mctformer")
MODEL_FACTORY = {
    "mctformer": "mctformerv2",
    "mctformer_plus": "mctformerplus",
}
EXPECTED_IMAGES = 1449
EXPECTED_POSITIVE_PAIRS = 2147
EXPECTED_MULTILABEL_IMAGES = 522
EXPECTED_LAYERS = 12
EXPECTED_CLASSES = 20
EXPECTED_PATCHES = 784
BOOTSTRAP_REPEATS = 5000
# Reuse the Experiment 2 paired-analysis seed so every Experiment 3 family can
# share one explicit, deterministic image-cluster draw convention.
BOOTSTRAP_SEED = 20260901
STRICT_TOLERANCE = 1e-6


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def json_dump(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def require_tgca_repro() -> None:
    active = os.environ.get("CONDA_DEFAULT_ENV")
    if active != "tgca-repro":
        raise RuntimeError(f"Experiment 3 requires tgca-repro; active={active!r}")


def assert_new_output(path: Path, immutable_inputs: Sequence[Path]) -> Path:
    output = path.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    for item in immutable_inputs:
        source = item.expanduser().resolve()
        base = source if source.is_dir() else source.parent
        try:
            output.relative_to(base)
        except ValueError:
            continue
        raise ValueError(f"output {output} is nested below immutable input {base}")
    return output


def git_state(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    def run(*args: str) -> str:
        result = subprocess.run(
            args,
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "status_short": run("git", "status", "--short").splitlines(),
        "repository_url": run("git", "config", "--get", "remote.origin.url"),
        "host": os.uname().nodename,
    }


def runtime_source_state(relative_paths: Iterable[str]) -> dict[str, object]:
    tracked: dict[str, bool] = {}
    hashes: dict[str, str] = {}
    for relative in sorted(set(relative_paths)):
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        tracked[relative] = (
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", relative],
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        )
        hashes[relative] = sha256_file(path)
    state = git_state()
    state["runtime_source_tracked"] = tracked
    state["runtime_source_sha256"] = hashes
    state["tracked_dirty"] = bool(
        subprocess.run(
            ["git", "diff", "--quiet"], cwd=REPO_ROOT, check=False
        ).returncode
        or subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT, check=False
        ).returncode
    )
    return state


def enforce_production_source(
    state: Mapping[str, object], *, allow_uncommitted: bool, limit: int
) -> None:
    if allow_uncommitted and limit <= 0:
        raise ValueError("--allow-uncommitted-source is smoke-only (--limit > 0)")
    tracked = state.get("runtime_source_tracked")
    if not isinstance(tracked, Mapping):
        raise TypeError("missing runtime source tracking metadata")
    untracked = sorted(str(key) for key, value in tracked.items() if value is not True)
    unsafe = bool(state.get("tracked_dirty")) or bool(untracked)
    if unsafe and not allow_uncommitted:
        raise RuntimeError(
            "full Experiment 3 runs require clean, tracked runtime sources; "
            f"tracked_dirty={state.get('tracked_dirty')}, untracked={untracked}"
        )


def ordered_val_ids(path: Path) -> list[str]:
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"VOC list is empty or contains duplicates: {path}")
    return ids


def load_image_labels(path: Path, image_ids: Sequence[str]) -> np.ndarray:
    payload = np.load(path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict in {path}")
    rows = []
    for image_id in image_ids:
        if image_id not in payload:
            raise KeyError(f"missing label for {image_id}")
        row = np.asarray(payload[image_id])
        if row.shape != (EXPECTED_CLASSES,) or not np.isin(row, (0, 1)).all():
            raise ValueError(f"invalid label for {image_id}: {row.shape}")
        rows.append(row.astype(np.uint8, copy=False))
    return np.stack(rows)


def parse_common_limit(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-uncommitted-source", action="store_true")
