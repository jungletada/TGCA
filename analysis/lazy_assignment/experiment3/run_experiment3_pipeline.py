#!/usr/bin/env python3
"""Initialize and execute the immutable Experiment 3 production queue.

The queue is intentionally sequential because every inference stage uses the
same GPU.  It creates a new run root, records the complete command list before
execution, maintains a fail-closed stage ledger, and delegates the final
atomic status/manifest update to ``generate_experiment3_report.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment3.common import (  # noqa: E402
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    assert_new_output,
    git_state,
    json_dump,
    read_json,
    require_tgca_repro,
    sha256_file,
    timestamp,
)


PIPELINE_NAME = "experiment3_three_inference_only_validations"


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]


def _script(name: str) -> str:
    path = REPO_ROOT / "analysis" / "lazy_assignment" / "experiment3" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def _test_paths(pattern: str) -> list[str]:
    paths = sorted((REPO_ROOT / "tests").glob(pattern))
    if not paths:
        raise RuntimeError(f"no tests match {pattern!r}")
    return [str(path) for path in paths]


def _build_stages(
    *,
    python: str,
    experiment2_root: Path,
    run_root: Path,
    device: str,
    num_workers: int,
) -> tuple[Stage, ...]:
    audit = run_root / "audit"
    source = audit / "source_metadata.json"
    a_root = run_root / "presence_axis"
    b_root = run_root / "cam_layer_intervention"
    c_root = run_root / "c2c_intervention"
    py = (python,)
    common_runner = (
        "--batch-size",
        "8",
        "--num-workers",
        str(num_workers),
        "--device",
        device,
    )
    stages = [
        Stage(
            "input_audit",
            (
                *py,
                _script("audit_experiment3_inputs.py"),
                "--experiment2-root",
                str(experiment2_root),
                "--output-dir",
                str(audit),
            ),
        ),
        Stage(
            "test_experiment3",
            (*py, "-m", "pytest", "-q", *_test_paths("test_experiment3_*.py")),
        ),
        Stage(
            "ruff_experiment3",
            (
                "ruff",
                "check",
                str(REPO_ROOT / "analysis/lazy_assignment/experiment3"),
                *_test_paths("test_experiment3_*.py"),
            ),
        ),
        Stage(
            "validation_a_smoke_mctformer_plus_50",
            (
                *py,
                _script("run_presence_axis_analysis.py"),
                "--model",
                "mctformer_plus",
                "--source-metadata",
                str(source),
                "--output-dir",
                str(a_root / "smoke" / "mctformer_plus_50"),
                *common_runner,
                "--limit",
                "50",
            ),
        ),
        Stage(
            "validation_a_full_mctformer_plus",
            (
                *py,
                _script("run_presence_axis_analysis.py"),
                "--model",
                "mctformer_plus",
                "--source-metadata",
                str(source),
                "--output-dir",
                str(a_root / "mctformer_plus"),
                *common_runner,
            ),
        ),
        Stage(
            "validation_a_full_mctformer",
            (
                *py,
                _script("run_presence_axis_analysis.py"),
                "--model",
                "mctformer",
                "--source-metadata",
                str(source),
                "--output-dir",
                str(a_root / "mctformer"),
                *common_runner,
            ),
        ),
        Stage(
            "validation_a_analysis",
            (
                *py,
                _script("analyze_presence_axis.py"),
                "--mctformer-run-root",
                str(a_root / "mctformer"),
                "--mctformer-plus-run-root",
                str(a_root / "mctformer_plus"),
                "--source-metadata",
                str(source),
                "--output-dir",
                str(a_root / "analysis"),
                "--bootstrap-repeats",
                str(BOOTSTRAP_REPEATS),
                "--bootstrap-seed",
                str(BOOTSTRAP_SEED),
            ),
        ),
        Stage(
            "validation_b_smoke_mctformer_plus_20",
            (
                *py,
                _script("run_cam_layer_intervention.py"),
                "--model",
                "mctformer_plus",
                "--source-metadata",
                str(source),
                "--output-dir",
                str(b_root / "smoke" / "mctformer_plus_20"),
                *common_runner,
                "--limit",
                "20",
            ),
        ),
        Stage(
            "validation_b_smoke_mctformer_20",
            (
                *py,
                _script("run_cam_layer_intervention.py"),
                "--model",
                "mctformer",
                "--source-metadata",
                str(source),
                "--output-dir",
                str(b_root / "smoke" / "mctformer_20"),
                *common_runner,
                "--limit",
                "20",
            ),
        ),
        Stage(
            "validation_b_smoke_analysis",
            (
                *py,
                _script("analyze_cam_layer_readout.py"),
                "--mctformer-run-root",
                str(b_root / "smoke" / "mctformer_20"),
                "--mctformer-plus-run-root",
                str(b_root / "smoke" / "mctformer_plus_20"),
                "--source-metadata",
                str(source),
                "--output-dir",
                str(b_root / "smoke" / "analysis"),
                "--bootstrap-repeats",
                "50",
                "--bootstrap-seed",
                str(BOOTSTRAP_SEED),
                "--allow-smoke",
            ),
        ),
        Stage(
            "validation_b_full_mctformer_plus",
            (
                *py,
                _script("run_cam_layer_intervention.py"),
                "--model",
                "mctformer_plus",
                "--source-metadata",
                str(source),
                "--output-dir",
                str(b_root / "mctformer_plus"),
                *common_runner,
            ),
        ),
        Stage(
            "validation_b_full_mctformer",
            (
                *py,
                _script("run_cam_layer_intervention.py"),
                "--model",
                "mctformer",
                "--source-metadata",
                str(source),
                "--output-dir",
                str(b_root / "mctformer"),
                *common_runner,
            ),
        ),
        Stage(
            "validation_b_analysis",
            (
                *py,
                _script("analyze_cam_layer_readout.py"),
                "--mctformer-run-root",
                str(b_root / "mctformer"),
                "--mctformer-plus-run-root",
                str(b_root / "mctformer_plus"),
                "--source-metadata",
                str(source),
                "--output-dir",
                str(b_root / "analysis"),
                "--bootstrap-repeats",
                str(BOOTSTRAP_REPEATS),
                "--bootstrap-seed",
                str(BOOTSTRAP_SEED),
            ),
        ),
        Stage(
            "validation_c_unit_tests",
            (*py, "-m", "pytest", "-q", *_test_paths("test_experiment3_c2c_*.py")),
        ),
        Stage(
            "validation_c_smoke_mctformer_plus_50",
            (
                *py,
                _script("run_c2c_intervention.py"),
                "--model",
                "mctformer_plus",
                "--source-metadata",
                str(source),
                "--output-dir",
                str(c_root / "smoke" / "mctformer_plus_50"),
                *common_runner,
                "--limit",
                "50",
            ),
        ),
        Stage(
            "validation_c_smoke_analysis",
            (
                *py,
                _script("analyze_c2c_intervention.py"),
                "--mctformer-plus-run-root",
                str(c_root / "smoke" / "mctformer_plus_50"),
                "--output-dir",
                str(c_root / "smoke" / "analysis"),
                "--bootstrap-repeats",
                "50",
                "--bootstrap-seed",
                str(BOOTSTRAP_SEED),
                "--allow-smoke",
            ),
        ),
        Stage(
            "validation_c_full_mctformer_plus",
            (
                *py,
                _script("run_c2c_intervention.py"),
                "--model",
                "mctformer_plus",
                "--source-metadata",
                str(source),
                "--output-dir",
                str(c_root / "mctformer_plus"),
                *common_runner,
            ),
        ),
        Stage(
            "validation_c_analysis",
            (
                *py,
                _script("analyze_c2c_intervention.py"),
                "--mctformer-plus-run-root",
                str(c_root / "mctformer_plus"),
                "--output-dir",
                str(c_root / "analysis"),
                "--bootstrap-repeats",
                str(BOOTSTRAP_REPEATS),
                "--bootstrap-seed",
                str(BOOTSTRAP_SEED),
            ),
        ),
        Stage(
            "final_source_immutability",
            (
                *py,
                _script("verify_source_immutability.py"),
                "--before-manifest",
                str(audit / "immutable_manifest_before.csv"),
                "--output-dir",
                str(audit / "final_immutability"),
            ),
        ),
        Stage(
            "rule_selected_examples",
            (
                *py,
                _script("render_experiment3_examples.py"),
                "--run-root",
                str(run_root),
                "--validation-a-root",
                str(a_root / "analysis"),
                "--validation-b-root",
                str(b_root / "analysis"),
                "--validation-c-root",
                str(c_root / "analysis"),
                "--source-metadata",
                str(source),
            ),
        ),
        Stage(
            "final_reports",
            (
                *py,
                _script("generate_experiment3_report.py"),
                "--run-root",
                str(run_root),
                "--validation-a-root",
                str(a_root / "analysis"),
                "--validation-b-root",
                str(b_root / "analysis"),
                "--validation-c-root",
                str(c_root / "analysis"),
                "--source-verification",
                str(audit / "final_immutability" / "immutability_verification.json"),
            ),
        ),
    ]
    if len({stage.name for stage in stages}) != len(stages):
        raise AssertionError("pipeline stage names must be unique")
    return tuple(stages)


def _write_exact_commands(
    path: Path, stages: Sequence[Stage], *, recorded_run_root: Path | None = None
) -> None:
    command_run_root = path.parent if recorded_run_root is None else recorded_run_root
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        'test "${CONDA_DEFAULT_ENV:-}" = "tgca-repro"',
        "",
        "# Audit trail generated before execution. Reproduction requires a new run root",
        "# because every stage deliberately refuses to overwrite an existing output.",
    ]
    for stage in stages:
        log = command_run_root / "logs" / f"{stage.name}.log"
        lines.extend(
            [
                "",
                f"# stage: {stage.name}",
                f"{shlex.join(stage.command)} 2>&1 | tee {shlex.quote(str(log))}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _git_tag_commit(tag: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"refs/tags/{tag}^{{}}"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"implementation tag is missing: {tag}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _context_hashes(paths: Sequence[Path]) -> Mapping[str, Mapping[str, object]]:
    records: dict[str, Mapping[str, object]] = {}
    for item in paths:
        path = item.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        records[str(path)] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return records


def initialize(
    *,
    run_root: Path,
    experiment2_root: Path,
    implementation_tag: str,
    device: str,
    num_workers: int,
    context_documents: Sequence[Path],
    invocation: Sequence[str],
) -> tuple[Path, tuple[Stage, ...]]:
    require_tgca_repro()
    source_state = git_state(REPO_ROOT)
    if source_state.get("status_short"):
        raise RuntimeError("production pipeline requires a completely clean worktree")
    head = str(source_state.get("commit", ""))
    if not head or _git_tag_commit(implementation_tag) != head:
        raise RuntimeError("implementation tag must resolve exactly to clean HEAD")
    experiment2_root = experiment2_root.expanduser().resolve()
    if not experiment2_root.is_dir():
        raise FileNotFoundError(experiment2_root)
    run_root = assert_new_output(run_root, (experiment2_root,)).resolve()
    context_hashes = _context_hashes(context_documents)
    stages = _build_stages(
        python=sys.executable,
        experiment2_root=experiment2_root,
        run_root=run_root,
        device=device,
        num_workers=num_workers,
    )
    started = timestamp()
    commands_path = run_root / "exact_commands.sh"
    metadata = {
        "status": "running",
        "pipeline": PIPELINE_NAME,
        "run_root": str(run_root),
        "experiment2_root": str(experiment2_root),
        "implementation_commit": head,
        "implementation_tag": implementation_tag,
        "git": source_state,
        "environment": {
            "conda": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.executable,
        },
        "evaluation": {
            "dataset": "PASCAL VOC 2012 val",
            "images": 1449,
            "input_size": 448,
            "batch_size": 8,
            "device": device,
            "num_workers": num_workers,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "image cluster",
        },
        "context_documents": context_hashes,
        "orchestrator_command": shlex.join(invocation),
        "stage_order": [stage.name for stage in stages],
        "started_at": started,
    }
    status = {
        "status": "running",
        "pipeline": PIPELINE_NAME,
        "run_root": str(run_root),
        "started_at": started,
        "active_stage": None,
        "stages": {
            stage.name: {"status": "pending", "command": shlex.join(stage.command)}
            for stage in stages
        },
    }
    run_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{run_root.name}.initialize-", dir=run_root.parent)
    )
    try:
        (temporary / "logs").mkdir()
        temporary_commands = temporary / "exact_commands.sh"
        _write_exact_commands(temporary_commands, stages, recorded_run_root=run_root)
        metadata["exact_commands"] = {
            "path": str(commands_path),
            "sha256": sha256_file(temporary_commands),
        }
        json_dump(temporary / "pipeline_metadata.json", metadata)
        json_dump(temporary / "pipeline_status.json", status)
        if run_root.exists():
            raise FileExistsError(run_root)
        temporary.replace(run_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return run_root, stages


def _record_stage(
    run_root: Path,
    stage: Stage,
    status_value: str,
    **extra: object,
) -> None:
    path = run_root / "pipeline_status.json"
    payload = read_json(path)
    stages = payload.setdefault("stages", {})
    if not isinstance(stages, dict):
        raise TypeError("pipeline stage ledger is not a JSON object")
    record = dict(stages.get(stage.name, {}))
    record.update({"status": status_value, **extra})
    stages[stage.name] = record
    payload["active_stage"] = stage.name if status_value == "running" else None
    if status_value == "failed":
        payload["status"] = "failed"
        payload["failed_stage"] = stage.name
    json_dump(path, payload)


def _run_stage(run_root: Path, stage: Stage, *, finalizer: bool = False) -> None:
    log_path = run_root / "logs" / f"{stage.name}.log"
    started_wall = timestamp()
    started_clock = time.perf_counter()
    _record_stage(run_root, stage, "running", started_at=started_wall)
    with log_path.open("x", encoding="utf-8") as log:
        command_text = shlex.join(stage.command)
        header = f"[{started_wall}] $ {command_text}\n"
        log.write(header)
        log.flush()
        print(header, end="", flush=True)
        process = subprocess.Popen(
            stage.command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        returncode = process.wait()
    elapsed = time.perf_counter() - started_clock
    if returncode:
        _record_stage(
            run_root,
            stage,
            "failed",
            returncode=returncode,
            completed_at=timestamp(),
            elapsed_seconds=elapsed,
            log=str(log_path),
        )
        raise subprocess.CalledProcessError(returncode, stage.command)
    if finalizer:
        # The finalizer atomically seals pipeline_status.json and includes its
        # digest in the delivery manifest.  It also closes this stage record;
        # no pipeline control may be modified after a successful return.
        sealed = read_json(run_root / "pipeline_status.json")
        if sealed.get("status") != "complete":
            raise RuntimeError("finalizer returned without sealing pipeline status")
        return
    _record_stage(
        run_root,
        stage,
        "complete",
        returncode=0,
        completed_at=timestamp(),
        elapsed_seconds=elapsed,
        log=str(log_path),
    )


def execute_pipeline(run_root: Path, stages: Sequence[Stage]) -> None:
    for index, stage in enumerate(stages):
        _run_stage(run_root, stage, finalizer=index == len(stages) - 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment2-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--implementation-tag", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--context-document",
        type=Path,
        action="append",
        default=[],
        help="Plan/report read before coding; stored by path, size, and SHA-256.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    invocation = (sys.executable, *sys.argv)
    run_root: Path | None = None
    try:
        run_root, stages = initialize(
            run_root=args.run_root,
            experiment2_root=args.experiment2_root,
            implementation_tag=args.implementation_tag,
            device=args.device,
            num_workers=args.num_workers,
            context_documents=args.context_document,
            invocation=invocation,
        )
        execute_pipeline(run_root, stages)
        print(json.dumps({"status": "complete", "run_root": str(run_root)}))
    except Exception as error:
        if run_root is not None:
            path = run_root / "pipeline_status.json"
            if path.is_file():
                status = read_json(path)
                if status.get("status") != "complete":
                    status["status"] = "failed"
                    status["error"] = repr(error)
                    status["traceback"] = traceback.format_exc()
                    status["failed_at"] = timestamp()
                    json_dump(path, status)
        raise


if __name__ == "__main__":
    main()
