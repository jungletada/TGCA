"""Frozen-model runtime helpers shared by the three Experiment 3 validations."""

from __future__ import annotations

import io
import json
import os
import platform
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis.lazy_assignment.experiment2.run_experiment2_signals import (
    region_code_maps,
)
from analysis.lazy_assignment.experiment2.voc_semantic_dataset import (
    VOCSemanticDataset,
)
from analysis.lazy_assignment.run_class_specific_patch_score import (
    checkpoint_payload,
    create_frozen_model,
    write_environment_manifests,
)
from analysis.lazy_assignment.experiment3.common import (
    EXPECTED_CLASSES,
    EXPECTED_IMAGES,
    EXPECTED_LAYERS,
    MODEL_FACTORY,
    assert_new_output,
    json_dump,
    read_json,
    require_tgca_repro,
    sha256_file,
    timestamp,
)


@dataclass(frozen=True)
class RuntimeInputs:
    source_metadata_path: Path
    source_metadata_sha256: str
    source_linkage_path: Path
    source_linkage_sha256: str
    immutable_manifest_path: Path
    immutable_manifest_sha256: str
    experiment2_root: Path
    experiment2_signal_root: Path
    signal_metadata_sha256: str
    signal_completion_sha256: str
    signal_manifest_sha256: str
    source_artifact_sha256: Mapping[str, str]
    checkpoint: Path
    checkpoint_sha256: str
    voc_root: Path
    list_path: Path
    labels_path: Path


class RunLog:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str) -> None:
        line = f"[{timestamp()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def resolve_runtime_inputs(source_metadata_path: Path, model: str) -> RuntimeInputs:
    require_tgca_repro()
    path = source_metadata_path.expanduser().resolve()
    metadata = read_json(path)
    if (
        metadata.get("status") != "complete"
        or metadata.get("integrity_passed") is not True
    ):
        raise RuntimeError(f"Experiment 3 input audit is not a completed PASS: {path}")
    if model not in MODEL_FACTORY:
        raise ValueError(f"unsupported model: {model}")
    checkpoints = metadata.get("checkpoints")
    roots = metadata.get("signal_roots")
    dataset = metadata.get("dataset")
    if not all(isinstance(value, Mapping) for value in (checkpoints, roots, dataset)):
        raise TypeError("input audit metadata has an invalid schema")
    record = checkpoints[model]  # type: ignore[index]
    if not isinstance(record, Mapping):
        raise TypeError(f"checkpoint record is invalid for {model}")
    checkpoint = Path(str(record["path"])).resolve()
    checkpoint_hash = str(record["actual_sha256"])
    if sha256_file(checkpoint) != checkpoint_hash:
        raise RuntimeError(f"checkpoint changed after input audit: {checkpoint}")
    signal_root = Path(str(roots[model])).resolve()  # type: ignore[index]
    linkage_path = Path(str(metadata["experiment2_linkage"])).resolve()
    linkage = read_json(linkage_path)
    linked_signals = linkage.get("signals")
    if not isinstance(linked_signals, Mapping):
        raise TypeError("input linkage lacks signal records")
    linked_signal = linked_signals.get(model)
    if not isinstance(linked_signal, Mapping):
        raise TypeError(f"input linkage lacks signal record for {model}")
    control_paths = {
        "metadata": signal_root / "metadata.json",
        "completion": signal_root / "completion.json",
        "manifest": signal_root / "manifest.jsonl",
    }
    control_hashes = {name: sha256_file(value) for name, value in control_paths.items()}
    for name, key in (
        ("metadata", "metadata_sha256"),
        ("completion", "completion_sha256"),
        ("manifest", "manifest_sha256"),
    ):
        if control_hashes[name] != str(linked_signal.get(key, "")):
            raise RuntimeError(f"Experiment 2 {model} {name} changed after audit")
    artifact_hashes: dict[str, str] = {}
    manifest_ids: list[str] = []
    for number, line in enumerate(
        control_paths["manifest"].read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise TypeError(f"invalid signal manifest row {number}")
        image_id = str(value.get("image_id", ""))
        if not image_id or image_id in artifact_hashes:
            raise RuntimeError(
                f"invalid/duplicate signal image at manifest row {number}"
            )
        manifest_ids.append(image_id)
        artifact_hashes[image_id] = str(value.get("artifact_sha256", ""))
    if len(manifest_ids) != EXPECTED_IMAGES:
        raise RuntimeError(f"signal manifest has {len(manifest_ids)} images")
    immutable = metadata.get("immutable_manifest")
    if not isinstance(immutable, Mapping):
        raise TypeError("input audit lacks immutable manifest metadata")
    immutable_path = Path(str(immutable["path"])).resolve()
    immutable_hash = str(immutable["sha256"])
    if sha256_file(immutable_path) != immutable_hash:
        raise RuntimeError("immutable input manifest changed after audit")
    voc_root = Path(str(dataset["voc_root"])).resolve()  # type: ignore[index]
    return RuntimeInputs(
        source_metadata_path=path,
        source_metadata_sha256=sha256_file(path),
        source_linkage_path=linkage_path,
        source_linkage_sha256=sha256_file(linkage_path),
        immutable_manifest_path=immutable_path,
        immutable_manifest_sha256=immutable_hash,
        experiment2_root=Path(str(metadata["experiment2_root"])).resolve(),
        experiment2_signal_root=signal_root,
        signal_metadata_sha256=control_hashes["metadata"],
        signal_completion_sha256=control_hashes["completion"],
        signal_manifest_sha256=control_hashes["manifest"],
        source_artifact_sha256=artifact_hashes,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        voc_root=voc_root,
        list_path=Path(str(dataset["list_path"])).resolve(),  # type: ignore[index]
        labels_path=Path(str(dataset["labels_path"])).resolve(),  # type: ignore[index]
    )


def create_runtime_model(model_key: str, inputs: RuntimeInputs, device: torch.device):
    require_tgca_repro()
    payload = checkpoint_payload(inputs.checkpoint)
    args = SimpleNamespace(model=MODEL_FACTORY[model_key], input_size=448)
    model, configuration, load_info = create_frozen_model(args, payload)
    model.to(device).eval()
    if (
        len(model.blocks) != EXPECTED_LAYERS
        or int(model.num_classes) != EXPECTED_CLASSES
    ):
        raise RuntimeError("Experiment 3 requires the native 12-layer, 20-class host")
    # The historical factory leaves requires_grad flags untouched.  Every
    # Experiment 3 caller enters torch.inference_mode(); changing flags here
    # would make the runtime state differ from the validated upstream path.
    return model, payload, configuration, load_info


def make_dataset_and_loader(
    inputs: RuntimeInputs,
    *,
    limit: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    pin_memory: bool,
) -> tuple[VOCSemanticDataset, DataLoader, int, int]:
    if batch_size != 8:
        raise ValueError("matched Experiment 1/2 reproduction requires batch size 8")
    if limit < 0 or num_workers < 0:
        raise ValueError("limit and num_workers must be non-negative")
    requested = min(limit, EXPECTED_IMAGES) if limit else EXPECTED_IMAGES
    # A smoke evaluates full surrounding batches so CUDA arithmetic follows
    # the same batching path as the immutable upstream runs.
    context = min(EXPECTED_IMAGES, int(np.ceil(requested / batch_size)) * batch_size)
    dataset = VOCSemanticDataset(
        inputs.voc_root,
        inputs.list_path,
        input_size=448,
        limit=context if limit else 0,
    )
    if len(dataset) != context:
        raise RuntimeError(
            f"dataset contains {len(dataset)} images, expected {context}"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(pin_memory and device.type == "cuda"),
        drop_last=False,
    )
    return dataset, loader, requested, context


def source_signal_path(inputs: RuntimeInputs, image_id: str) -> Path:
    if image_id not in inputs.source_artifact_sha256:
        raise KeyError(f"image is absent from audited signal manifest: {image_id}")
    path = inputs.experiment2_signal_root / "signals" / f"{image_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_source_signal(inputs: RuntimeInputs, image_id: str) -> dict[str, np.ndarray]:
    path = source_signal_path(inputs, image_id)
    actual = sha256_file(path)
    expected = inputs.source_artifact_sha256[image_id]
    if actual != expected:
        raise RuntimeError(f"source signal changed after input audit: {path}")
    with np.load(path, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    if str(payload["image_id"].item()) != image_id:
        raise RuntimeError(f"source signal image mismatch: {path}")
    return payload


def mask_region_codes(
    mask: torch.Tensor, positive: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from analysis.lazy_assignment.experiment2.patch_regions import patch_label_counts

    counts = patch_label_counts(mask, patch_size=16)
    return (
        counts.astype(np.uint16, copy=False),
        region_code_maps(counts, positive, 0.5, (28, 28)),
        region_code_maps(counts, positive, 0.7, (28, 28)),
    )


def save_npz_atomic(path: Path, payload: Mapping[str, np.ndarray]) -> str:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Build in memory so a failed validation cannot leave a partially valid
    # artifact at the final immutable path.
    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as output:
        output.write(stream.getvalue())
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    return sha256_file(path)


def reload_npz_checked(
    path: Path,
    *,
    expected_sha256: str,
    expected_image_id: str,
    allow_nan_keys: frozenset[str] = frozenset(),
) -> dict[str, np.ndarray]:
    """Fail closed on a just-written derived artifact before completion.

    Hash verification catches partial/replaced files; a complete NPZ reload
    catches corrupt archives and checks that derived floating signals are
    finite. Runner-specific code remains responsible for its exact key and
    shape contract.
    """

    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"derived artifact hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    image = payload.get("image_id")
    if image is None or image.shape != () or str(image.item()) != expected_image_id:
        raise RuntimeError(f"derived artifact image identity mismatch: {path}")
    for key, value in payload.items():
        if value.dtype.kind not in "fc":
            continue
        if key in allow_nan_keys:
            valid = not np.isinf(value).any()
        else:
            valid = np.isfinite(value).all()
        if not valid:
            raise RuntimeError(f"non-finite derived artifact values: {path}:{key}")
    return payload


def runtime_environment(device: torch.device) -> dict[str, object]:
    gpu = None
    if device.type == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": gpu,
    }


def initialize_run_directory(
    output_dir: Path,
    *,
    analysis: str,
    model: str,
    inputs: RuntimeInputs,
    execution: Mapping[str, object],
    git: Mapping[str, object],
) -> tuple[Path, dict[str, object], RunLog]:
    output = assert_new_output(
        output_dir,
        [
            inputs.source_metadata_path,
            inputs.experiment2_root,
            inputs.experiment2_signal_root,
            inputs.checkpoint,
            inputs.voc_root,
        ],
    )
    output.mkdir(parents=True, exist_ok=False)
    write_environment_manifests(output)
    command = shlex.join([sys.executable, *sys.argv])
    (output / "command.txt").write_text(command + "\n", encoding="utf-8")
    metadata: dict[str, object] = {
        "status": "running",
        "analysis": analysis,
        "model": model,
        "run_kind": "smoke" if int(execution.get("limit", 0)) else "full",
        "execution": dict(execution),
        "source_metadata": str(inputs.source_metadata_path),
        "source_metadata_sha256": inputs.source_metadata_sha256,
        "source_linkage": str(inputs.source_linkage_path),
        "source_linkage_sha256": inputs.source_linkage_sha256,
        "immutable_manifest": str(inputs.immutable_manifest_path),
        "immutable_manifest_sha256": inputs.immutable_manifest_sha256,
        "experiment2_signal_root": str(inputs.experiment2_signal_root),
        "experiment2_signal_control_sha256": {
            "metadata.json": inputs.signal_metadata_sha256,
            "completion.json": inputs.signal_completion_sha256,
            "manifest.jsonl": inputs.signal_manifest_sha256,
        },
        "checkpoint": {
            "path": str(inputs.checkpoint),
            "sha256": inputs.checkpoint_sha256,
        },
        "dataset": {
            "voc_root": str(inputs.voc_root),
            "list_path": str(inputs.list_path),
            "labels_path": str(inputs.labels_path),
            "input_size": 448,
            "patch_size": 16,
            "transform": "bicubic short-side Resize(512) -> CenterCrop(448) -> ImageNet normalize; nearest-neighbor matched mask",
        },
        "git": dict(git),
        "environment": None,
        "command": command,
        "started_at": timestamp(),
    }
    json_dump(output / "metadata.json", metadata)
    return output, metadata, RunLog(output / "run.log")


def assert_inputs_unchanged(inputs: RuntimeInputs) -> None:
    if sha256_file(inputs.source_metadata_path) != inputs.source_metadata_sha256:
        raise RuntimeError("Experiment 3 source metadata changed during inference")
    if sha256_file(inputs.checkpoint) != inputs.checkpoint_sha256:
        raise RuntimeError("checkpoint changed during inference")
    checks = {
        inputs.source_linkage_path: inputs.source_linkage_sha256,
        inputs.immutable_manifest_path: inputs.immutable_manifest_sha256,
        inputs.experiment2_signal_root / "metadata.json": inputs.signal_metadata_sha256,
        inputs.experiment2_signal_root
        / "completion.json": inputs.signal_completion_sha256,
        inputs.experiment2_signal_root
        / "manifest.jsonl": inputs.signal_manifest_sha256,
    }
    for path, expected in checks.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"immutable Experiment 3 input changed: {path}")


def finish_metadata(
    output: Path,
    metadata: dict[str, object],
    *,
    started: float,
    updates: Mapping[str, object],
) -> None:
    metadata.update(updates)
    metadata.update(
        {
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - started,
            "completed_at": timestamp(),
        }
    )
    json_dump(output / "metadata.json", metadata)
    completion = {
        "status": "complete",
        "analysis": metadata["analysis"],
        "model": metadata["model"],
        "run_kind": metadata["run_kind"],
        "num_images": updates.get("processed_images"),
        "completed_at": metadata["completed_at"],
    }
    json_dump(output / "completion.json", completion)
