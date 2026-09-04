#!/usr/bin/env python3
"""Run Validation B CAM layer readouts with one matched frozen forward."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment2.evaluation_metrics import (  # noqa: E402
    upsample_and_normalize_active_cams,
)
from analysis.lazy_assignment.experiment2.native_cam_stages import (  # noqa: E402
    decompose_native_cam_reduced,
)
from analysis.lazy_assignment.experiment2.patch_regions import (  # noqa: E402
    patch_label_counts,
)
from analysis.lazy_assignment.experiment2.signal_collector import (  # noqa: E402
    SignalCollector,
    assert_no_change,
)
from analysis.lazy_assignment.experiment3.cam_layer_intervention import (  # noqa: E402
    CAM_VARIANT_SPECS,
    cam_threshold_grid,
    construct_all_cam_readouts,
)
from analysis.lazy_assignment.experiment3.common import (  # noqa: E402
    EXPECTED_CLASSES,
    EXPECTED_PATCHES,
    STRICT_TOLERANCE,
    enforce_production_source,
    json_dump,
    runtime_source_state,
    sha256_file,
)
from analysis.lazy_assignment.experiment3.runtime import (  # noqa: E402
    assert_inputs_unchanged,
    create_runtime_model,
    finish_metadata,
    initialize_run_directory,
    load_source_signal,
    make_dataset_and_loader,
    resolve_runtime_inputs,
    reload_npz_checked,
    runtime_environment,
    save_npz_atomic,
)


RUNTIME_SOURCES = (
    "analysis/lazy_assignment/experiment3/common.py",
    "analysis/lazy_assignment/experiment3/runtime.py",
    "analysis/lazy_assignment/experiment3/cam_layer_intervention.py",
    "analysis/lazy_assignment/experiment3/run_cam_layer_intervention.py",
    "analysis/lazy_assignment/experiment2/evaluation_metrics.py",
    "analysis/lazy_assignment/experiment2/native_cam_stages.py",
    "analysis/lazy_assignment/experiment2/patch_regions.py",
    "analysis/lazy_assignment/experiment2/signal_collector.py",
    "analysis/lazy_assignment/experiment2/voc_semantic_dataset.py",
    "analysis/lazy_assignment/run_class_specific_patch_score.py",
    "models/mctformer.py",
    "models/mctformer_plus.py",
    "models/vit.py",
    "models/tgca.py",
)
SOURCE_DIAGNOSTIC_KEYS = {
    "B0": "final_cam",
    "B1": "diagnostic_c2p_cam_l10",
    "B2": "diagnostic_c2p_cam_l11",
    "B3": "diagnostic_c2p_cam_l12",
    "B5": "diagnostic_c2p_cam_mid3",
}
DERIVED_KEYS = {
    "image_id",
    "positive_class_ids",
    "variant_codes",
    "thresholds",
    "attention_raw",
    "attention_conditional",
    "preprop_cam",
    "final_cam",
    "confusions",
    "source_signal_sha256",
}


def _validate_artifact_tree(
    output: Path,
    *,
    expected_image_ids: Sequence[str],
    source_hashes: Mapping[str, str],
) -> dict[str, object]:
    manifest_path = output / "manifest.jsonl"
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(record.get("image_id", "")) for record in records]
    if ids != list(expected_image_ids):
        raise RuntimeError("derived CAM manifest order/membership mismatch")
    expected_variants = np.asarray([spec.code for spec in CAM_VARIANT_SPECS])
    expected_thresholds = cam_threshold_grid()
    hashes: list[str] = []
    for record in records:
        image_id = str(record["image_id"])
        path = (output / str(record["artifact_path"])).resolve()
        try:
            path.relative_to(output.resolve())
        except ValueError as error:
            raise RuntimeError(f"derived CAM path escapes output: {path}") from error
        digest = str(record["artifact_sha256"])
        payload = reload_npz_checked(
            path,
            expected_sha256=digest,
            expected_image_id=image_id,
        )
        if set(payload) != DERIVED_KEYS:
            raise RuntimeError(f"derived CAM schema mismatch: {path}")
        positive = np.asarray(payload["positive_class_ids"], dtype=np.int64)
        p = len(positive)
        shapes = {
            "variant_codes": (len(CAM_VARIANT_SPECS),),
            "thresholds": (len(expected_thresholds),),
            "attention_raw": (len(CAM_VARIANT_SPECS), p, EXPECTED_PATCHES),
            "attention_conditional": (
                len(CAM_VARIANT_SPECS),
                p,
                EXPECTED_PATCHES,
            ),
            "preprop_cam": (len(CAM_VARIANT_SPECS), p, EXPECTED_PATCHES),
            "final_cam": (len(CAM_VARIANT_SPECS), p, EXPECTED_PATCHES),
            "confusions": (
                len(CAM_VARIANT_SPECS),
                len(expected_thresholds),
                EXPECTED_CLASSES + 1,
                EXPECTED_CLASSES + 1,
            ),
        }
        for key, shape in shapes.items():
            if payload[key].shape != shape:
                raise RuntimeError(
                    f"derived CAM shape mismatch {path}:{key}: "
                    f"{payload[key].shape} != {shape}"
                )
        if record.get("positive_class_ids") != positive.tolist():
            raise RuntimeError(f"derived CAM manifest class mismatch: {path}")
        if not np.array_equal(payload["variant_codes"], expected_variants):
            raise RuntimeError(f"derived CAM variant mismatch: {path}")
        if not np.array_equal(payload["thresholds"], expected_thresholds):
            raise RuntimeError(f"derived CAM threshold mismatch: {path}")
        if not np.allclose(
            payload["attention_conditional"].sum(axis=-1), 1.0, atol=1e-6, rtol=0
        ):
            raise RuntimeError(f"derived conditional attention invalid: {path}")
        if np.any(payload["attention_raw"] < 0) or np.any(payload["confusions"] < 0):
            raise RuntimeError(f"derived CAM artifact has negative mass/count: {path}")
        if str(payload["source_signal_sha256"].item()) != source_hashes[image_id]:
            raise RuntimeError(f"derived CAM source linkage mismatch: {path}")
        hashes.append(digest)
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "artifacts_reloaded": len(records),
        "artifact_hashes_reverified": len(hashes),
        "schema": sorted(DERIVED_KEYS),
        "passed": True,
    }


def _threshold_confusions(
    normalized_cam: np.ndarray,
    positive_class_ids: np.ndarray,
    mask: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Compute all thresholds from one confidence-bin histogram.

    A foreground prediction is accepted only when its normalized maximum is
    strictly greater than the background threshold.  This preserves the
    background-first tie rule without repeating 448x448 argmax 41 times.
    """

    values = np.asarray(normalized_cam)
    classes = np.asarray(positive_class_ids, dtype=np.int64)
    target = np.asarray(mask, dtype=np.int64)
    if values.ndim != 3 or values.shape[1:] != (448, 448):
        raise ValueError("normalized_cam must have shape [K,448,448]")
    if values.shape[0] != len(classes) or target.shape != (448, 448):
        raise ValueError("CAM/class/mask shapes disagree")
    if not np.isfinite(values).all() or not np.all(np.diff(thresholds) > 0):
        raise ValueError("CAM and threshold grid must be finite/ordered")
    comparison_thresholds = np.asarray(thresholds, dtype=values.dtype)
    if not np.isfinite(comparison_thresholds).all() or not np.all(
        np.diff(comparison_thresholds) > 0
    ):
        raise ValueError("threshold grid is invalid in the CAM comparison dtype")
    foreground_offset = values.argmax(axis=0)
    confidence = values.max(axis=0)
    predicted_foreground = classes[foreground_offset] + 1
    valid = target != 255
    gt = target[valid]
    pred = predicted_foreground[valid]
    # Number of grid thresholds strictly below each confidence.  Exact ties do
    # not pass and therefore remain background at that threshold.
    pass_count = np.searchsorted(comparison_thresholds, confidence[valid], side="left")
    bins = len(thresholds) + 1
    encoded = (gt * 21 + pred) * bins + pass_count
    histogram = np.bincount(encoded, minlength=21 * 21 * bins).reshape(21, 21, bins)
    cumulative = histogram[:, :, ::-1].cumsum(axis=2)[:, :, ::-1]
    total_target = np.bincount(gt, minlength=21).astype(np.int64, copy=False)
    output = np.zeros((len(thresholds), 21, 21), dtype=np.int64)
    for index in range(len(thresholds)):
        foreground = cumulative[:, :, index + 1]
        output[index] = foreground
        output[index, :, 0] = total_target - foreground.sum(axis=1)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("mctformer", "mctformer_plus"), required=True
    )
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--allow-uncommitted-source", action="store_true")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.batch_size != 8 or args.num_workers < 0 or args.limit < 0:
        raise ValueError("batch-size must be 8; workers/limit must be non-negative")
    return args


def execute(args: argparse.Namespace) -> None:
    inputs = resolve_runtime_inputs(args.source_metadata, args.model)
    state = runtime_source_state(RUNTIME_SOURCES)
    enforce_production_source(
        state,
        allow_uncommitted=bool(args.allow_uncommitted_source),
        limit=int(args.limit),
    )
    output, metadata, log = initialize_run_directory(
        args.output_dir,
        analysis="experiment3_validation_b_cam_layer_readout",
        model=args.model,
        inputs=inputs,
        execution={
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "limit": args.limit,
            "seed": args.seed,
            "allow_uncommitted_source": bool(args.allow_uncommitted_source),
        },
        git=state,
    )
    started = time.perf_counter()
    head_outputs: list[torch.Tensor] = []
    head_handle = None
    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        metadata["environment"] = runtime_environment(device)
        model, checkpoint, configuration, load_info = create_runtime_model(
            args.model, inputs, device
        )
        metadata["model_configuration"] = configuration
        metadata["strict_checkpoint_load"] = load_info
        metadata["checkpoint_epoch"] = (
            checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None
        )
        dataset, loader, requested, context = make_dataset_and_loader(
            inputs,
            limit=args.limit,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            pin_memory=not args.no_pin_memory,
        )
        metadata["execution"]["requested_images"] = requested
        metadata["execution"]["batch_context_images"] = context
        thresholds = cam_threshold_grid()
        metadata["cam_contract"] = {
            "variant_order": [spec.code for spec in CAM_VARIANT_SPECS],
            "variants": {
                spec.code: {
                    "name": spec.name,
                    "layers_one_based": list(spec.layers_one_based),
                }
                for spec in CAM_VARIANT_SPECS
            },
            "attention": "raw head-mean global-softmax A_c2p; never conditionalized",
            "mctformer": "selected-layer sum, A_c2p * ReLU(patch logits), all-layer P2P sum",
            "mctformer_plus": "selected-layer mean, sqrt(A_c2p * ReLU(patch logits)), all-layer P2P sum",
            "thresholds": thresholds.tolist(),
            "primary_threshold": 0.45,
            "prediction": "GT-positive gating; bilinear 28->448 align_corners=False; per-class minmax; background-first ties",
        }

        guard = dataset[0]["image"].unsqueeze(0).to(device)
        with torch.inference_mode():
            plain = model(guard)
            with SignalCollector(
                model, num_classes=EXPECTED_CLASSES
            ) as guard_collector:
                guard_collector.clear(expected_num_patches=EXPECTED_PATCHES)
                observed = model(guard)
                guard_collector.consume()
        guard_difference = assert_no_change(plain, observed, tolerance=0.0)
        metadata["first_image_no_change_guard"] = {
            "native_output_max_abs_diff": guard_difference,
            "passed": guard_difference == 0.0,
        }
        del guard, plain, observed

        def head_hook(_module, _inputs, value):
            if not isinstance(value, torch.Tensor):
                raise TypeError("patch head output must be a tensor")
            head_outputs.append(value.detach())
            return None

        head_handle = model.head.register_forward_hook(head_hook)
        signal_dir = output / "cams"
        signal_dir.mkdir()
        processed = 0
        maximum_native_difference = 0.0
        maximum_source_patch_difference = 0.0
        maximum_source_variant_difference = {
            code: 0.0 for code in SOURCE_DIAGNOSTIC_KEYS
        }
        maximum_mask_count_difference = 0
        with (output / "manifest.jsonl").open("x", encoding="utf-8") as manifest:
            with SignalCollector(model, num_classes=EXPECTED_CLASSES) as collector:
                with torch.inference_mode():
                    for batch_number, batch in enumerate(loader):
                        images = batch["image"].to(
                            device,
                            non_blocking=(
                                device.type == "cuda" and not args.no_pin_memory
                            ),
                        )
                        image_ids = list(batch["name"])
                        labels = batch["label"].cpu().numpy()
                        masks = batch["mask"]
                        head_outputs.clear()
                        collector.clear(expected_num_patches=EXPECTED_PATCHES)
                        native_cam = model(images)
                        capture = collector.consume()
                        if len(head_outputs) != 1:
                            raise RuntimeError(
                                f"patch head fired {len(head_outputs)} times"
                            )
                        patch_logits = head_outputs[0]
                        native_stages = decompose_native_cam_reduced(
                            args.model,
                            patch_logits,
                            capture.attn_c2p_raw,
                            capture.patch_to_patch_sum,
                            num_classes=EXPECTED_CLASSES,
                        )
                        native_difference = float(
                            (native_stages["final_cam"] - native_cam).abs().max().item()
                        )
                        if native_difference >= STRICT_TOLERANCE:
                            raise RuntimeError(
                                f"native CAM reproduction failed: {native_difference}"
                            )
                        maximum_native_difference = max(
                            maximum_native_difference, native_difference
                        )
                        readouts = construct_all_cam_readouts(
                            args.model,
                            native_stages["patch_cam"],
                            capture.attn_c2p_raw,
                            capture.patch_to_patch_sum,
                        )
                        for local, image_id in enumerate(image_ids):
                            if processed >= requested:
                                break
                            positive = np.flatnonzero(labels[local] > 0).astype(
                                np.int64
                            )
                            positive_index = torch.as_tensor(
                                positive, dtype=torch.long, device=device
                            )
                            source = load_source_signal(inputs, image_id)
                            counts = patch_label_counts(masks[local], patch_size=16)
                            count_difference = int(
                                np.max(
                                    np.abs(
                                        counts.astype(np.int32)
                                        - source["patch_label_counts"].astype(np.int32)
                                    )
                                )
                            )
                            maximum_mask_count_difference = max(
                                maximum_mask_count_difference, count_difference
                            )
                            if count_difference:
                                raise RuntimeError(
                                    f"transformed-mask mismatch for {image_id}"
                                )
                            patch_positive = (
                                native_stages["patch_cam"][local]
                                .index_select(0, positive_index)
                                .flatten(1)
                            )
                            patch_difference = float(
                                (
                                    patch_positive.float()
                                    - torch.as_tensor(
                                        source["patch_cam"], device=device
                                    ).float()
                                )
                                .abs()
                                .max()
                                .item()
                            )
                            maximum_source_patch_difference = max(
                                maximum_source_patch_difference, patch_difference
                            )
                            if patch_difference >= STRICT_TOLERANCE:
                                raise RuntimeError(
                                    f"patch CAM source mismatch for {image_id}: {patch_difference}"
                                )

                            raw_attention = []
                            conditional_attention = []
                            preprop = []
                            final = []
                            confusions = []
                            for spec in CAM_VARIANT_SPECS:
                                stage = readouts[spec.code]
                                raw = stage.raw_c2p[local].index_select(
                                    0, positive_index
                                )
                                conditional = raw / raw.sum(dim=-1, keepdim=True)
                                c2p = (
                                    stage.preprop_cam[local]
                                    .index_select(0, positive_index)
                                    .flatten(1)
                                )
                                result = (
                                    stage.final_cam[local]
                                    .index_select(0, positive_index)
                                    .flatten(1)
                                )
                                raw_attention.append(raw)
                                conditional_attention.append(conditional)
                                preprop.append(c2p)
                                final.append(result)
                                normalized = upsample_and_normalize_active_cams(
                                    result.reshape(len(positive), 28, 28)
                                )
                                if isinstance(normalized, torch.Tensor):
                                    normalized_np = normalized.cpu().numpy()
                                else:
                                    normalized_np = np.asarray(normalized)
                                confusions.append(
                                    _threshold_confusions(
                                        normalized_np,
                                        positive,
                                        masks[local].cpu().numpy(),
                                        thresholds,
                                    )
                                )
                                if spec.code in SOURCE_DIAGNOSTIC_KEYS:
                                    reference = torch.as_tensor(
                                        source[SOURCE_DIAGNOSTIC_KEYS[spec.code]],
                                        device=device,
                                    )
                                    difference = float(
                                        (result.float() - reference.float())
                                        .abs()
                                        .max()
                                        .item()
                                    )
                                    maximum_source_variant_difference[spec.code] = max(
                                        maximum_source_variant_difference[spec.code],
                                        difference,
                                    )
                                    if difference >= STRICT_TOLERANCE:
                                        raise RuntimeError(
                                            f"{spec.code} source equivalence failed for {image_id}: {difference}"
                                        )
                            payload = {
                                "image_id": np.asarray(image_id),
                                "positive_class_ids": positive,
                                "variant_codes": np.asarray(
                                    [spec.code for spec in CAM_VARIANT_SPECS]
                                ),
                                "thresholds": thresholds,
                                "attention_raw": torch.stack(raw_attention)
                                .float()
                                .cpu()
                                .numpy(),
                                "attention_conditional": torch.stack(
                                    conditional_attention
                                )
                                .float()
                                .cpu()
                                .numpy(),
                                "preprop_cam": torch.stack(preprop)
                                .float()
                                .cpu()
                                .numpy(),
                                "final_cam": torch.stack(final).float().cpu().numpy(),
                                "confusions": np.stack(confusions).astype(
                                    np.int64, copy=False
                                ),
                                "source_signal_sha256": np.asarray(
                                    sha256_file(
                                        inputs.experiment2_signal_root
                                        / "signals"
                                        / f"{image_id}.npz"
                                    )
                                ),
                            }
                            path = signal_dir / f"{image_id}.npz"
                            digest = save_npz_atomic(path, payload)
                            manifest.write(
                                json.dumps(
                                    {
                                        "image_id": image_id,
                                        "positive_class_ids": positive.tolist(),
                                        "artifact_path": str(path.relative_to(output)),
                                        "artifact_sha256": digest,
                                    },
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            manifest.flush()
                            processed += 1
                        log(f"batch={batch_number + 1} images={processed}/{requested}")
                        del (
                            images,
                            native_cam,
                            capture,
                            patch_logits,
                            native_stages,
                            readouts,
                        )

        if processed != requested or len(list(signal_dir.glob("*.npz"))) != requested:
            raise RuntimeError(f"saved {processed} CAM artifacts, expected {requested}")
        if maximum_mask_count_difference != 0:
            raise RuntimeError("semantic mask/count reproduction failed")
        artifact_validation = _validate_artifact_tree(
            output,
            expected_image_ids=[str(value) for value in dataset.image_ids[:requested]],
            source_hashes=inputs.source_artifact_sha256,
        )
        assert_inputs_unchanged(inputs)
        finish_metadata(
            output,
            metadata,
            started=started,
            updates={
                "processed_images": processed,
                "native_cam_max_abs_diff": maximum_native_difference,
                "source_patch_cam_max_abs_diff": maximum_source_patch_difference,
                "source_variant_final_cam_max_abs_diff": maximum_source_variant_difference,
                "mask_patch_count_max_abs_diff": maximum_mask_count_difference,
                "manifest": "manifest.jsonl",
                "cams": "cams",
                "derived_artifact_validation": artifact_validation,
            },
        )
        log(
            f"complete images={processed} native_diff={maximum_native_difference:.3g} "
            f"source_diffs={maximum_source_variant_difference}"
        )
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        json_dump(output / "metadata.json", metadata)
        raise
    finally:
        if head_handle is not None:
            head_handle.remove()


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
