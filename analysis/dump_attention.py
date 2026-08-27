#!/usr/bin/env python3
"""Export the unified VOC map contract for BCSS diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from analysis._common import dump_manifest, image_tensor, load_cam_model, load_labels, read_ids


MAP_KEYS = (
    "class_logits", "patch_cam", "class_to_patch", "final_cam",
    "patch_feature_norm", "class_ownership", "background_ownership",
    "background_raw_score", "background_attention", "register_to_patch",
    "patch_to_register", "background_to_patch", "patch_to_background", "ownership",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--id-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("e0", "e1", "e2", "e4", "e5", "e6"))
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-layer-head", action="store_true")
    return parser.parse_args()


def cpu_array(tensor):
    return tensor.detach().float().cpu().numpy()


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    device = torch.device(args.device)
    model, config = load_cam_model(
        args.checkpoint, args.input_size, device, args.variant)
    image_ids = read_ids(args.id_list)
    if args.max_images is not None:
        image_ids = image_ids[:args.max_images]
    labels = load_labels(args.voc_root)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    dump_manifest(args.output_dir, {
        "format_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "bcss": config,
        "input_size": args.input_size,
        "id_list": str(args.id_list.resolve()),
        "num_images": len(image_ids),
        "label_known_localization": True,
        "save_layer_head": args.save_layer_head,
        "possible_maps": list(MAP_KEYS),
    })
    sample_dir = args.output_dir / "samples"
    sample_dir.mkdir()
    started = time.time()
    with torch.no_grad():
        for index, image_id in enumerate(image_ids):
            image, inputs = image_tensor(
                args.voc_root / "JPEGImages" / f"{image_id}.jpg",
                args.input_size,
                device,
            )
            target = torch.from_numpy(labels[image_id]).to(device).unsqueeze(0)
            diagnostics = model(
                inputs, active_labels=target, return_diagnostics=True)
            arrays = {
                "image_id": np.asarray(image_id),
                "image_size": np.asarray((image.height, image.width), dtype=np.int32),
                "input_size": np.asarray(inputs.shape[-2:], dtype=np.int32),
                "label": labels[image_id].astype(np.uint8),
                "variant": np.asarray(config.get("variant", "e0")),
            }
            grid_size = tuple(diagnostics["patch_feature_norm"].shape[-2:])
            for key in MAP_KEYS:
                if key not in diagnostics:
                    continue
                if key in ("register_to_patch", "patch_to_register",
                           "background_to_patch", "patch_to_background"):
                    value = diagnostics[key][:, 0]
                    if args.save_layer_head:
                        value = value.reshape(*value.shape[:-1], *grid_size)
                    else:
                        value = value[-3:].mean(dim=(0, 1)).reshape(grid_size)
                else:
                    value = diagnostics[key][0]
                arrays[key] = cpu_array(value)
            if args.save_layer_head:
                arrays["class_to_patch_heads"] = cpu_array(
                    diagnostics["class_to_patch_heads"][:, 0])
                arrays["patch_to_class_heads"] = cpu_array(
                    diagnostics["patch_to_class_heads"][:, 0])
            np.savez_compressed(sample_dir / f"{image_id}.npz", **arrays)
            if (index + 1) % 25 == 0 or index + 1 == len(image_ids):
                print(f"dumped {index + 1}/{len(image_ids)}")
    completion = {
        "num_images": len(image_ids),
        "elapsed_seconds": time.time() - started,
        "complete": True,
    }
    (args.output_dir / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
