#!/usr/bin/env python3
"""Render unified map dumps with raw, shared-within-metric color limits."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

from analysis._common import load_segmentation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--classes-per-image", type=int, default=2)
    return parser.parse_args()


def show_map(axis, values, title, vmax=None):
    axis.imshow(values, cmap="magma", vmin=0, vmax=vmax)
    axis.set_title(title, fontsize=8)
    axis.axis("off")


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    paths = sorted((args.dump_dir / "samples").glob("*.npz"))[:args.max_images]
    for path in paths:
        with np.load(path) as sample:
            image_id = str(sample["image_id"])
            label = sample["label"].astype(bool)
            class_ids = np.flatnonzero(label)[:args.classes_per_image]
            class_to_patch = sample["class_to_patch"]
            patch_cam = sample["patch_cam"]
            final_cam = sample["final_cam"]
            if "background_ownership" in sample:
                background = sample["background_ownership"]
                background_title = "Background ownership"
            elif "background_attention" in sample:
                background = sample["background_attention"].mean(axis=0)
                background_title = "Background query -> patch"
            else:
                background = None
                background_title = None
            register = sample["register_to_patch"] if "register_to_patch" in sample else None
        image = PIL.Image.open(
            args.voc_root / "JPEGImages" / f"{image_id}.jpg").convert("RGB")
        target = load_segmentation(args.voc_root, image_id)
        extra = int(background is not None) + int(register is not None)
        columns = 2 + 3 * len(class_ids) + extra
        figure, axes = plt.subplots(1, columns, figsize=(2.3 * columns, 2.6), constrained_layout=True)
        axes[0].imshow(image)
        axes[0].set_title("Input", fontsize=8)
        axes[0].axis("off")
        axes[1].imshow(target, cmap="tab20", vmin=0, vmax=20)
        axes[1].set_title("GT", fontsize=8)
        axes[1].axis("off")
        column = 2
        for class_id in class_ids:
            maps = (patch_cam[class_id], class_to_patch[class_id], final_cam[class_id])
            titles = (f"C{class_id} patch", f"C{class_id} c2p", f"C{class_id} final")
            for values, title in zip(maps, titles):
                show_map(axes[column], values, title, vmax=float(np.max(values)))
                column += 1
        if register is not None:
            show_map(axes[column], register, "REG query -> patch", vmax=float(register.max()))
            column += 1
        if background is not None:
            show_map(
                axes[column], background, background_title,
                vmax=1.0 if background_title == "Background ownership" else float(background.max()))
        figure.savefig(args.output_dir / f"{image_id}.png", dpi=180)
        figure.savefig(args.output_dir / f"{image_id}.pdf")
        plt.close(figure)


if __name__ == "__main__":
    main()
