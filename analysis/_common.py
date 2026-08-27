import json
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from models.adapter_modules import resize_input_minbound
from models.mctformer_plus import MCTformerPlusCam


def read_ids(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def load_labels(voc_root):
    return np.load(
        Path(voc_root) / "ImageLabel" / "cls_labels.npy", allow_pickle=True
    ).item()


def image_tensor(path, min_size, device):
    image = PIL.Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    tensor = resize_input_minbound(transform(image).unsqueeze(0), min_size=min_size)
    return image, tensor.to(device)


def checkpoint_bcss_config(checkpoint):
    return checkpoint.get("bcss", {
        "variant": "e0",
        "num_background_slots": 1,
        "tau": 0.5,
        "beta": 0.5,
        "class_threshold": 0.5,
    })


def load_cam_model(checkpoint_path, input_size, device, variant=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint_bcss_config(checkpoint)
    checkpoint_variant = config.get("variant", "e0")
    if variant is not None and variant != checkpoint_variant:
        raise ValueError(
            f"Checkpoint variant is {checkpoint_variant}, requested {variant}"
        )
    normalization = checkpoint.get("attention_normalization", {})
    model = MCTformerPlusCam(
        num_classes=20,
        input_size=input_size,
        attention_normalization=normalization.get("mode", "vanilla"),
        attention_gamma=normalization.get("gamma", 1.0),
        bcss_variant=checkpoint_variant,
        bcss_num_background_slots=config.get("num_background_slots", 1),
        bcss_tau=config.get("tau", 0.5),
        bcss_beta=config.get("beta", 0.5),
        bcss_cls_threshold=config.get("class_threshold", 0.5),
    )
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.set_bcss_epoch(8)
    model.to(device).eval()
    return model, config


def segmentation_path(voc_root, image_id):
    root = Path(voc_root)
    primary = root / "SegmentationClass" / f"{image_id}.png"
    if primary.is_file():
        return primary
    return root / "SegmentationClassAug" / f"{image_id}.png"


def load_segmentation(voc_root, image_id, size=None):
    path = segmentation_path(voc_root, image_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    mask = PIL.Image.open(path)
    if size is not None:
        mask = mask.resize((size[1], size[0]), resample=PIL.Image.NEAREST)
    return np.asarray(mask, dtype=np.uint8)


def dump_manifest(output_dir, payload):
    Path(output_dir).mkdir(parents=True, exist_ok=False)
    (Path(output_dir) / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def finite_mean(values):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": None, "std": None, "stderr": None, "count": 0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "stderr": float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0,
        "count": int(len(array)),
    }
