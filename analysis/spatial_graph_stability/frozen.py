"""Frozen official-LaST and PatchFinalLN model/data adapters."""

from __future__ import annotations

import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.transforms import transforms

from analysis.lazy_assignment.experiment2.voc_semantic_dataset import (
    VOCSemanticDataset,
)
from datasets_cam import load_img_name_list
from models.mctformer_plus import (
    build_mctformerplus,
    resolve_mctformerplus_checkpoint_variant,
    validate_mctformerplus_final_norm_checkpoint,
)

from .provenance import sha256_file


OFFICIAL_LAST_REPOSITORY = "https://github.com/ChengShiest/LAST-ViT"
OFFICIAL_LAST_COMMIT = "cdeb884af65e7774f2da80f666d95cf09a76b717"
OFFICIAL_LAST_CHECKPOINT_SHA256 = (
    "8d4593229dda2b8774a04210cb0f185efd3bd35e948ecb14036141ca7eed3253"
)
PATCH_FINAL_CHECKPOINT_SHA256 = (
    "a2a67b293aec37baea859e80a1e09f17b9fbe72afc2b87cbde22907638cc3227"
)


class VOCNaturalImageDataset(Dataset):
    """Deterministic VOC JPEG stream with the official LaST eval transform."""

    def __init__(
        self,
        voc_root: Path,
        list_path: Path,
        *,
        limit: int = 0,
    ) -> None:
        self.voc_root = voc_root.expanduser().resolve()
        self.list_path = list_path.expanduser().resolve()
        if not self.voc_root.is_dir() or not self.list_path.is_file():
            raise FileNotFoundError(f"VOC root/list missing: {self.voc_root}, {self.list_path}")
        ids = list(load_img_name_list(str(self.list_path)))
        if limit:
            ids = ids[: int(limit)]
        if not ids:
            raise ValueError("natural-image dataset is empty")
        self.image_ids = ids
        # This is the exact deterministic validation transform in the official
        # LaST config: Resize(256), CenterCrop(224), ImageNet normalization.
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_id = self.image_ids[index]
        path = self.voc_root / "JPEGImages" / f"{image_id}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        with PIL.Image.open(path) as source:
            image = source.convert("RGB")
        return {"name": image_id, "image": self.transform(image)}


def _official_repository_commit(repository: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(f"cannot inspect official LaST checkout: {process.stderr}")
    return process.stdout.strip()


def load_official_last(
    checkpoint_path: Path,
    repository: Path,
) -> tuple[VisionTransformer, dict[str, object]]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    repository = repository.expanduser().resolve()
    if not checkpoint_path.is_file() or not repository.is_dir():
        raise FileNotFoundError(f"missing official LaST input: {checkpoint_path}, {repository}")
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != OFFICIAL_LAST_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"official LaST checkpoint hash mismatch: {checkpoint_hash}"
        )
    repository_commit = _official_repository_commit(repository)
    if repository_commit != OFFICIAL_LAST_COMMIT:
        raise RuntimeError(
            f"official LaST checkout is {repository_commit}, expected {OFFICIAL_LAST_COMMIT}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw_state = checkpoint.get("model", checkpoint)
    if not isinstance(raw_state, dict):
        raise TypeError("official checkpoint lacks a model state dictionary")
    prefix = "model."
    if not raw_state or not all(str(key).startswith(prefix) for key in raw_state):
        raise RuntimeError("official state keys do not share the expected model. prefix")
    state = OrderedDict((key[len(prefix) :], value) for key, value in raw_state.items())
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=1000,
    )
    incompatibility = model.load_state_dict(state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"official LaST strict load failed: {incompatibility}")
    metadata = {
        "repository": OFFICIAL_LAST_REPOSITORY,
        "repository_path": str(repository),
        "repository_commit": repository_commit,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_iteration": checkpoint.get("iteration"),
        "strict_state_load": True,
        "architecture": {
            "image_size": 224,
            "patch_size": 16,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "mlp_dim": 3072,
            "num_classes": 1000,
        },
    }
    del checkpoint, raw_state, state
    return model, metadata


def official_last_tokens(
    model: VisionTransformer, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return final-normalized CLS, patches, and ordinary classifier logits."""

    patches = model._process_input(images)
    class_token = model.class_token.expand(images.shape[0], -1, -1)
    tokens = model.encoder(torch.cat((class_token, patches), dim=1))
    query = tokens[:, 0]
    patch_tokens = tokens[:, 1:]
    logits = model.heads(query)
    if patch_tokens.shape[1:] != (196, 768):
        raise RuntimeError(f"unexpected official LaST patch tokens {patch_tokens.shape}")
    return query, patch_tokens, logits


def load_patch_final_mctformer(
    checkpoint_path: Path,
) -> tuple[torch.nn.Module, dict[str, object]]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != PATCH_FINAL_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"PatchFinalLN checkpoint hash mismatch: {checkpoint_hash}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    resolution = resolve_mctformerplus_checkpoint_variant(checkpoint, "mctformerplus")
    if resolution["variant"] != "small":
        raise RuntimeError(f"expected MCTformer+-Small, got {resolution['variant']}")
    validate_mctformerplus_final_norm_checkpoint(
        checkpoint,
        expected=False,
        expected_patch=True,
        expected_last_mct=False,
        expected_class_stable_last=False,
    )
    attention = checkpoint.get("attention_normalization", {})
    bcss = checkpoint.get("bcss", {"variant": "e0"})
    psl = checkpoint.get("psl", {"variant": "baseline"})
    cti = checkpoint.get("cti_bgt", {"enabled": False})
    if (
        attention.get("mode", "vanilla") != "vanilla"
        or float(attention.get("gamma", 1.0)) != 1.0
        or bool(attention.get("relation_bias", False))
        or bcss.get("variant", "e0") != "e0"
        or psl.get("variant", "baseline") != "baseline"
        or bool(cti.get("enabled", False))
    ):
        raise RuntimeError("checkpoint is not the required vanilla/E0/native baseline")
    model = build_mctformerplus(
        "small",
        cam=True,
        num_classes=20,
        input_size=448,
        attention_normalization="vanilla",
        attention_gamma=1.0,
        bcss_variant="e0",
        psl_variant="baseline",
        cti_bgt=False,
        final_norm=False,
        patch_final_norm=True,
        last_mct=False,
        class_stable_last=False,
    )
    state = checkpoint.get("model", checkpoint)
    incompatibility = model.load_state_dict(state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"PatchFinalLN strict load failed: {incompatibility}")
    training = checkpoint.get("training_spec", {})
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "epoch": checkpoint.get("epoch"),
        "strict_state_load": True,
        "variant_resolution": resolution,
        "flags": {
            "final_norm": False,
            "patch_final_norm": True,
            "last_mct": False,
            "class_stable_last": False,
        },
        "training_spec": training,
        "head": {
            "kernel_size": list(model.head.kernel_size),
            "in_channels": int(model.head.in_channels),
            "out_channels": int(model.head.out_channels),
        },
    }
    if model.head.kernel_size != (3, 3) or model.head.in_channels != 384:
        raise RuntimeError("PatchFinalLN spatial classifier is not the expected 3x3 head")
    del checkpoint, state
    return model, metadata


def build_mct_dataset(
    voc_root: Path,
    list_path: Path,
    *,
    limit: int = 0,
) -> VOCSemanticDataset:
    return VOCSemanticDataset(
        voc_root.expanduser().resolve(),
        list_path.expanduser().resolve(),
        input_size=448,
        limit=int(limit),
    )
