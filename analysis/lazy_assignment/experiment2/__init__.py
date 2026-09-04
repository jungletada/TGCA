"""Experiment 2 semantic-ownership analysis utilities.

This package is evaluation-only.  In particular, the dataset and region helpers
defined here do not modify model code, checkpoints, or Experiment 1 artifacts.
"""

from .patch_regions import (
    PAIR_REGION_CODE_TO_NAME,
    PAIR_REGION_NAME_TO_CODE,
    REGION_CODE_TO_NAME,
    REGION_NAME_TO_CODE,
    assign_pair_patch_regions,
    assign_pair_patch_regions_from_counts,
    assign_patch_regions,
    assign_patch_regions_from_counts,
    patch_label_counts,
)
from .voc_semantic_dataset import (
    Experiment2JointTransform,
    VOCSemanticDataset,
    build_joint_transform,
    resolve_semantic_mask_path,
)

__all__ = [
    "Experiment2JointTransform",
    "PAIR_REGION_CODE_TO_NAME",
    "PAIR_REGION_NAME_TO_CODE",
    "REGION_CODE_TO_NAME",
    "REGION_NAME_TO_CODE",
    "VOCSemanticDataset",
    "assign_pair_patch_regions",
    "assign_pair_patch_regions_from_counts",
    "assign_patch_regions",
    "assign_patch_regions_from_counts",
    "build_joint_transform",
    "patch_label_counts",
    "resolve_semantic_mask_path",
]
