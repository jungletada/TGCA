"""Background-Aware Competitive Semantic Slots (BCSS).

This module is intentionally independent from the ViT attention implementation.
BCSS reads the frozen token roles produced by the encoder and never writes slot
features back into the patch stream.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BCSSVariantSpec:
    backbone_register: bool = False
    backbone_background: bool = False
    independent_background: bool = False
    competitive_ownership: bool = False
    foreground_anchor: bool = False
    foreground_mass_anchor: bool = False
    background_null: bool = False
    slot_update: bool = False


BCSS_VARIANTS: Dict[str, BCSSVariantSpec] = {
    "e0": BCSSVariantSpec(),
    "e1": BCSSVariantSpec(backbone_register=True),
    "e2": BCSSVariantSpec(
        backbone_background=True,
        independent_background=True,
    ),
    "e4": BCSSVariantSpec(
        competitive_ownership=True,
        foreground_anchor=True,
    ),
    "e4_mass": BCSSVariantSpec(
        competitive_ownership=True,
        foreground_anchor=True,
        foreground_mass_anchor=True,
    ),
    "e5": BCSSVariantSpec(
        competitive_ownership=True,
        foreground_anchor=True,
        background_null=True,
    ),
    "e6": BCSSVariantSpec(
        competitive_ownership=True,
        foreground_anchor=True,
        background_null=True,
        slot_update=True,
    ),
}


def validate_bcss_variant(variant: str) -> BCSSVariantSpec:
    try:
        return BCSS_VARIANTS[variant.lower()]
    except KeyError as exc:
        supported = ", ".join(BCSS_VARIANTS)
        raise ValueError(f"Unsupported BCSS variant {variant!r}; choose from {supported}") from exc


def bcss_schedule(epoch: int, final_tau: float, final_beta: float) -> Dict[str, float]:
    """Return the prespecified BCSS warm-up schedule.

    Epochs 0--2 leave CAM gating and slot refinement disabled. Epochs 3--8
    linearly ramp them while temperature moves from 1 to its final value.
    """
    if epoch <= 2:
        progress = 0.0
    elif epoch >= 8:
        progress = 1.0
    else:
        progress = (epoch - 2) / 6.0
    return {
        "progress": progress,
        "tau": 1.0 + progress * (final_tau - 1.0),
        "beta": progress * final_beta,
        "refinement_strength": progress,
    }


def infer_active_classes(class_logits: torch.Tensor, threshold: float) -> torch.Tensor:
    active = torch.sigmoid(class_logits) > threshold
    no_active = ~active.any(dim=1)
    if no_active.any():
        top1 = class_logits.argmax(dim=1)
        active[no_active] = False
        active[no_active, top1[no_active]] = True
    return active


def ownership_calibrate_attention(
    class_to_patch: torch.Tensor,
    class_ownership: torch.Tensor,
    beta: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Redistribute each class-to-patch row without changing its total mass.

    Retaining the original row mass makes beta=0 numerically identical to the
    MCTformer+ path and avoids introducing a hidden attention-scale confound.
    """
    if beta == 0:
        return class_to_patch
    original_mass = class_to_patch.sum(dim=-1, keepdim=True)
    weighted = class_to_patch * (class_ownership.to(class_to_patch.dtype) + eps).pow(beta)
    return weighted / weighted.sum(dim=-1, keepdim=True).clamp_min(eps) * original_mass


class SemanticSlotDecoder(nn.Module):
    """One-way semantic slot-to-patch decoder used by E4--E6."""

    def __init__(
        self,
        dim: int,
        num_classes: int,
        num_background_slots: int = 1,
        enable_slot_update: bool = False,
    ) -> None:
        super().__init__()
        if num_background_slots < 1:
            raise ValueError("BCSS requires at least one background slot")
        self.dim = dim
        self.num_classes = num_classes
        self.num_background_slots = num_background_slots
        self.enable_slot_update = enable_slot_update

        self.background_slots = nn.Parameter(
            torch.empty(1, num_background_slots, dim)
        )
        self.slot_norm = nn.LayerNorm(dim)
        self.patch_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)

        if enable_slot_update:
            self.update_norm = nn.LayerNorm(dim)
            self.update_mlp = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
            )
            self.update_gate = nn.Parameter(torch.zeros(()))
        else:
            self.update_norm = None
            self.update_mlp = None
            self.register_parameter("update_gate", None)

        nn.init.trunc_normal_(self.background_slots, std=0.02)

    def _energies(self, slots: torch.Tensor, patch_keys: torch.Tensor) -> torch.Tensor:
        queries = self.q_proj(self.slot_norm(slots))
        return torch.matmul(queries.float(), patch_keys.float().transpose(-2, -1)) / math.sqrt(self.dim)

    @staticmethod
    def _aggregate(ownership: torch.Tensor, patch_values: torch.Tensor) -> torch.Tensor:
        spatial = ownership / ownership.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.matmul(spatial.to(patch_values.dtype), patch_values)

    def forward(
        self,
        class_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
        active_classes: torch.Tensor,
        tau: float,
        competitive: bool,
        refinement_strength: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if tau <= 0:
            raise ValueError("BCSS temperature must be positive")
        batch = class_tokens.shape[0]
        active_classes = active_classes.to(device=class_tokens.device, dtype=torch.bool)
        background = self.background_slots.expand(batch, -1, -1)
        patch_keys = self.k_proj(self.patch_norm(patch_tokens))
        patch_values = self.v_proj(self.patch_norm(patch_tokens))

        if not competitive:
            raw = self._energies(background, patch_keys)
            attention = torch.softmax(raw / tau, dim=-1)
            aggregated = self._aggregate(attention, patch_values)
            return {
                "background_raw_score": raw,
                "background_attention": attention,
                "background_aggregate": aggregated,
            }

        slots = torch.cat((class_tokens, background), dim=1)
        ownership, energies = self._competitive_pass(
            slots, patch_keys, active_classes, tau
        )
        aggregates = self._aggregate(ownership, patch_values)

        if self.enable_slot_update:
            update = self.update_mlp(self.update_norm(aggregates))
            slots = slots + refinement_strength * self.update_gate * update
            ownership, energies = self._competitive_pass(
                slots, patch_keys, active_classes, tau
            )
            aggregates = self._aggregate(ownership, patch_values)

        class_ownership = ownership[:, :self.num_classes]
        background_ownership_slots = ownership[:, self.num_classes:]
        return {
            "energies": energies,
            "ownership": ownership,
            "class_ownership": class_ownership,
            "background_ownership_slots": background_ownership_slots,
            "background_ownership": background_ownership_slots.sum(dim=1),
            "class_aggregate": aggregates[:, :self.num_classes],
            "background_aggregate": aggregates[:, self.num_classes:],
            "active_classes": active_classes,
        }

    def _competitive_pass(
        self,
        slots: torch.Tensor,
        patch_keys: torch.Tensor,
        active_classes: torch.Tensor,
        tau: float,
    ):
        energies = self._energies(slots, patch_keys)
        class_energies = energies[:, :self.num_classes]
        class_energies = class_energies.masked_fill(
            ~active_classes.unsqueeze(-1), torch.finfo(class_energies.dtype).min
        )
        masked = torch.cat((class_energies, energies[:, self.num_classes:]), dim=1)
        ownership = torch.softmax(masked / tau, dim=1)
        ownership = ownership.masked_fill(
            torch.cat(
                (
                    ~active_classes,
                    torch.zeros(
                        active_classes.shape[0],
                        self.num_background_slots,
                        dtype=torch.bool,
                        device=active_classes.device,
                    ),
                ),
                dim=1,
            ).unsqueeze(-1),
            0.0,
        )
        return ownership, masked


def semantic_slot_losses(
    auxiliary: Dict[str, torch.Tensor],
    classifier_weight: torch.Tensor,
    classifier_bias: Optional[torch.Tensor],
    targets: torch.Tensor,
    use_foreground_anchor: bool,
    use_background_null: bool,
    retain_foreground_ownership_mass: bool = False,
    semantic_temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute only the losses enabled by a frozen experiment variant."""
    losses: Dict[str, torch.Tensor] = {}
    if use_foreground_anchor:
        class_logits = F.linear(
            auxiliary["class_aggregate"], classifier_weight, classifier_bias
        ) / semantic_temperature
        if retain_foreground_ownership_mass:
            ownership_mass = auxiliary["class_ownership"].float().mean(dim=-1)
            class_logits = class_logits * ownership_mass.to(class_logits.dtype).unsqueeze(-1)
        batch, classes, _ = class_logits.shape
        target_ids = torch.arange(classes, device=class_logits.device).expand(batch, -1)
        per_slot = F.cross_entropy(
            class_logits.reshape(-1, classes), target_ids.reshape(-1), reduction="none"
        ).reshape(batch, classes)
        active = targets.to(per_slot.dtype)
        losses["foreground_anchor"] = (
            (per_slot * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)
        ).mean()

    if use_background_null:
        background_logits = F.linear(
            auxiliary["background_aggregate"], classifier_weight, classifier_bias
        )
        losses["background_null"] = F.softplus(background_logits).mean()
    return losses
