from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


Arch = Literal["smp_deeplabv3plus", "smp_unetplusplus", "tv_deeplabv3_resnet50"]


@dataclass(frozen=True)
class ModelOutputs:
    logits: torch.Tensor


def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, dict) and "out" in out:
        return out["out"]
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def build_model(
    arch: Arch,
    num_classes: int,
    encoder: str = "resnet50",
    pretrained_backbone: bool = True,
) -> nn.Module:
    if arch.startswith("smp_"):
        import segmentation_models_pytorch as smp

        encoder_weights = "imagenet" if pretrained_backbone else None
        if arch == "smp_deeplabv3plus":
            return smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=num_classes,
                activation=None,
            )
        if arch == "smp_unetplusplus":
            return smp.UnetPlusPlus(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=num_classes,
                activation=None,
            )
        raise ValueError(f"Unknown arch: {arch}")

    if arch == "tv_deeplabv3_resnet50":
        from torchvision.models.segmentation import deeplabv3_resnet50
        from torchvision.models.segmentation.deeplabv3 import DeepLabHead

        model = deeplabv3_resnet50(weights="DEFAULT" if pretrained_backbone else None)
        model.classifier = DeepLabHead(2048, num_classes)
        if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
            model.aux_classifier = None
        return model

    raise ValueError(f"Unknown arch: {arch}")

