from __future__ import annotations

import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.detach().to(torch.int64).view(-1)
        targets = targets.detach().to(torch.int64).view(-1)

        valid = targets != self.ignore_index
        preds = preds[valid]
        targets = targets[valid]
        if preds.numel() == 0:
            return

        k = (targets * self.num_classes + preds).to(torch.int64)
        bincount = torch.bincount(k, minlength=self.num_classes**2)
        self.mat += bincount.view(self.num_classes, self.num_classes).cpu()

    def compute(self) -> dict[str, torch.Tensor]:
        mat = self.mat.to(torch.float64)
        diag = torch.diag(mat)
        sum_row = mat.sum(dim=1)
        sum_col = mat.sum(dim=0)
        denom = sum_row + sum_col - diag
        iou = torch.where(denom > 0, diag / denom, torch.zeros_like(denom))

        acc = diag.sum() / mat.sum().clamp_min(1.0)
        miou = iou.mean()
        return {"iou": iou, "miou": miou, "acc": acc}


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    targets = targets.to(torch.int64)
    valid = targets != ignore_index

    if valid.sum() == 0:
        return logits.sum() * 0.0

    probs = probs.permute(0, 2, 3, 1).contiguous()
    probs = probs[valid]
    targets = targets[valid]

    one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).to(probs.dtype)
    intersect = (probs * one_hot).sum(dim=0)
    union = probs.sum(dim=0) + one_hot.sum(dim=0)
    dice = (2.0 * intersect + eps) / (union + eps)
    return 1.0 - dice.mean()

