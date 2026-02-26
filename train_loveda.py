from __future__ import annotations

import argparse
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from PIL import Image

from loveda_seg.data import LoveDADataset, TrainTransform, ValTransform
from loveda_seg.metrics import ConfusionMatrix, dice_loss
from loveda_seg.models import Arch, build_model, forward_logits


@dataclass(frozen=True)
class TrainState:
    epoch: int
    step: int
    best_miou: float


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def freeze_batchnorm(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters(recurse=False):
                p.requires_grad = False



def poly_lr(base_lr: float, step: int, total_steps: int, power: float = 0.9) -> float:
    if total_steps <= 1:
        return base_lr
    return base_lr * (1.0 - step / total_steps) ** power


COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)


def loveda_colors(num_classes: int) -> np.ndarray:
    base = np.array(list(COLOR_MAP.values()), dtype=np.uint8)
    if num_classes <= base.shape[0]:
        return base[:num_classes].copy()
    extra = np.random.RandomState(0).randint(0, 255, size=(num_classes - base.shape[0], 3), dtype=np.uint8)
    return np.concatenate([base, extra], axis=0).copy()


def colorize_mask(mask: np.ndarray, colors: np.ndarray, ignore_index: int, ignore_color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    valid = mask != ignore_index
    idx = np.clip(mask, 0, colors.shape[0] - 1)
    out[valid] = colors[idx[valid]]
    out[~valid] = np.array(ignore_color, dtype=np.uint8)
    return out


def error_heatmap_wrong(gt: np.ndarray, pred: np.ndarray, probs: np.ndarray, ignore_index: int) -> np.ndarray:
    valid = gt != ignore_index
    wrong = valid & (gt != pred)

    h, w = gt.shape
    yy, xx = np.indices((h, w))
    gt_safe = gt.copy()
    gt_safe[~valid] = 0
    p_gt = probs[gt_safe, yy, xx]

    heat = np.zeros((h, w), dtype=np.float32)
    heat[wrong] = (1.0 - p_gt[wrong]).astype(np.float32)
    return heat


def error_heatmap_fpfn(gt: np.ndarray, pred: np.ndarray, probs: np.ndarray, ignore_index: int, background: int = 0) -> np.ndarray:
    valid = gt != ignore_index
    fp = valid & (gt == background) & (pred != background)
    fn = valid & (gt != background) & (pred == background)

    prob_bg = probs[background].astype(np.float32)
    prob_non_bg = (1.0 - prob_bg).astype(np.float32)

    heat = np.zeros(gt.shape, dtype=np.float32)
    heat[fp] = prob_non_bg[fp]
    heat[fn] = -prob_bg[fn]
    return heat


def save_epoch_visuals(
    model: nn.Module,
    val_ds: LoveDADataset,
    vis_indices: list[int],
    out_dir: Path,
    num_classes: int,
    crop_size: int,
    stride: int,
    device: torch.device,
    amp: bool,
    ignore_index: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = loveda_colors(num_classes)
    model.eval()

    rows: list[tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = []

    for k, ds_idx in enumerate(vis_indices):
        pair = val_ds.pairs[ds_idx]
        rgb = Image.open(pair.image_path).convert("RGB")
        gt_img = Image.open(pair.mask_path)
        gt = np.array(gt_img, dtype=np.int64)
        gt[gt == 7] = ignore_index

        img_t, _ = ValTransform(ignore_index=ignore_index)(rgb, gt_img)
        img_t = img_t.to(device)

        if img_t.shape[-1] > crop_size or img_t.shape[-2] > crop_size:
            probs = sliding_window_logits(
                model=model,
                image=img_t,
                crop_size=crop_size,
                stride=stride,
                num_classes=num_classes,
                device=device,
                amp=amp,
            )
            probs_np = probs.to(torch.float32).cpu().numpy()
            pred = probs_np.argmax(axis=0).astype(np.int64)
        else:
            with autocast(enabled=amp):
                logits = forward_logits(model, img_t.unsqueeze(0))
            probs_np = torch.softmax(logits.squeeze(0), dim=0).to(torch.float32).cpu().numpy()
            pred = probs_np.argmax(axis=0).astype(np.int64)

        gt_color = colorize_mask(gt, colors, ignore_index=ignore_index)
        pred_color = colorize_mask(pred, colors, ignore_index=ignore_index)
        wrong_heat = error_heatmap_wrong(gt, pred, probs_np, ignore_index=ignore_index)
        fpfn_heat = error_heatmap_fpfn(gt, pred, probs_np, ignore_index=ignore_index, background=0)

        stem = pair.image_path.stem
        rows.append((rgb, gt_color, pred_color, wrong_heat, fpfn_heat, stem))

        fig = plt.figure(figsize=(12, 8), dpi=150)
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(rgb)
        ax1.set_title("image")
        ax1.axis("off")

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.imshow(gt_color)
        ax2.set_title("gt")
        ax2.axis("off")

        ax3 = fig.add_subplot(2, 3, 3)
        ax3.imshow(pred_color)
        ax3.set_title("pred")
        ax3.axis("off")

        ax4 = fig.add_subplot(2, 3, 4)
        ax4.imshow(wrong_heat, cmap="jet", vmin=0.0, vmax=1.0)
        ax4.set_title("error heat: wrong (0..1)")
        ax4.axis("off")

        ax5 = fig.add_subplot(2, 3, 5)
        ax5.imshow(fpfn_heat, cmap="bwr", vmin=-1.0, vmax=1.0)
        ax5.set_title("error heat: fp(+)/fn(-)")
        ax5.axis("off")

        ax6 = fig.add_subplot(2, 3, 6)
        overlay = np.array(rgb, dtype=np.float32).copy()
        alpha = np.clip(wrong_heat, 0.0, 1.0) * 0.75
        overlay[..., 0] = overlay[..., 0] * (1.0 - alpha) + 255.0 * alpha
        overlay = np.clip(overlay, 0.0, 255.0).astype(np.uint8)
        ax6.imshow(overlay)
        ax6.set_title("overlay wrong heat")
        ax6.axis("off")

        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{k:02d}_{stem}.png")
        plt.close(fig)

        plt.imsave(out_dir / f"sample_{k:02d}_{stem}_error_wrong_heat.png", wrong_heat, cmap="jet", vmin=0.0, vmax=1.0)
        plt.imsave(out_dir / f"sample_{k:02d}_{stem}_error_fpfn_heat.png", fpfn_heat, cmap="bwr", vmin=-1.0, vmax=1.0)

    if rows:
        fig = plt.figure(figsize=(18, 3.5 * len(rows)), dpi=150)
        for r, (rgb, gt_color, pred_color, wrong_heat, fpfn_heat, stem) in enumerate(rows):
            ax = fig.add_subplot(len(rows), 5, r * 5 + 1)
            ax.imshow(rgb)
            ax.set_title(f"image ({stem})")
            ax.axis("off")

            ax = fig.add_subplot(len(rows), 5, r * 5 + 2)
            ax.imshow(gt_color)
            ax.set_title("gt")
            ax.axis("off")

            ax = fig.add_subplot(len(rows), 5, r * 5 + 3)
            ax.imshow(pred_color)
            ax.set_title("pred")
            ax.axis("off")

            ax = fig.add_subplot(len(rows), 5, r * 5 + 4)
            ax.imshow(wrong_heat, cmap="jet", vmin=0.0, vmax=1.0)
            ax.set_title("wrong heat")
            ax.axis("off")

            ax = fig.add_subplot(len(rows), 5, r * 5 + 5)
            ax.imshow(fpfn_heat, cmap="bwr", vmin=-1.0, vmax=1.0)
            ax.set_title("fp/fn heat")
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(out_dir / "epoch_grid.png")
        plt.close(fig)


@torch.no_grad()
def sliding_window_logits(
    model: nn.Module,
    image: torch.Tensor,
    crop_size: int,
    stride: int,
    num_classes: int,
    device: torch.device,
    amp: bool = True,
) -> torch.Tensor:
    _, h, w = image.shape
    out_probs = torch.zeros((num_classes, h, w), device=device, dtype=torch.float32)
    count = torch.zeros((1, h, w), device=device, dtype=torch.float32)

    y_steps = list(range(0, max(h - crop_size, 0) + 1, stride))
    x_steps = list(range(0, max(w - crop_size, 0) + 1, stride))
    if not y_steps:
        y_steps = [0]
    if not x_steps:
        x_steps = [0]
    last_y = max(h - crop_size, 0)
    last_x = max(w - crop_size, 0)
    if y_steps[-1] != last_y:
        y_steps.append(last_y)
    if x_steps[-1] != last_x:
        x_steps.append(last_x)

    for y in y_steps:
        for x in x_steps:
            tile = image[:, y : y + crop_size, x : x + crop_size]
            pad_h = max(0, crop_size - tile.shape[1])
            pad_w = max(0, crop_size - tile.shape[2])
            if pad_h > 0 or pad_w > 0:
                tile = F.pad(tile, (0, pad_w, 0, pad_h), value=0.0)

            tile = tile.unsqueeze(0).to(device, non_blocking=True)
            with autocast(enabled=amp):
                logits = forward_logits(model, tile).squeeze(0)
            logits = logits[:, : tile.shape[2] - pad_h, : tile.shape[3] - pad_w]
            probs = torch.softmax(logits, dim=0)

            out_probs[:, y : y + probs.shape[1], x : x + probs.shape[2]] += probs
            count[:, y : y + probs.shape[1], x : x + probs.shape[2]] += 1.0

    out_probs = out_probs / count.clamp_min(1.0)
    return out_probs


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    crop_size: int,
    stride: int,
    device: torch.device,
    amp: bool,
    ignore_index: int,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    for batch_idx, (images, masks) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        for b in range(images.shape[0]):
            img = images[b]
            mask = masks[b]
            if img.shape[-1] > crop_size or img.shape[-2] > crop_size:
                probs = sliding_window_logits(
                    model=model,
                    image=img,
                    crop_size=crop_size,
                    stride=stride,
                    num_classes=num_classes,
                    device=device,
                    amp=amp,
                )
                pred = probs.argmax(dim=0)
            else:
                with autocast(enabled=amp):
                    logits = forward_logits(model, img.unsqueeze(0))
                pred = logits.argmax(dim=1).squeeze(0)
            cm.update(preds=pred, targets=mask)

    stats = cm.compute()
    return {"miou": float(stats["miou"].item()), "acc": float(stats["acc"].item())}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    state: TrainState,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": state.epoch,
            "step": state.step,
            "best_miou": state.best_miou,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    map_location: str = "cpu",
) -> TrainState:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return TrainState(epoch=int(ckpt.get("epoch", 0)), step=int(ckpt.get("step", 0)), best_miou=float(ckpt.get("best_miou", 0.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="LoveDA")
    parser.add_argument("--domain", type=str, default="Both", choices=["Urban", "Rural", "Both"])
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--ignore_index", type=int, default=255)

    parser.add_argument("--arch", type=str, default="smp_deeplabv3plus", choices=["smp_deeplabv3plus", "smp_unetplusplus", "tv_deeplabv3_resnet50"])
    parser.add_argument("--encoder", type=str, default="resnet50")
    parser.add_argument("--pretrained_backbone", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--freeze_bn", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--save_dir", type=str, default="checkpoints_loveda")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--eval_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vis_dir", type=str, default="vis_loveda")
    parser.add_argument("--vis_every", type=int, default=15)
    parser.add_argument("--vis_start", type=int, default=15)
    parser.add_argument("--vis_samples", type=int, default=5)
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 6e-4
    if args.weight_decay is None:
        args.weight_decay = 0.01

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    train_ds = LoveDADataset(
        data_root=args.data_root,
        split="Train",
        domain=args.domain,
        transform=TrainTransform(crop_size=args.crop_size, ignore_index=args.ignore_index),
    )
    val_ds = LoveDADataset(
        data_root=args.data_root,
        split="Val",
        domain=args.domain,
        transform=ValTransform(ignore_index=args.ignore_index),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.workers // 2),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_model(
        arch=args.arch,
        num_classes=args.num_classes,
        encoder=args.encoder,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    state = TrainState(epoch=0, step=0, best_miou=0.0)
    if args.resume:
        state = load_checkpoint(Path(args.resume), model=model, optimizer=optimizer if not args.eval_only else None, scaler=scaler if not args.eval_only else None, map_location=str(device))

    if args.eval_only:
        stats = evaluate(
            model=model,
            loader=val_loader,
            num_classes=args.num_classes,
            crop_size=args.crop_size,
            stride=args.stride,
            device=device,
            amp=args.amp and device.type == "cuda",
            ignore_index=args.ignore_index,
            max_batches=args.max_val_batches,
        )
        print(f"val miou={stats['miou']:.4f} acc={stats['acc']:.4f}")
        return

    steps_per_epoch = len(train_loader)
    if args.max_train_batches and args.max_train_batches > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_train_batches)
    total_steps = args.epochs * steps_per_epoch
    ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    vis_indices: list[int] = []
    if args.vis and args.vis_samples > 0:
        k = min(args.vis_samples, len(val_ds))
        xs = torch.linspace(0, len(val_ds) - 1, steps=k).tolist()
        vis_indices = [int(round(x)) for x in xs]

    for epoch in range(state.epoch, args.epochs):
        model.train()
        if args.freeze_bn:
            freeze_batchnorm(model)
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()

        for batch_idx, (images, masks) in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            lr = poly_lr(args.lr, state.step, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp and device.type == "cuda"):
                logits = forward_logits(model, images)
                loss = args.ce_weight * ce(logits, masks) + args.dice_weight * dice_loss(
                    logits=logits,
                    targets=masks,
                    num_classes=args.num_classes,
                    ignore_index=args.ignore_index,
                )

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().item())
            n_batches += 1
            state = TrainState(epoch=epoch, step=state.step + 1, best_miou=state.best_miou)
            if args.max_steps and state.step >= args.max_steps:
                break

        if args.max_steps and state.step >= args.max_steps:
            print("stopping early due to --max_steps")

        epoch_loss /= max(1, n_batches)
        dt = time.time() - epoch_start
        print(f"epoch {epoch+1}/{args.epochs} loss={epoch_loss:.4f} lr={optimizer.param_groups[0]['lr']:.3e} time={dt:.1f}s")

        if (epoch + 1) % args.val_every == 0:
            stats = evaluate(
                model=model,
                loader=val_loader,
                num_classes=args.num_classes,
                crop_size=args.crop_size,
                stride=args.stride,
                device=device,
                amp=args.amp and device.type == "cuda",
                ignore_index=args.ignore_index,
                max_batches=args.max_val_batches,
            )
            print(f"val miou={stats['miou']:.4f} acc={stats['acc']:.4f}")

            save_dir = Path(args.save_dir)
            save_checkpoint(save_dir / "last.pt", model, optimizer, scaler, state)
            if stats["miou"] > state.best_miou:
                state = TrainState(epoch=state.epoch, step=state.step, best_miou=stats["miou"])
                save_checkpoint(save_dir / "best.pt", model, optimizer, scaler, state)

        if (
            args.vis
            and vis_indices
            and args.vis_every > 0
            and (epoch + 1) >= max(1, args.vis_start)
            and ((epoch + 1 - args.vis_start) % args.vis_every == 0)
        ):
            save_epoch_visuals(
                model=model,
                val_ds=val_ds,
                vis_indices=vis_indices,
                out_dir=Path(args.vis_dir) / f"epoch_{epoch+1:03d}",
                num_classes=args.num_classes,
                crop_size=args.crop_size,
                stride=args.stride,
                device=device,
                amp=args.amp and device.type == "cuda",
                ignore_index=args.ignore_index,
            )
            print(f"saved visuals: {Path(args.vis_dir) / f'epoch_{epoch+1:03d}'}")

        if args.max_steps and state.step >= args.max_steps:
            save_dir = Path(args.save_dir)
            save_checkpoint(save_dir / "last.pt", model, optimizer, scaler, state)
            break


if __name__ == "__main__":
    main()
