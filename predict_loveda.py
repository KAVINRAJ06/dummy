from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from loveda_seg.data import ValTransform
from loveda_seg.models import build_model, forward_logits


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--arch", type=str, default="smp_deeplabv3plus", choices=["smp_deeplabv3plus", "smp_unetplusplus", "tv_deeplabv3_resnet50"])
    parser.add_argument("--encoder", type=str, default="resnet50")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    model = build_model(arch=args.arch, num_classes=args.num_classes, encoder=args.encoder, pretrained_backbone=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    transform = ValTransform()
    in_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(in_dir.glob("*.png"), key=lambda p: p.stem):
        img = load_image(img_path)
        img_t, _ = transform(img, Image.fromarray(np.zeros((img.height, img.width), dtype=np.uint8)))
        img_t = img_t.to(device)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            logits = forward_logits(model, img_t.unsqueeze(0))
        pred = logits.argmax(dim=1).squeeze(0).to(torch.uint8).cpu().numpy()
        Image.fromarray(pred, mode="L").save(out_dir / img_path.name)


if __name__ == "__main__":
    main()
