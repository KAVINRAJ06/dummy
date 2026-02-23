from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F


LoveDADomain = Literal["Urban", "Rural", "Both"]


@dataclass(frozen=True)
class LoveDAPaths:
    image_path: Path
    mask_path: Path


def _iter_domains(domain: LoveDADomain) -> Iterable[str]:
    if domain == "Both":
        return ("Urban", "Rural")
    return (domain,)


def collect_loveda_pairs(data_root: str | Path, split: Literal["Train", "Val"], domain: LoveDADomain) -> list[LoveDAPaths]:
    data_root = Path(data_root)
    pairs: list[LoveDAPaths] = []
    for d in _iter_domains(domain):
        images_dir = data_root / split / d / "images_png"
        masks_dir = data_root / split / d / "masks_png"
        if not images_dir.exists():
            raise FileNotFoundError(str(images_dir))
        if not masks_dir.exists():
            raise FileNotFoundError(str(masks_dir))

        for img_path in sorted(images_dir.glob("*.png"), key=lambda p: p.stem):
            mask_path = masks_dir / img_path.name
            if not mask_path.exists():
                raise FileNotFoundError(str(mask_path))
            pairs.append(LoveDAPaths(image_path=img_path, mask_path=mask_path))
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found under {data_root} for {split=} {domain=}")
    return pairs


class LoveDADataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        data_root: str | Path,
        split: Literal["Train", "Val"],
        domain: LoveDADomain = "Both",
        transform: callable | None = None,
    ) -> None:
        self.pairs = collect_loveda_pairs(data_root=data_root, split=split, domain=domain)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]
        img = Image.open(pair.image_path).convert("RGB")
        mask = Image.open(pair.mask_path)
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask


class TrainTransform:
    def __init__(
        self,
        crop_size: int = 512,
        scale_range: tuple[float, float] = (0.5, 2.0),
        ignore_index: int = 255,
    ) -> None:
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.ignore_index = ignore_index

        self.color = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        self.blur = T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))
        self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, img: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(()) < 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)

        scale = float(torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item())
        new_h = max(1, int(round(img.height * scale)))
        new_w = max(1, int(round(img.width * scale)))

        img = F.resize(img, size=[new_h, new_w], interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, size=[new_h, new_w], interpolation=InterpolationMode.NEAREST)

        if torch.rand(()) < 0.8:
            img = self.color(img)
        if torch.rand(()) < 0.2:
            img = self.blur(img)

        pad_h = max(0, self.crop_size - new_h)
        pad_w = max(0, self.crop_size - new_w)
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, padding=[0, 0, pad_w, pad_h], fill=0)
            mask = F.pad(mask, padding=[0, 0, pad_w, pad_h], fill=self.ignore_index)

        i, j, h, w = T.RandomCrop.get_params(img, output_size=(self.crop_size, self.crop_size))
        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        img_t = F.to_image(img)
        img_t = F.to_dtype(img_t, torch.float32, scale=True)
        img_t = self.normalize(img_t)

        mask_arr = np.array(mask, dtype=np.int64)
        mask_arr[mask_arr == 7] = self.ignore_index
        mask_t = torch.as_tensor(mask_arr, dtype=torch.int64)
        return img_t, mask_t


class ValTransform:
    def __init__(self, ignore_index: int = 255) -> None:
        self.ignore_index = ignore_index
        self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, img: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        img_t = F.to_image(img)
        img_t = F.to_dtype(img_t, torch.float32, scale=True)
        img_t = self.normalize(img_t)
        mask_arr = np.array(mask, dtype=np.int64)
        mask_arr[mask_arr == 7] = self.ignore_index
        mask_t = torch.as_tensor(mask_arr, dtype=torch.int64)
        return img_t, mask_t
