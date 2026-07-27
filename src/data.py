"""
src/data.py -- Implements the get_dataloaders signature frozen in
src/interfaces.py.

Contract (from interfaces.py):
    get_dataloaders(variant, ratio, seed, batch_size) -> (DataLoader, DataLoader)
    yields (img: Tensor[B,3,224,224], label: Tensor[B]), labels 0..37

`return_val` is a keyword-only argument with a default, so every existing call
site keeps working unchanged and the frozen signature is not broken. GM2 opts
in when checkpointing best-by-val-F1.

pin_memory is device-aware (CUDA only) rather than hardcoded True, since this
file needs to run correctly on Apple Silicon (MPS, where pin_memory is
unsupported) as well as CUDA boxes without anyone hand-editing it per machine.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DATA_ROOT = Path("data")
SPLIT_DIR = Path("splits")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
SOURCE_SIZE = 256  # PlantVillage ships at 256x256; no pre-resize needed


@lru_cache(maxsize=1)
def class_names(data_root: Path = DATA_ROOT) -> list[str]:
    """Frozen label ordering, written once by prepare_data.py."""
    return json.loads((data_root / "classes.json").read_text())


def label_of(relpath: str, data_root: Path = DATA_ROOT) -> int:
    return class_names(data_root).index(relpath.split("/", 1)[0])


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------
def build_transforms(train: bool) -> transforms.Compose:
    """
    Identical for color and grayscale, and for pretrained and scratch. If these
    ever diverge across the four cells, the experiment stops being a controlled
    comparison.

    RandomResizedCrop's default scale=(0.08, 1.0) is tuned for ImageNet's
    cluttered scenes. On a centred leaf against a flat background, an 8% crop is
    usually just background, so the lower bound is raised to 0.7.

    Grayscale is handled by .convert("RGB") in the dataset, which replicates the
    single channel three times. Both variants therefore share one model code
    path and one normalisation, and any color-vs-gray gap is attributable to the
    pixels rather than to the plumbing.
    """
    if train:
        ops = [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        ops = [
            transforms.Resize(SOURCE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
        ]
    return transforms.Compose(ops + [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
class PlantVillageDataset(Dataset):
    """Reads images by relative path from a committed split file."""

    def __init__(self, root: Path, relpaths: list[str], transform, data_root: Path = DATA_ROOT):
        self.root = Path(root)
        self.relpaths = relpaths
        self.transform = transform
        self.labels = [label_of(p, data_root) for p in relpaths]

    def __len__(self) -> int:
        return len(self.relpaths)

    def __getitem__(self, idx: int):
        path = self.root / self.relpaths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img), self.labels[idx]


def load_split(variant: str, ratio: float, seed: int, split_dir: Path = SPLIT_DIR) -> dict:
    path = Path(split_dir) / f"{variant}_{ratio:g}_{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run scripts/make_splits.py and commit splits/.")
    return json.loads(path.read_text())


def _loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def get_dataloaders(
    variant: str,
    ratio: float,
    seed: int,
    batch_size: int,
    *,
    return_val: bool = False,
    test_split: str = "test",
    data_root: Path = DATA_ROOT,
    split_dir: Path = SPLIT_DIR,
    num_workers: int = 4,
):
    """
    Returns (train_loader, test_loader), or (train, val, test) if return_val.

    test_split selects "test" (Mohanty-style complement, for the baseline
    comparison table) or "test_fixed" (constant 20% holdout, for the
    accuracy-vs-training-set-size curve).
    """
    split = load_split(variant, ratio, seed, split_dir)
    root = Path(data_root) / "raw" / variant

    def make(key: str, train: bool) -> Dataset:
        return PlantVillageDataset(root, split[key], build_transforms(train), Path(data_root))

    train_loader = _loader(make("train", True), batch_size, True, num_workers)
    test_loader = _loader(make(test_split, False), batch_size, False, num_workers)

    if not return_val:
        return train_loader, test_loader

    if not split["val"]:
        raise ValueError(
            "This split has no val set (generated with --val-frac 0). Either "
            "regenerate the splits or call with return_val=False and select on test."
        )
    val_loader = _loader(make("val", False), batch_size, False, num_workers)
    return train_loader, val_loader, test_loader



class FakeDataset(Dataset):
    """Random tensors of the correct shape and label range."""

    def __init__(self, n: int = 256, num_classes: int = 38, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g)
        self.y = torch.randint(0, num_classes, (n,), generator=g)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def get_fake_dataloaders(batch_size: int = 64, n: int = 256):
    return (
        _loader(FakeDataset(n, seed=0), batch_size, True, 0),
        _loader(FakeDataset(n // 4, seed=1), batch_size, False, 0),
    )


if __name__ == "__main__":
    # Done-when check from the plan: any teammate runs this and gets your numbers.
    tr, te = get_dataloaders("color", 0.8, 42, 64)
    xb, yb = next(iter(tr))
    print(f"train batches={len(tr)}  test batches={len(te)}")
    print(f"x={tuple(xb.shape)} {xb.dtype}  y={tuple(yb.shape)} range=[{yb.min()},{yb.max()}]")
    print(f"train images={len(tr.dataset)}  test images={len(te.dataset)}")
