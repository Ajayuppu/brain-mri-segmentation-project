"""Manifest-driven dataset. TIFFs read with tifffile (preserves stored channel order)."""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def load_manifest(data_root: str, split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(split)
    df = pd.read_csv(os.path.join(data_root, f"{split}_manifest.csv"))
    if not (df["split"] == split).all():
        raise ValueError(f"manifest rows not all {split}")
    return df.reset_index(drop=True)


def read_image(path: str) -> np.ndarray:
    import tifffile
    img = tifffile.imread(path)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"unexpected image shape {img.shape} for {path}")
    return img.astype(np.float32)  # HWC, channel order as stored


def read_mask(path: str) -> np.ndarray:
    import tifffile
    m = tifffile.imread(path)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)  # README: stored 0/255 -> binary at load


def fit_channel_stats(data_root: str, train_df: pd.DataFrame) -> dict:
    """Per-channel mean/std computed on TRAINING images only."""
    s = np.zeros(3); ss = np.zeros(3); n = 0
    for p in train_df["image_path"]:
        x = read_image(os.path.join(data_root, p)).reshape(-1, 3).astype(np.float64)
        s += x.sum(0); ss += (x ** 2).sum(0); n += x.shape[0]
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-12))
    return {"mean": mean.tolist(), "std": std.tolist(), "fitted_on": "train", "n_images": int(len(train_df))}


def save_stats(stats: dict, path: str):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Aligned geometric augmentation: identical flip/rot90 applied to image and mask."""
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    k = int(rng.integers(0, 4))
    img, mask = np.rot90(img, k, axes=(0, 1)), np.rot90(mask, k, axes=(0, 1))
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class SegDataset:
    """torch-compatible dataset (no torch import needed to construct)."""

    def __init__(self, data_root, df, stats, train=False, seed=0):
        self.root, self.df, self.train = data_root, df, train
        self.mean = np.asarray(stats["mean"], np.float32)
        self.std = np.asarray(stats["std"], np.float32)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = read_image(os.path.join(self.root, r["image_path"]))
        mask = read_mask(os.path.join(self.root, r["mask_path"]))
        if self.train:
            img, mask = augment(img, mask, self.rng)
        img = (img - self.mean) / self.std
        return img.transpose(2, 0, 1).astype(np.float32), mask[None].astype(np.float32), i
