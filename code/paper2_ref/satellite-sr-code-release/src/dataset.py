"""Paired LR-HR dataset built on top of fixed split files.

Splits are read from text files (one image path per line) so that the exact
train/val/test partition is reproducible. LR images are generated on-the-fly
with the deterministic degradation pipeline in degrade.py.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .degrade import degrade


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0   # [0,1]
    arr = (arr - 0.5) / 0.5                                           # [-1,1]
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()


class SatelliteDataset(Dataset):
    def __init__(self, split_file, hr_size=256, scale=4, degradation="bicubic",
                 augment=False, degrade_kwargs=None):
        with open(split_file) as f:
            self.files = [ln.strip() for ln in f if ln.strip()]
        self.hr_size = hr_size
        self.scale = scale
        self.degradation = degradation
        self.augment = augment
        self.degrade_kwargs = degrade_kwargs or {}

    def __len__(self):
        return len(self.files)

    def _random_crop(self, img):
        w, h = img.size
        if w < self.hr_size or h < self.hr_size:
            img = img.resize((max(w, self.hr_size), max(h, self.hr_size)), Image.BICUBIC)
            w, h = img.size
        x = np.random.randint(0, w - self.hr_size + 1)
        y = np.random.randint(0, h - self.hr_size + 1)
        return img.crop((x, y, x + self.hr_size, y + self.hr_size))

    def _augment(self, img):
        if np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        k = np.random.randint(0, 4)
        if k:
            img = img.rotate(90 * k)
        return img

    def __getitem__(self, idx):
        hr = Image.open(self.files[idx]).convert("RGB")
        hr = self._random_crop(hr)
        if self.augment:
            hr = self._augment(hr)
        lr = degrade(hr, mode=self.degradation, scale=self.scale, **self.degrade_kwargs)
        return _to_tensor(lr), _to_tensor(hr)


def build_loader(split_file, batch_size=16, hr_size=256, scale=4,
                 degradation="bicubic", augment=False, shuffle=True, num_workers=4):
    ds = SatelliteDataset(split_file, hr_size, scale, degradation, augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle)
