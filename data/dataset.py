"""
PyTorch Dataset over pre-tokenized binary shards.

Usage:
    ds = TokenDataset("data/cache/train.bin", context_len=256)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    x, y = next(iter(loader))   # x: (B, T), y: (B, T) shifted by 1
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TokenDataset(Dataset):
    """
    Memory-maps a flat uint32 token file.
    Each __getitem__ returns a (context_len,) input and a (context_len,) target
    where target = input shifted left by 1 (next-token prediction).
    """

    def __init__(self, bin_path: Path | str, context_len: int) -> None:
        super().__init__()
        self.context_len = context_len
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r")
        # Number of non-overlapping chunks; -1 because target needs one extra token
        self.n_chunks = (len(self.data) - 1) // context_len

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.context_len
        chunk = torch.from_numpy(
            self.data[start : start + self.context_len + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


def build_loaders(
    cache_dir: Path | str,
    context_len: int,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, dict]:
    """Return (train_loader, val_loader, meta)."""
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())

    train_ds = TokenDataset(cache_dir / "train.bin", context_len)
    val_ds   = TokenDataset(cache_dir / "val.bin",   context_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, val_loader, meta
