"""
AdamW optimizer — PyTorch backend.

Re-exports torch.optim.AdamW for a uniform interface with optim/scratch/adamw.py.
"""

from torch.optim import AdamW

__all__ = ["AdamW"]
