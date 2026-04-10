"""
Normalization layers — scratch implementation (pure tensor ops).

  LayerNorm  — standard layer normalization  (used by: GPT-2)
  RMSNorm    — root-mean-square normalization (used by: Qwen3, TODO)

Reference torch implementation: models/layers/torch/norm.py
"""

import torch
import torch.nn as nn
from torch import Tensor


class LayerNorm(nn.Module):
    """Layer normalization — scratch implementation.

    Used by: GPT-2 (NanoGPT2)

    Formula:
      mean    = E[x]           over last dimension
      var     = E[(x - mean)²] over last dimension
      x_norm  = (x - mean) / sqrt(var + eps)
      output  = weight * x_norm + bias   (learned affine transform)
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))   # scale (γ), init = 1
        self.bias   = nn.Parameter(torch.zeros(d_model))  # shift (β), init = 0
        self.eps    = eps

    def forward(self, x: Tensor) -> Tensor:
        # Compute mean and variance over the last dimension (d_model)
        mean = x.mean(dim=-1, keepdim=True)                        # (B, T, 1)
        var  = ((x - mean) ** 2).mean(dim=-1, keepdim=True)        # (B, T, 1)

        # Normalize, then apply learned affine transform
        x_norm = (x - mean) / torch.sqrt(var + self.eps)           # (B, T, d_model)
        return self.weight * x_norm + self.bias


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization — scratch implementation.

    Used by: Qwen3 (TODO)

    Formula:
      rms    = sqrt( mean(x²) + eps )
      x_norm = x / rms
      output = weight * x_norm

    Difference from LayerNorm: no mean subtraction (no re-centering),
    which makes it faster and works equally well in practice.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps    = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)  # (B, T, 1)
        return self.weight * (x / rms)
