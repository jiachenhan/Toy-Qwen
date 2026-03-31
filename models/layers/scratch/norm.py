"""
Normalization layers — scratch implementation (hand-rolled tensor ops).

  LayerNorm  — standard layer normalization  (used by: GPT-2)
  RMSNorm    — root-mean-square normalization (used by: Qwen3, TODO)

Template: implement each class using only basic torch.Tensor operations.
Reference torch implementation: models/layers/torch/norm.py
"""

import torch.nn as nn
from torch import Tensor


class LayerNorm(nn.Module):
    """Layer normalization — scratch implementation.

    Used by: GPT-2 (NanoGPT2)

    Implement: x_norm = (x - mean) / sqrt(var + eps) * weight + bias
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(__import__("torch").ones(d_model))
        self.bias   = nn.Parameter(__import__("torch").zeros(d_model))
        self.eps    = eps

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization — scratch implementation.

    Used by: Qwen3 (TODO)

    Implement: x_norm = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(__import__("torch").ones(d_model))
        self.eps    = eps

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
