"""
Positional encoding layers — scratch implementation (hand-rolled tensor ops).

  AbsolutePE  — learned absolute positional embeddings  (used by: GPT-2)
  RoPE        — Rotary Position Embedding               (used by: Qwen3, TODO)

Template: implement each class using only basic torch.Tensor operations.
Reference torch implementation: models/layers/torch/positional.py
"""

import torch.nn as nn
from torch import Tensor


class RoPE(nn.Module):
    """Rotary Position Embedding — scratch implementation.

    Used by: Qwen3 (TODO)

    Implement the rotation matrix applied to Q and K in attention.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
