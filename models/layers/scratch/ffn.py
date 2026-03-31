"""
Feed-Forward Network layers — scratch implementation (hand-rolled tensor ops).

  MLP      — GELU activation, 4x expansion  (used by: GPT-2)
  SwiGLU   — Swish-Gated Linear Unit        (used by: Qwen3, TODO)

Template: implement each class using only basic torch.Tensor operations.
Reference torch implementation: models/layers/torch/ffn.py
"""

import torch.nn as nn
from torch import Tensor

from configs.nano_gpt2 import ModelConfig


class MLP(nn.Module):
    """Two-layer MLP with GELU activation — scratch implementation.

    Used by: GPT-2 (NanoGPT2)

    Implement GELU from scratch instead of using nn.GELU().
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc   = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
