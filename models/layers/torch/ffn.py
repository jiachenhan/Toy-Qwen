"""
Feed-Forward Network layers — PyTorch high-level implementation.

  MLP      — GELU activation, 4x expansion  (used by: GPT-2)
  SwiGLU   — Swish-Gated Linear Unit        (used by: Qwen3, TODO)
"""

import torch.nn as nn
from torch import Tensor

from configs.nano_gpt2 import ModelConfig


class MLP(nn.Module):
    """Two-layer MLP with GELU activation and 4x hidden expansion.

    Used by: GPT-2 (NanoGPT2)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc   = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.act  = nn.GELU()
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.act(self.fc(x)))
