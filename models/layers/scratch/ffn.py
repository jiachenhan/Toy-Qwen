"""
Feed-Forward Network layers — scratch implementation (pure tensor ops, no nn.Linear).

  MLP      — GELU activation, 4x expansion  (used by: GPT-2)
  SwiGLU   — Swish-Gated Linear Unit        (used by: Qwen3, TODO)

Reference torch implementation: models/layers/torch/ffn.py
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from configs.nano_gpt2 import ModelConfig
from models.layers.scratch.init import kaiming_uniform


class MLP(nn.Module):
    """Two-layer MLP with GELU activation — scratch implementation.

    Used by: GPT-2 (NanoGPT2)

    Architecture:  x → fc (d → 4d) → GELU → proj (4d → d)

    GELU (tanh approximation, used by GPT-2):
      gelu(x) = 0.5 · x · (1 + tanh( sqrt(2/π) · (x + 0.044715 · x³) ))

      Why not plain ReLU? GELU is smooth near 0 and gives slightly better
      training dynamics for transformers. The tanh form is the original
      Hendrycks & Gimpel approximation; PyTorch's nn.GELU also defaults to this.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model

        # First linear: d_model → 4 * d_model
        self.W_fc   = nn.Parameter(torch.empty(4 * d, d))
        self.b_fc   = nn.Parameter(torch.empty(4 * d))
        kaiming_uniform(self.W_fc,   fan_in=d)
        kaiming_uniform(self.b_fc,   fan_in=d)

        # Second linear (projection): 4 * d_model → d_model
        self.W_proj = nn.Parameter(torch.empty(d, 4 * d))
        self.b_proj = nn.Parameter(torch.empty(d))
        kaiming_uniform(self.W_proj, fan_in=4 * d)
        kaiming_uniform(self.b_proj, fan_in=4 * d)

    def forward(self, x: Tensor) -> Tensor:
        # 1. Expand: (B, T, d) → (B, T, 4d)
        h = x @ self.W_fc.T + self.b_fc

        # 2. GELU activation (tanh approximation)
        h = 0.5 * h * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (h + 0.044715 * h ** 3)))

        # 3. Project back: (B, T, 4d) → (B, T, d)
        return h @ self.W_proj.T + self.b_proj
