"""
Attention layers — scratch implementation (hand-rolled tensor ops).

  CausalSelfAttention  — Multi-Head Attention with causal mask  (used by: GPT-2)
  GQA                  — Grouped-Query Attention                (used by: Qwen3, TODO)

Template: implement each class using only basic torch.Tensor operations.
Reference torch implementation: models/layers/torch/attention.py
"""

import torch.nn as nn
from torch import Tensor

from configs.nano_gpt2 import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention — scratch implementation.

    Used by: GPT-2 (NanoGPT2)

    Implement without F.scaled_dot_product_attention:
      1. compute Q, K, V projections
      2. split into heads
      3. compute attention scores with causal mask
      4. softmax + weighted sum of V
      5. project output
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
