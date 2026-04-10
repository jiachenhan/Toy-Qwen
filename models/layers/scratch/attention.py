"""
Attention layers — scratch implementation (pure tensor ops, no nn.Linear).

  CausalSelfAttention  — Multi-Head Attention with causal mask  (used by: GPT-2)
  GQA                  — Grouped-Query Attention                (used by: Qwen3, TODO)

Reference torch implementation: models/layers/torch/attention.py
"""

import torch
import torch.nn as nn
from torch import Tensor

from configs.nano_gpt2 import ModelConfig
from models.layers.scratch.init import kaiming_uniform


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention — scratch implementation.

    Used by: GPT-2 (NanoGPT2)

    Forward steps:
      1. QKV projection:  x @ W_qkv.T + b_qkv
      2. Split Q, K, V and reshape into (B, n_heads, T, head_dim)
      3. Attention scores: Q @ K^T / sqrt(head_dim)
      4. Causal mask: fill upper triangle with -inf
      5. Softmax (numerically stable, manual)
      6. Weighted sum of V
      7. Merge heads and output projection
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        d = cfg.d_model

        # QKV projection weights: map d_model → 3 * d_model (Q, K, V stacked)
        self.W_qkv = nn.Parameter(torch.empty(3 * d, d))
        self.b_qkv = nn.Parameter(torch.empty(3 * d))
        kaiming_uniform(self.W_qkv, fan_in=d)
        kaiming_uniform(self.b_qkv, fan_in=d)

        # Output projection weights: map d_model → d_model
        self.W_proj = nn.Parameter(torch.empty(d, d))
        self.b_proj = nn.Parameter(torch.empty(d))
        kaiming_uniform(self.W_proj, fan_in=d)
        kaiming_uniform(self.b_proj, fan_in=d)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape  # batch, sequence length, d_model

        # 1. QKV projection: (B, T, C) → (B, T, 3C), then split
        qkv = x @ self.W_qkv.T + self.b_qkv   # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)         # each: (B, T, C)

        # 2. Split into heads: (B, T, C) → (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. Attention scores: (B, n_heads, T, T)
        #    Divide by sqrt(head_dim) to keep variance ~1 regardless of head_dim
        scale  = self.head_dim ** -0.5
        scores = q @ k.transpose(-2, -1) * scale  # (B, n_heads, T, T)

        # 4. Causal mask: positions in the upper triangle cannot attend to future tokens
        #    triu diagonal=1 masks out (i, j) where j > i → set to -inf so softmax → 0
        mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        # 5. Softmax (numerically stable): subtract row max before exp to avoid overflow
        #    softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
        scores_shifted = scores - scores.max(dim=-1, keepdim=True).values
        exp_scores     = torch.exp(scores_shifted)
        weights        = exp_scores / exp_scores.sum(dim=-1, keepdim=True)  # (B, n_heads, T, T)

        # 6. Weighted sum of V: (B, n_heads, T, T) @ (B, n_heads, T, head_dim) → (B, n_heads, T, head_dim)
        out = weights @ v

        # 7. Merge heads: (B, n_heads, T, head_dim) → (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        return out @ self.W_proj.T + self.b_proj
