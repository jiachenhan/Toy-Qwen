"""
Attention layers — PyTorch high-level implementation.

  CausalSelfAttention  — Multi-Head Attention with causal mask  (used by: GPT-2)
  GQA                  — Grouped-Query Attention                (used by: Qwen3, TODO)
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from configs.nano_gpt2 import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention using F.scaled_dot_product_attention.

    Used by: GPT-2 (NanoGPT2)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1) # (B, T, 384) → q(B,T,128), k(B,T,128), v(B,T,128)
        # reshape to (B, n_heads, T, head_dim) for sdpa
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # is_causal=True: no explicit mask needed, sdpa builds it internally
        """
            1. 注意力分数
            scores = q @ k.transpose(-2, -1)
            # (B,4,T,32) @ (B,4,32,T) → (B,4,T,T)

            scores = scores / (32 ** 0.5)
            # 除以 sqrt(head_dim)，防止点积太大

            2. 因果掩码：每个位置只能看到自己和之前的
            mask = torch.triu(torch.ones(T,T), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf')) # 未来位置设为 -inf

            施加掩码前的分数（随便填的数）：
                    pos0  pos1  pos2  pos3   ← Key（被看的）
            pos0   [ 0.9   0.3   0.7   0.2 ]
            pos1   [ 0.4   0.8   0.1   0.6 ]
            pos2   [ 0.2   0.5   0.9   0.3 ]
            pos3   [ 0.1   0.7   0.4   0.8 ]
            ↑ Query（当前位置）

            因果掩码是上三角（diagonal=1）：
                    pos0  pos1  pos2  pos3
            pos0   [  F     T     T     T  ]   ← pos0 只能看 pos0，后面全掩掉
            pos1   [  F     F     T     T  ]   ← pos1 能看 pos0, pos1
            pos2   [  F     F     F     T  ]   ← pos2 能看 pos0~pos2
            pos3   [  F     F     F     F  ]   ← pos3 能看全部
            True 的位置填 -inf，softmax 后变 0（不贡献任何权重）

            施加掩码后：
                    pos0  pos1  pos2  pos3
            pos0   [ 0.9   -inf  -inf  -inf ]
            pos1   [ 0.4   0.8   -inf  -inf ]
            pos2   [ 0.2   0.5   0.9   -inf ]
            pos3   [ 0.1   0.7   0.4   0.8  ]

            softmax 后（每行加起来=1）：
                    pos0  pos1  pos2  pos3
            pos0   [ 1.0   0.0   0.0   0.0 ] ← 只看自己
            pos1   [ 0.4   0.6   0.0   0.0 ]
            pos2   [ 0.2   0.3   0.5   0.0 ]
            pos3   [ 0.1   0.3   0.2   0.4 ]


            3. softmax 得到注意力权重
            weights = F.softmax(scores, dim=-1)   # (B,4,T,T) 每行加起来=1

            4. 加权求和 V
            out = weights @ v
            # (B,4,T,T) @ (B,4,T,32) → (B,4,T,32)
        """

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)
