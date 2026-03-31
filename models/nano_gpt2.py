"""
Nano GPT-2 style model.

Architecture:
  - Learned absolute positional embedding
  - TransformerBlock: LayerNorm → MHA → residual, LayerNorm → MLP → residual
  - MLP activation: GELU
  - Final LayerNorm + linear LM head (no weight tying)

Layer implementations are selected by ModelConfig.impl:
  "torch"   → models/layers/torch/   (PyTorch high-level API)
  "scratch" → models/layers/scratch/ (hand-rolled tensor ops)
"""

import torch
import torch.nn as nn
from torch import Tensor

from configs.nano_gpt2 import ModelConfig


def _get_layer_classes(impl: str):
    if impl == "torch":
        from models.layers.torch.attention import CausalSelfAttention
        from models.layers.torch.ffn import MLP
    else:
        from models.layers.scratch.attention import CausalSelfAttention
        from models.layers.scratch.ffn import MLP
    return CausalSelfAttention, MLP


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, Attn, FFN) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = Attn(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.mlp  = FFN(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoGPT2(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        Attn, FFN = _get_layer_classes(cfg.impl)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.d_model)
        self.blocks  = nn.ModuleList([TransformerBlock(cfg, Attn, FFN) for _ in range(cfg.n_layers)])
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: Tensor) -> Tensor:
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))   # (B, T, vocab_size)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
