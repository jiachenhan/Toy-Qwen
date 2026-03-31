"""
Normalization layers — PyTorch high-level implementation.

  LayerNorm  — standard layer normalization  (used by: GPT-2)
  RMSNorm    — root-mean-square normalization (used by: Qwen3, TODO)
"""

from torch.nn import LayerNorm  # re-export; GPT-2 uses nn.LayerNorm directly

__all__ = ["LayerNorm"]
