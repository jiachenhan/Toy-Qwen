"""
Positional encoding layers — PyTorch high-level implementation.

  AbsolutePE  — learned absolute positional embeddings via nn.Embedding  (used by: GPT-2)
  RoPE        — Rotary Position Embedding                                 (used by: Qwen3, TODO)

Note: GPT-2 uses nn.Embedding for positional encoding directly in NanoGPT2.__init__,
so no separate wrapper class is needed in the torch backend.
"""
