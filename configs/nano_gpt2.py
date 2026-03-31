"""
Nano GPT-2 model and training configuration.

配置与模型代码分离的原因：
  1. 跨架构复用同一套数值 — GPT-2 和 Qwen3 宏观结构相同（Embedding → Block * N → Norm → LM Head），
     vocab_size / d_model / n_layers / n_heads / context_len 对两个架构含义一致，
     用相同数值可以保证对比的是架构差异而非规模差异。
  2. 同一架构跑不同规模 — nano / small / base 只是数字不同，模型代码不变。
  3. 解耦结构定义与实例化参数 — 模型代码描述"怎么连接"，配置描述"连多宽多深"。
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 100277
    context_len: int = 256
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4       # d_model // n_heads = 32
    dropout: float = 0.0
    impl: str = "torch"    # "torch" | "scratch"


@dataclass
class TrainConfig:
    # data
    cache_dir: str = "data/cache"
    # training (step-based)
    batch_size: int = 16
    max_steps: int = 150_000        # safety bound；early stopping 通常更早触发
    eval_interval: int = 800         # 每 N step：eval + 写 log + 生成样本
    ckpt_interval: int = 10_000     # 每 N step：保存 checkpoint（约 1 小时一次）
    eval_steps: int = 50            # 每次 eval 用多少个 val batch 估算损失
    early_stop_patience: int = 10   # 连续 N 次 eval val loss 不降则停止
    # optimizer
    lr: float = 3e-4
    weight_decay: float = 0.1
    # device
    device: str = "mps"


@dataclass
class InferConfig:
    run_dir: str = "runs/nano_gpt2_20260330_233710"
    ckpt: str = "ckpt_best.pt"
    prompt: str = "Once upon a time"
    max_new_tokens: int = 200
    temperature: float = 0.8
