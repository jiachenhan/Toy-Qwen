# Toy-Qwen

在 Apple Silicon 上从零理解大模型。

本项目实现并对比多个主流 LLM 架构（GPT-2、Qwen3），每个模型提供两套实现：

- **`torch`**：使用 PyTorch 高层 API，可直接运行
- **`scratch`**：与 `torch` 接口一致，算子手动实现

---

## 模型路线图

| 模型 | 状态 | 架构要点 |
|------|------|----------|
| `nano_gpt2` | ✅ torch 完成 | Learned Pos Emb · MHA · Pre-LayerNorm · GELU MLP |
| `nano_qwen3` | 🚧 计划中 | RoPE · GQA · RMSNorm · SwiGLU MLP |

---

## 项目结构

```
Toy-Qwen/
├── configs/        # 超参配置
├── models/
│   └── layers/
│       ├── torch/      # 参考实现
│       └── scratch/    # 手动实现
├── optim/
│   ├── torch/
│   └── scratch/
├── data/
│   ├── raw/        # 原始数据集
│   ├── cache/      # tokenize 后的二进制缓存
│   ├── prepare.py
│   └── dataset.py
├── utils/
├── train.py
└── infer.py
```

---

## 快速开始

**环境**：Python 3.12+，推荐 [uv](https://github.com/astral-sh/uv)

```bash
uv sync

# 准备数据（TinyStories，~470 MB）
uv run toy-prepare

# 训练
uv run toy-train                        # 默认：nano_gpt2，torch 实现，自动选择 MPS/CUDA/CPU
uv run toy-train --impl scratch         # scratch 实现

# 推理（更新 infer.py 中的 _DEFAULT_CKPT 后）
uv run toy-infer
uv run toy-infer --prompt "Once upon a time" --tokens 300
```

---

## 数据集

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)：专为小模型设计的儿童故事语料，适合在笔记本上验证模型能否生成连贯文本。Tokenizer 使用 `tiktoken cl100k_base`。
