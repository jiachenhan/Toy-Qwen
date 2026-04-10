"""
Inference entry point. Loads a checkpoint from a run directory and generates text.

配置优先级（低 → 高）：
  推理默认值  configs/<model>.py :: InferConfig  — prompt / tokens / temperature
  运行时记录  config.json                        — model_name / impl
  CLI 覆盖    --prompt / --tokens / --temperature / --device

_DEFAULT_CKPT 是唯一需要手动更新的位置，每次训完改这一行。

Usage:
    uv run toy-infer
    uv run toy-infer --ckpt runs/nano_gpt2_torch_20260331_120000/ckpt_step_0010000.pt
    uv run toy-infer --prompt "The dragon" --tokens 300
"""

import argparse
import json
from pathlib import Path

import torch

from models.registry import REGISTRY
from utils.generate import generate_sample
from utils.tool import auto_device

# 每次训完只需更新这一行
_DEFAULT_CKPT = "runs/nano_gpt2_torch_20260331_224551/ckpt_best.pt"


def _parse_args(icfg) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a saved checkpoint")

    # 每次训完只需更新 _DEFAULT_CKPT
    p.add_argument("--ckpt",        default=_DEFAULT_CKPT)

    # 推理默认值 — 来自 InferConfig，可 CLI 覆盖
    p.add_argument("--prompt",      default=icfg.prompt)
    p.add_argument("--tokens",      type=int,   default=icfg.max_new_tokens)
    p.add_argument("--temperature", type=float, default=icfg.temperature)

    # 实验变量覆盖 — 默认自动识别本地环境
    p.add_argument("--device",      default=auto_device())

    return p.parse_args()


def main() -> None:
    # 第一步：解析 ckpt 路径 → 加载 config.json → 确定模型和 InferConfig 默认值
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--ckpt", default=_DEFAULT_CKPT)
    known, _ = pre.parse_known_args()

    cfg_path = Path(known.ckpt).parent / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found: {cfg_path}")

    cfg        = json.loads(cfg_path.read_text())
    model_name = cfg["model_name"]
    entry      = REGISTRY[model_name]

    # 第二步：用 InferConfig 默认值完整解析所有参数
    icfg = entry.infer_cfg_cls()
    args = _parse_args(icfg)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    mcfg   = entry.model_cfg_cls(**cfg["model"])
    device = args.device

    model = entry.model_cls(mcfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])

    print(f"Loaded {model_name}  step={ckpt['step']:,}  val_loss={ckpt['val_loss']:.4f}  device={device}")
    print(f"Prompt: {args.prompt!r}")
    print()

    output = generate_sample(
        model, args.prompt, args.tokens, args.temperature, mcfg.context_len, device
    )
    print(output)


if __name__ == "__main__":
    main()
