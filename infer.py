"""
Inference entry point. Loads a checkpoint from a run directory and generates text.

Config defaults come from InferConfig in configs/<model>.py.

Usage:
    uv run toy-infer
    uv run toy-infer --model nano_gpt2_torch
    uv run toy-infer --run runs/nano_gpt2_torch_20260330_233710 --prompt "The dragon"
    uv run toy-infer --tokens 300 --temperature 0.9
"""

import argparse
import json
from pathlib import Path

import torch

from models.registry import REGISTRY
from utils.generate import generate_sample


def _parse_args(icfg) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a saved checkpoint")
    p.add_argument("--model",       default="nano_gpt2_torch",   choices=list(REGISTRY.keys()))
    p.add_argument("--run",         default=icfg.run_dir,         help="Path to run directory")
    p.add_argument("--ckpt",        default=icfg.ckpt,            help="Checkpoint filename inside --run")
    p.add_argument("--prompt",      default=icfg.prompt,          help="Text prompt to continue")
    p.add_argument("--tokens",      type=int,   default=icfg.max_new_tokens, help="Number of tokens to generate")
    p.add_argument("--temperature", type=float, default=icfg.temperature,    help="Sampling temperature (0 = greedy)")
    p.add_argument("--device",      default=None, help="Override device (default: from config.json)")
    return p.parse_args()


def main() -> None:
    # First pass: get --model to load InferConfig defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", default="nano_gpt2_torch", choices=list(REGISTRY.keys()))
    known, _ = pre.parse_known_args()

    icfg = REGISTRY[known.model].infer_cfg_cls()
    args = _parse_args(icfg)

    run_dir   = Path(args.run)
    ckpt_path = run_dir / args.ckpt
    cfg_path  = run_dir / "config.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found: {cfg_path}")

    entry  = REGISTRY[args.model]
    cfg    = json.loads(cfg_path.read_text())
    mcfg   = entry.model_cfg_cls(**cfg["model"])
    device = args.device or cfg["train"].get("device", "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    model = entry.model_cls(mcfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])

    print(f"Loaded {args.model}  step={ckpt['step']:,}  val_loss={ckpt['val_loss']:.4f}  device={device}")
    print(f"Prompt: {args.prompt!r}")
    print()

    output = generate_sample(
        model, args.prompt, args.tokens, args.temperature, mcfg.context_len, device
    )
    print(output)


if __name__ == "__main__":
    main()
