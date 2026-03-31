"""
Training entry point.

Config is loaded in priority order (low → high):
  1. configs/<model>.py  — model architecture + training hyperparams (incl. device)
  2. CLI args            — --model, --device, --max-steps, etc.

Usage:
    uv run toy-train
    uv run toy-train --model nano_gpt2_torch
    uv run toy-train --device cpu --max-steps 100000
"""

import argparse
import time

import torch
import torch.nn.functional as F

from data.dataset import build_loaders
from models.registry import REGISTRY
from utils.checkpoint import EarlyStopper, save_best_checkpoint, save_checkpoint, save_config, setup_run_dir
from utils.generate import generate_sample
from utils.logger import setup_logger


# ---------------------------------------------------------------------------
# Training internals
# ---------------------------------------------------------------------------

def _train_step(model, x, y, optimizer, device: str) -> float:
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def _infinite_loader(loader):
    """Yield batches from loader indefinitely."""
    while True:
        yield from loader


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a toy LLM")
    p.add_argument("--model",      default="nano_gpt2_torch", choices=list(REGISTRY.keys()))
    p.add_argument("--device",     default=None, help="Override TrainConfig.device")
    p.add_argument("--max-steps",  type=int,   default=None, help="Override TrainConfig.max_steps")
    p.add_argument("--batch-size", type=int,   default=None, help="Override TrainConfig.batch_size")
    p.add_argument("--lr",         type=float, default=None, help="Override TrainConfig.lr")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def eval_loss(model, loader, steps: int, device: str) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= steps:
                break # 只看前 steps 个 batch 来估算 val loss
            x, y = x.to(device), y.to(device)
            logits = model(x)
            # 把 (B, T, vocab) 展平成 (B*T, vocab) ；logits: (16, 256, 100277)
            total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    model.train()
    return total / steps


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train() -> None:
    args = _parse_args()

    entry = REGISTRY[args.model]
    mcfg  = entry.model_cfg_cls()
    tcfg  = entry.train_cfg_cls()
    icfg  = entry.infer_cfg_cls()

    # CLI overrides
    if args.device     is not None: tcfg.device     = args.device
    if args.max_steps  is not None: tcfg.max_steps  = args.max_steps
    if args.batch_size is not None: tcfg.batch_size = args.batch_size
    if args.lr         is not None: tcfg.lr         = args.lr

    device = tcfg.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"

    run_dir = setup_run_dir(args.model)
    save_config(mcfg, tcfg, run_dir)
    logger = setup_logger(run_dir / "train.log")

    train_loader, val_loader, _ = build_loaders(
        tcfg.cache_dir, mcfg.context_len, tcfg.batch_size
    )

    model = entry.model_cls(mcfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
    )

    logger.info(f"Run dir : {run_dir}")
    logger.info(f"Model   : {args.model} | params: {model.n_params():,} | device: {device}")
    logger.info(f"Steps   : max {tcfg.max_steps:,} | eval every {tcfg.eval_interval:,} | ckpt every {tcfg.ckpt_interval:,}")
    logger.info(f"Patience: {tcfg.early_stop_patience} evals without improvement")
    logger.info("")

    early_stopper = EarlyStopper(tcfg.early_stop_patience)
    step = 0
    running_loss = 0.0
    t0 = time.time()

    for x, y in _infinite_loader(train_loader):
        step += 1
        running_loss += _train_step(model, x, y, optimizer, device)

        if step % tcfg.eval_interval == 0:
            avg_train_loss = running_loss / tcfg.eval_interval
            running_loss = 0.0

            val_loss = eval_loss(model, val_loader, tcfg.eval_steps, device)
            sample   = generate_sample(
                model, icfg.prompt, icfg.max_new_tokens,
                icfg.temperature, mcfg.context_len, device,
            )
            elapsed = time.time() - t0

            best = early_stopper.is_best(val_loss)
            should_stop = early_stopper.step(val_loss)

            logger.info(f"step {step:7,d} | train {avg_train_loss:.4f} | val {val_loss:.4f} | {elapsed:.0f}s{'  [best]' if best else ''}")
            logger.info(f"  sample: {sample!r}")

            if best:
                save_best_checkpoint(model, optimizer, step, val_loss, run_dir)
                logger.info(f"  best checkpoint saved → {run_dir.name}")

            if step % tcfg.ckpt_interval == 0:
                save_checkpoint(model, optimizer, step, val_loss, run_dir)
                logger.info(f"  checkpoint saved → {run_dir.name}")

            if should_stop:
                logger.info("")
                logger.info(f"Early stopping: val loss has not improved for {tcfg.early_stop_patience} evals.")
                break

        if step >= tcfg.max_steps:
            logger.info(f"Reached max_steps={tcfg.max_steps:,}, stopping.")
            break

    logger.info("")
    logger.info(f"Done. Best val loss: {early_stopper.best_val_loss:.4f}")
    logger.info(f"Artifacts saved to: {run_dir}")
    logger.handlers.clear()


def main() -> None:
    train()


if __name__ == "__main__":
    main()
