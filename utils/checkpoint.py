import dataclasses
import json
import time
from pathlib import Path

import torch


class EarlyStopper:
    """Stops training when val loss stops improving for `patience` consecutive evals."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_val_loss = float("inf")
        self._bad_evals = 0

    def is_best(self, val_loss: float) -> bool:
        return val_loss < self.best_val_loss

    def step(self, val_loss: float) -> bool:
        """Call after each eval. Returns True if training should stop."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._bad_evals = 0
            return False
        self._bad_evals += 1
        return self._bad_evals >= self.patience


def setup_run_dir(model_name: str) -> Path:
    """Create a timestamped run directory under runs/."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(mcfg, tcfg, run_dir: Path, *, model_name: str, impl: str, device: str) -> None:
    config = {
        "model_name": model_name,
        "model": dataclasses.asdict(mcfg),
        "train": dataclasses.asdict(tcfg),
        "run":   {"impl": impl, "device": device},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))


def save_checkpoint(
    model, optimizer, step: int, val_loss: float, run_dir: Path,
) -> None:
    """Save a numbered checkpoint (called on ckpt_interval schedule)."""
    ckpt = {
        "step": step,
        "val_loss": val_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(ckpt, run_dir / f"ckpt_step_{step:07d}.pt")


def save_best_checkpoint(
    model, optimizer, step: int, val_loss: float, run_dir: Path,
) -> None:
    """Overwrite ckpt_best.pt whenever a new best val loss is achieved."""
    ckpt = {
        "step": step,
        "val_loss": val_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(ckpt, run_dir / "ckpt_best.pt")
