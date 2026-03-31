"""
Model registry: maps model name → (model_class, model_config, train_config).

To add a new model:
  1. Implement the model class in models/
  2. Add its configs to configs/
  3. Add one entry here
"""

from dataclasses import dataclass


@dataclass
class RegistryEntry:
    model_cls: type
    model_cfg_cls: type
    train_cfg_cls: type
    infer_cfg_cls: type


def _build() -> dict[str, RegistryEntry]:
    # Import lazily so missing optional deps don't break unrelated models
    from configs.nano_gpt2 import InferConfig as NanoGPT2InferCfg
    from configs.nano_gpt2 import ModelConfig as NanoGPT2ModelCfg
    from configs.nano_gpt2 import TrainConfig as NanoGPT2TrainCfg
    from models.nano_gpt2_torch import NanoGPT2Torch

    return {
        "nano_gpt2_torch": RegistryEntry(
            model_cls=NanoGPT2Torch,
            model_cfg_cls=NanoGPT2ModelCfg,
            train_cfg_cls=NanoGPT2TrainCfg,
            infer_cfg_cls=NanoGPT2InferCfg,
        ),
    }


REGISTRY: dict[str, RegistryEntry] = _build()
