"""
AdamW optimizer — scratch implementation (hand-rolled).

Implements the decoupled weight decay variant (Loshchilov & Hutter, 2019).
Reference: https://arxiv.org/abs/1711.05101

Reference torch implementation: optim/torch/adamw.py
"""

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """AdamW with decoupled weight decay — scratch implementation.

    Implement the parameter update:
      m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
      v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
      m_hat = m_t / (1 - beta1^t)
      v_hat = v_t / (1 - beta2^t)
      theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta_{t-1})
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        raise NotImplementedError
