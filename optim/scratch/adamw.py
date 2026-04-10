"""
AdamW optimizer — scratch implementation (hand-rolled).

Implements the decoupled weight decay variant (Loshchilov & Hutter, 2019).
Reference: https://arxiv.org/abs/1711.05101

Reference torch implementation: optim/torch/adamw.py
"""

import torch
from collections.abc import Callable
from typing import overload

from torch.optim import Optimizer


class AdamW(Optimizer):
    """AdamW（解耦权重衰减）— scratch 实现。

    ── 直觉：SGD 的两个问题 ──────────────────────────────────────────

    普通 SGD：theta -= lr * g
      问题1：单步梯度噪声大，方向不稳定。
      问题2：所有参数用同一个 lr，但不同参数的梯度尺度可能差几个数量级，
             谁也不知道该设多大的 lr 才能让所有参数都更新得合理。

    Adam 的解法：为每个参数维护两个历史统计量，用来稳定方向、均衡步长：
      m（一阶矩）：梯度的历史加权均值 → 平滑单步噪声，让方向更稳定
      v（二阶矩）：梯度平方的历史加权均值 → 估计该参数梯度的典型幅度

    ── 什么是矩，为什么用二阶矩而不是方差 ──────────────────────────

    统计学里，k 阶矩 = E[X^k]：
      一阶矩 E[g]    = 梯度均值，描述"梯度平均指向哪个方向"
      二阶矩 E[g^2]  = 梯度平方的均值，描述"梯度的典型幅度有多大"
      方差   Var[g]  = E[g^2] - E[g]^2 = 二阶矩 - 一阶矩的平方

    Adam 用二阶矩 v ≈ E[g^2] 做分母，而不是方差，原因：
      方差只衡量梯度的"波动程度"（噪声），
      二阶矩 = 方差 + 均值²，同时包含噪声和梯度本身的幅度。

      如果某参数梯度方向一致、幅度很大（低噪声但高均值）：
        方差很小 → 用方差做分母会给大步长 → 容易冲过头
        二阶矩很大 → 步长小 → 更保守，合理

      所以用 E[g^2] 而不是 E[g]：平方消掉了符号，左右震荡的梯度不会相互抵消，
      sqrt(E[g^2])（即 RMS）始终反映梯度的真实幅度，和方向无关。
      可以证明 |E[g]| <= sqrt(E[g^2]) 恒成立，差距就是方差项 sqrt(Var(g))。
        梯度历来幅度大（v 大）→ 1/sqrt(v) 小 → 步长小
        梯度历来幅度小（v 小）→ 1/sqrt(v) 大 → 步长大

      效果：无论某个参数的梯度天然在 0.01 还是 100 量级，
      经过归一化后各参数都在相近的比例下更新，优化更均衡稳定。

    ── 更新规则与偏差修正 ────────────────────────────────────────────

    指数移动平均（EMA）：不存所有历史梯度，只保留一个滚动平均，
    新值权重小（1-beta），旧值权重大（beta），越近的梯度影响越大。

      m_t = beta1 * m_{t-1} + (1 - beta1) * g_t       # 一阶矩 EMA，beta1 通常 0.9
      v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2     # 二阶矩 EMA，beta2 通常 0.999

    偏差修正：m 和 v 初始化为 0，把 EMA 递推展开成求和式：
      m_t = (1 - beta1) * sum_{i=1}^{t} beta1^{t-i} * g_i
    两边取期望（假设各步 E[g] 相同）：
      E[m_t] = E[g] * (1 - beta1^t)
    所以 m_t 系统性地比真实期望小了 (1 - beta1^t) 倍，除以它即可还原：
      m_hat = m_t / (1 - beta1^t)  →  E[m_hat] = E[g]

    这个偏差随 t 增大自动消失（t=1 放大 10x，t=100 几乎不放大），
    只在训练最初几百步有实质影响。

      m_hat = m_t / (1 - beta1^t)
      v_hat = v_t / (1 - beta2^t)

      theta -= lr * m_hat / (sqrt(v_hat) + eps)
                ↑ 方向：平滑后的梯度均值
                               ↑ 幅度：被各参数自己的历史梯度幅度归一化

    ── 权重衰减（Weight Decay）与 Adam 的核心区别 ───────────────────

    目的：防止权重过大（过拟合），训练中持续把权重往 0 拉。

    L2 正则化：在 loss 上加 (lambda/2) * ||theta||^2，
    对梯度求导后等价于每步额外加一个 lambda * theta 到梯度里：
      g_eff = g + lambda * theta
      theta -= lr * g_eff

    Adam + L2 的问题：这个额外的 lambda * theta 也会被 1/sqrt(v) 缩放，
      theta -= lr * (m_hat + lambda * theta) / (sqrt(v_hat) + eps)
    导致 v 大的参数（梯度活跃）实际衰减弱，v 小的参数衰减强，
    不同参数受到的正则化强度不一样，lambda 的效果难以预期。

    AdamW 的修正（解耦）：把权重衰减从梯度里拿出来，直接乘在参数上，
    绕过自适应缩放，所有参数以相同比例衰减：
      theta_t = (1 - lr * lambda) * theta_{t-1}        # 先等比例缩小参数
              - lr * m_hat / (sqrt(v_hat) + eps)        # 再做 Adam 梯度步
    这两步独立，衰减强度不受 v 影响，正则化强度对所有参数一视同仁。
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

    @overload
    def step(self, closure: None = ...) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        # 某些优化器（比如 L-BFGS）在一次 step 里需要多次重新计算 loss（做线搜索），所以把"计算 loss 的函数"传进来反复调用。
        # 忽略closure
        loss = None
        if closure is not None:
            loss = closure()

        # 对于 AdamW(model.parameters(), lr=3e-4) 这种写法，所有的参数共用一组超参（lr、betas、eps、weight_decay）
        # 所以 param_groups 里只有一个 group。
        for group in self.param_groups:
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            eps          = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    # 跳过没参与前向的参数（比如某层的权重在前向里没被用到，或者冻结了 requires_grad=False）
                    continue

                g     = p.grad.data # 当前梯度
                state = self.state[p] # 历史状态：包含 step 计数器 m（梯度的一阶矩） v（梯度平方的二阶矩）

                # 第一次 step 时初始化状态
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)  # 一阶矩，初始化为 0
                    state["v"] = torch.zeros_like(p.data)  # 二阶矩，初始化为 0

                state["step"] += 1
                t    = state["step"]
                m, v = state["m"], state["v"]

                # 1. 更新一阶矩：梯度的指数移动平均，
                #    原地操作避免每步分配新 tensor
                #   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                m.mul_(beta1)
                m.add_((1 - beta1) * g)

                # 2. 更新二阶矩：梯度平方的指数移动平均
                #   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                v.mul_(beta2)
                v.add_((1 - beta2) * g * g)

                # 3. 偏差修正：m_hat = m_t / (1 - beta1^t)，v_hat = v_t / (1 - beta2^t)
                #    初始化为 0 导致早期低估，除以 (1-beta^t) 还原真实期望；t 增大后自动退化为 1
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # 4. 解耦权重衰减：theta = (1 - lr * lambda) * theta
                #    直接缩小参数，不经过 1/sqrt(v) 缩放，所有参数衰减比例相同（区别于 Adam+L2）
                p.data.mul_(1 - lr * weight_decay)

                # 5. Adam 梯度步：theta -= lr * m_hat / (sqrt(v_hat) + eps)
                #    m_hat 决定方向，1/sqrt(v_hat) 归一化各参数的步长幅度
                p.data.sub_(lr * m_hat / (v_hat.sqrt() + eps))

        return loss
