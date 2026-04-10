"""
参数初始化函数 — scratch 实现，对应 torch.nn.init。
"""

import math

from torch import Tensor


def kaiming_uniform(t: Tensor, fan_in: int) -> None:
    """Kaiming uniform 初始化，匹配 nn.Linear 的默认行为。

    目标：让线性层输出的方差等于输入的方差，防止信号随层数增大或缩小。

    推导：
      对 y = x @ W.T，若 x 和 W 独立且均值为 0：
        Var(y) = fan_in * Var(x) * Var(W)
      令 Var(y) = Var(x)，需要 Var(W) = 1 / fan_in。

      均匀分布 Uniform(-a, a) 的方差 = a^2 / 3，令其等于 1/fan_in：
        a = sqrt(3 / fan_in)

      nn.Linear 假设激活函数为 LeakyReLU(a=sqrt(5))，gain 修正后化简为：
        bound = 1 / sqrt(fan_in)

    结论：W ~ Uniform(-bound, bound)，bound = 1 / sqrt(fan_in)
    """
    bound = 1.0 / math.sqrt(fan_in)
    t.data.uniform_(-bound, bound)
