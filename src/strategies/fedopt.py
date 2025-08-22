# src/strategies/fedopt.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from flwr.server.strategy import FedAvg
from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    Parameters,
    FitRes,
)

def _weighted_mean(weighted: List[Tuple[float, List[np.ndarray]]]) -> List[np.ndarray]:
    """加权平均一组参数列表"""
    if not weighted:
        return []
    total_w = float(sum(w for w, _ in weighted))
    if total_w <= 0:
        total_w = 1.0
    n_layers = len(weighted[0][1])
    out: List[np.ndarray] = []
    for i in range(n_layers):
        s = None
        for w, arrs in weighted:
            s = arrs[i] * (w / total_w) if s is None else s + arrs[i] * (w / total_w)
        out.append(s)
    return out


class FedOptLike(FedAvg):
    """
    FedOpt 风格（类似 FedAdam/FedYogi）：
    1) 先做加权平均得到“目标参数”（target）
    2) 再用服务器端优化器（Adam 或 Yogi）更新全局参数 self._params
    """

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        variant: str = "adam",  # "adam" | "yogi"（不区分大小写）
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.variant = variant.lower().strip()

        self._params: Optional[List[np.ndarray]] = None  # 服务器端当前全局参数
        self._m: Optional[List[np.ndarray]] = None       # 一阶矩
        self._v: Optional[List[np.ndarray]] = None       # 二阶矩
        self._t: int = 0                                 # 步数（做偏置校正）

    # 若上游提供了初始参数，这里会被调用
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        p = super().initialize_parameters(client_manager)
        if p is not None:
            self._params = parameters_to_ndarrays(p)
            self._m = [np.zeros_like(x) for x in self._params]
            self._v = [np.zeros_like(x) for x in self._params]
            self._t = 0
        return p

    def _ensure_moments(self) -> None:
        """在首次拿到 _params 时，确保动量缓冲区已初始化。"""
        if self._params is not None and (self._m is None or self._v is None):
            self._m = [np.zeros_like(x) for x in self._params]
            self._v = [np.zeros_like(x) for x in self._params]
            self._t = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[str, FitRes]],
        failures,
    ) -> Optional[Tuple[Parameters, Dict]]:
        # 无结果时，遵循 Flower 约定返回 None
        if not results:
            return None

        # 1) FedAvg 加权平均 → target 参数
        weighted: List[Tuple[float, List[np.ndarray]]] = []
        for _, fit_res in results:
            weighted.append((float(fit_res.num_examples), parameters_to_ndarrays(fit_res.parameters)))
        target = _weighted_mean(weighted)

        # 2) 首次没有全局参数：用 target 初始化，同时初始化动量，然后直接返回
        if self._params is None:
            self._params = [arr.copy() for arr in target]
            self._ensure_moments()
            return ndarrays_to_parameters(self._params), {}

        # 3) 计算 “梯度” g = target - current
        g = [t - p for t, p in zip(target, self._params)]

        # 4) Adam / Yogi 更新
        self._ensure_moments()
        self._t += 1
        b1, b2, eps = self.beta1, self.beta2, self.eps

        for i in range(len(self._params)):
            gi = g[i]
            # 一阶矩
            self._m[i] = b1 * self._m[i] + (1.0 - b1) * gi
            # 二阶矩
            if self.variant == "yogi":
                # Yogi: v <- v - (1-b2) * g^2 * sign(v - g^2)
                gi2 = gi * gi
                self._v[i] = self._v[i] - (1.0 - b2) * gi2 * np.sign(self._v[i] - gi2)
            else:
                # Adam: v <- b2*v + (1-b2)*g^2
                self._v[i] = b2 * self._v[i] + (1.0 - b2) * (gi * gi)

            # 偏置校正
            m_hat = self._m[i] / (1.0 - b1 ** self._t)
            v_hat = self._v[i] / (1.0 - b2 ** self._t)

            # 服务器端一步：向 target 方向迈一步
            self._params[i] = self._params[i] + self.lr * (m_hat / (np.sqrt(v_hat) + eps))

        return ndarrays_to_parameters(self._params), {}
