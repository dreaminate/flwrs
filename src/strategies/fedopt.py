from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, Parameters, FitRes

def _weighted_mean(weighted: List[Tuple[float, List[np.ndarray]]]) -> List[np.ndarray]:
    total_w = sum(w for w, _ in weighted) + 1e-12
    n_layers = len(weighted[0][1])
    out: List[np.ndarray] = []
    for i in range(n_layers):
        out.append(sum(w * arrs[i] for w, arrs in weighted) / total_w)
    return out

class FedOptLike(FedAvg):
    """
    FedOpt 风格（类似 FedAdam/FedYogi）：
    - 先做加权平均得到“目标参数”（或 delta）
    - 再用服务器端优化器（Adam/Yogi）更新全局参数
    """
    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        variant: str = "adam",     # "adam" | "yogi"
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.variant = variant
        self._params: Optional[List[np.ndarray]] = None
        self._m: Optional[List[np.ndarray]] = None
        self._v: Optional[List[np.ndarray]] = None
        self._t: int = 0

    def initialize_parameters(self, client_manager):
        p = super().initialize_parameters(client_manager)
        if p is not None:
            self._params = parameters_to_ndarrays(p)
            self._m = [np.zeros_like(x) for x in self._params]
            self._v = [np.zeros_like(x) for x in self._params]
            self._t = 0
        return p

    def aggregate_fit(self, server_round: int, results: List[Tuple[str, FitRes]], failures):
        if not results:
            return None, {}
        weighted = []
        for _, fit_res in results:
            weighted.append((float(fit_res.num_examples), parameters_to_ndarrays(fit_res.parameters)))
        target = _weighted_mean(weighted)  # 目标参数（平均后的全局候选）

        if self._params is None:
            self._params = target
            return ndarrays_to_parameters(self._params), {}

        # g = target - current 作为“梯度”
        g = [t - p for t, p in zip(target, self._params)]

        # Adam/Yogi 更新
        self._t += 1
        for i in range(len(self._params)):
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * g[i]
            if self.variant == "yogi":
                self._v[i] = self._v[i] - (1 - self.beta2) * np.sign(self._v[i] - g[i] * g[i]) * (g[i] * g[i])
            else:  # adam
                self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (g[i] * g[i])

            m_hat = self._m[i] / (1 - self.beta1 ** self._t)
            v_hat = self._v[i] / (1 - self.beta2 ** self._t)
            self._params[i] = self._params[i] + self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return ndarrays_to_parameters(self._params), {}
