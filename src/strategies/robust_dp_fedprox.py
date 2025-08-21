from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import (
    parameters_to_ndarrays, ndarrays_to_parameters, FitRes, Parameters, Scalar
)

def _l2_clip(arrs: List[np.ndarray], max_norm: float) -> Tuple[List[np.ndarray], float]:
    total = 0.0
    for a in arrs:
        total += np.sum(a.astype(np.float64) ** 2)
    norm = float(np.sqrt(total) + 1e-12)
    scale = min(1.0, max_norm / norm)
    return [a * scale for a in arrs], norm

def _gaussian_noise(arrs: List[np.ndarray], sigma: float) -> List[np.ndarray]:
    if sigma <= 0: 
        return arrs
    rng = np.random.default_rng()
    return [a + rng.normal(0.0, sigma, size=a.shape).astype(a.dtype) for a in arrs]

def _aggregate_median(weighted: List[Tuple[float, List[np.ndarray]]]) -> List[np.ndarray]:
    n_layers = len(weighted[0][1])
    stacks = [np.stack([arrs[i] for _, arrs in weighted], axis=0) for i in range(n_layers)]
    return [np.median(s, axis=0) for s in stacks]

def _aggregate_trimmed_mean(weighted, trim_ratio=0.1):
    n_layers = len(weighted[0][1])
    out = []
    for i in range(n_layers):
        X = np.stack([arrs[i] for _, arrs in weighted], axis=0)
        k = int(X.shape[0] * trim_ratio)
        X_sort = np.sort(X, axis=0)
        X_trim = X_sort[k: X.shape[0]-k] if 2*k < X.shape[0] else X_sort
        out.append(np.mean(X_trim, axis=0))
    return out

class RobustDPFedProx(FedAvg):
    """
    单模型族策略：鲁棒聚合 + DP（delta裁剪/噪声）+ FedProx（mu由server下发，client在本地loss里加正则）
    用于 baseline/对照实验；多模型同轮请用 MultiModelCohort。
    """
    def __init__(
        self,
        dp_max_norm: float = 1.0,
        dp_noise_sigma: float = 0.0,
        robust: str = "weighted",          # "weighted" | "median" | "trimmed"
        trim_ratio: float = 0.1,
        mu_prox: float = 0.0,              # FedProx μ；客户端需按此加本地正则
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dp_max_norm = dp_max_norm
        self.dp_noise_sigma = dp_noise_sigma
        self.robust = robust
        self.trim_ratio = trim_ratio
        self.mu_prox = mu_prox
        self._current_params: Optional[List[np.ndarray]] = None

    def initialize_parameters(self, client_manager):
        p = super().initialize_parameters(client_manager)
        if p is not None:
            self._current_params = parameters_to_ndarrays(p)
        return p

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        ret = super().configure_fit(server_round, parameters, client_manager)
        if isinstance(ret, tuple):
            _, ins = ret
        else:
            ins = ret
        # 将 μ 下发给客户端以启用 FedProx，本地在 loss 中加 0.5*mu*||w-w_global||^2
        for client, fitins in ins:
            fitins.config["mu_prox"] = float(self.mu_prox)
        return ins

    def aggregate_fit(self, server_round: int, results: List[Tuple[str, FitRes]], failures):
        if not results:
            return None, {}

        weighted_deltas = []
        for _, fit_res in results:
            client_params = parameters_to_ndarrays(fit_res.parameters)
            if self._current_params is None:
                self._current_params = client_params
            delta = [c - g for c, g in zip(client_params, self._current_params)]
            # DP：delta 裁剪 + 噪声
            delta, _ = _l2_clip(delta, self.dp_max_norm)
            delta = _gaussian_noise(delta, self.dp_noise_sigma)
            w = float(fit_res.num_examples)
            weighted_deltas.append((w, delta))

        # 鲁棒聚合
        if self.robust == "median":
            agg_delta = _aggregate_median(weighted_deltas)
        elif self.robust == "trimmed":
            agg_delta = _aggregate_trimmed_mean(weighted_deltas, self.trim_ratio)
        else:
            total_w = sum(w for w, _ in weighted_deltas) + 1e-12
            agg_delta = []
            for i in range(len(weighted_deltas[0][1])):
                agg_delta.append(sum(w * arrs[i] for w, arrs in weighted_deltas) / total_w)

        # 更新全局
        new_params = [g + d for g, d in zip(self._current_params, agg_delta)]
        self._current_params = new_params
        return ndarrays_to_parameters(new_params), {}
