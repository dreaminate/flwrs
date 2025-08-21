from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Optional
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import (
    Parameters, Scalar, FitRes,
    ndarrays_to_parameters, parameters_to_ndarrays,
)

def _weighted_mean(weighted: List[Tuple[float, List[np.ndarray]]]) -> List[np.ndarray]:
    total_w = sum(w for w, _ in weighted) + 1e-12
    n_layers = len(weighted[0][1])
    out: List[np.ndarray] = []
    for i in range(n_layers):
        out.append(sum(w * arrs[i] for w, arrs in weighted) / total_w)
    return out

def _aggregate_median(weighted: List[Tuple[float, List[np.ndarray]]]) -> List[np.ndarray]:
    n_layers = len(weighted[0][1])
    stacks = [np.stack([arrs[i] for _, arrs in weighted], axis=0) for i in range(n_layers)]
    return [np.median(s, axis=0) for s in stacks]

def _aggregate_trimmed_mean(weighted: List[Tuple[float, List[np.ndarray]]], trim_ratio=0.1) -> List[np.ndarray]:
    n_layers = len(weighted[0][1])
    out: List[np.ndarray] = []
    for i in range(n_layers):
        X = np.stack([arrs[i] for _, arrs in weighted], axis=0)
        k = int(X.shape[0] * trim_ratio)
        if 2 * k < X.shape[0]:
            X = np.sort(X, axis=0)[k: X.shape[0] - k]
        out.append(np.mean(X, axis=0))
    return out

class MultiModelCohort(FedAvg):
    """
    同一轮内维护多套全局参数，按模型名分桶聚合。
    - params_map: {model_name: [np.ndarray...]}
    - assign_fn(cid)->model_name
    - aggregator: "weighted" | "median" | "trimmed"
    """
    def __init__(
        self,
        init_params_map: Dict[str, List[np.ndarray]],
        assign_fn: Callable[[str], str],
        aggregator: str = "weighted",
        trim_ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.params_map = init_params_map
        self.assign_fn = assign_fn
        self.aggregator = aggregator
        self.trim_ratio = trim_ratio

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        any_name = next(iter(self.params_map))
        return ndarrays_to_parameters(self.params_map[any_name])

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        # 兼容不同 Flower 版本：有的返回 (cfg, ins)，有的只返回 ins
        ret = super().configure_fit(server_round, parameters, client_manager)
        ins = ret[1] if isinstance(ret, tuple) else ret  # List[(ClientProxy, FitIns)]

        # 为每个客户端下发“匹配其模型族”的参数，并写入 model_name
        for client, fitins in ins:
            cid = client.cid
            model_name = self.assign_fn(cid)
            fitins.parameters = ndarrays_to_parameters(self.params_map[model_name])
            fitins.config["model_name"] = model_name
        return ins
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        # Flower 在评估阶段默认用“全局参数”（我们返回了 any_name），这里要覆写成 per-client 的
        ret = super().configure_evaluate(server_round, parameters, client_manager)
        ins = ret[1] if isinstance(ret, tuple) else ret  # List[(ClientProxy, EvaluateIns)]

        for client, evalins in ins:
            cid = client.cid
            model_name = self.assign_fn(cid)
            evalins.parameters = ndarrays_to_parameters(self.params_map[model_name])
            # 可选：传下去便于日志核对
            evalins.config["model_name"] = model_name
        return ins

    def aggregate_fit(self, server_round: int, results: List[Tuple], failures):
        if not results:
            any_name = next(iter(self.params_map))
            return ndarrays_to_parameters(self.params_map[any_name]), {}

        buckets: Dict[str, List[Tuple[float, List[np.ndarray]]]] = {}
        for client_proxy, fit_res in results:
            arrs = parameters_to_ndarrays(fit_res.parameters)
            w = float(getattr(fit_res, "num_examples", 1))
            mname = fit_res.metrics.get("model_name") if fit_res.metrics else None
            if not mname:
                mname = self.assign_fn(client_proxy.cid)
            buckets.setdefault(mname, []).append((w, arrs))

        for mname, items in buckets.items():
            if self.aggregator == "median":
                agg = _aggregate_median(items)
            elif self.aggregator == "trimmed":
                agg = _aggregate_trimmed_mean(items, self.trim_ratio)
            else:
                agg = _weighted_mean(items)
            self.params_map[mname] = agg

        any_name = next(iter(self.params_map))
        return ndarrays_to_parameters(self.params_map[any_name]), {}
