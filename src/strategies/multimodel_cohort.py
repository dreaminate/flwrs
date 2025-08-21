from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Optional, Any
import hashlib
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
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

def _aggregate_trimmed_mean(weighted: List[Tuple[float, List[np.ndarray]]], trim_ratio: float = 0.1) -> List[np.ndarray]:
    n_layers = len(weighted[0][1])
    out: List[np.ndarray] = []
    for i in range(n_layers):
        X = np.stack([arrs[i] for _, arrs in weighted], axis=0)
        k = int(X.shape[0] * trim_ratio)
        if 2 * k < X.shape[0]:
            X = np.sort(X, axis=0)[k : X.shape[0] - k]
        out.append(np.mean(X, axis=0))
    return out

def _sig_from_arrays(arrs: List[np.ndarray]) -> Dict[str, Any]:
    shapes = [tuple(a.shape) for a in arrs]
    dtypes = [str(a.dtype) for a in arrs]
    payload = f"{len(arrs)}|" + "|".join(f"{s}:{d}" for s, d in zip(shapes, dtypes))
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return {"sig_count": len(arrs), "sig_hash": h}

class MultiModelCohort(FedAvg):
    """
    多模型同轮：按“模型指纹(signature)”自动匹配。
    - 初始化时为每个模型族计算签名 self.sig_map[name] = {sig_count, sig_hash}
    - 在 configure_fit / configure_evaluate 中，通过 client.get_properties() 读取客户端指纹，
      首选“指纹完全匹配”的模型族；否则回落到 assign_fn。
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
        # 预计算每个族的签名
        self.sig_map: Dict[str, Dict[str, Any]] = {name: _sig_from_arrays(arrs) for name, arrs in init_params_map.items()}
        # 记住“已知映射”：cid -> model_name（拿过属性后缓存起来，加速后续轮次）
        self.cid2model: Dict[str, str] = {}

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        any_name = next(iter(self.params_map))
        return ndarrays_to_parameters(self.params_map[any_name])

    # ---- 工具：查询客户端属性（兼容不同返回形态） ----
    def _get_client_props(self, client_proxy) -> Dict[str, Any]:
        try:
            res = client_proxy.get_properties({})  # 可能返回对象或 dict
            props = getattr(res, "properties", None) or res  # 兼容
            return dict(props)
        except Exception:
            return {}

    def _match_by_signature(self, props: Dict[str, Any]) -> Optional[str]:
        sc, sh = props.get("sig_count"), props.get("sig_hash")
        if sc is None or sh is None:
            return None
        for name, sig in self.sig_map.items():
            if sig["sig_count"] == sc and sig["sig_hash"] == sh:
                return name
        return None

    def _decide_model_for_client(self, client) -> str:
        cid = client.cid
        # 1) 有缓存则直接用
        if cid in self.cid2model:
            return self.cid2model[cid]
        # 2) 读属性 → 指纹匹配
        props = self._get_client_props(client)
        mname = self._match_by_signature(props)
        if mname is None:
            # 3) 没有指纹就回落到 assign_fn
            mname = self.assign_fn(cid)
        # 4) 缓存映射
        self.cid2model[cid] = mname
        return mname

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        ret = super().configure_fit(server_round, parameters, client_manager)
        ins = ret[1] if isinstance(ret, tuple) else ret  # List[(ClientProxy, FitIns)]
        for client, fitins in ins:
            mname = self._decide_model_for_client(client)
            fitins.parameters = ndarrays_to_parameters(self.params_map[mname])
            # 把签名也下发给客户端，用于二次校验
            sig = self.sig_map[mname]
            fitins.config.update({"model_name": mname, "sig_count": sig["sig_count"], "sig_hash": sig["sig_hash"]})
        return ins

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        ret = super().configure_evaluate(server_round, parameters, client_manager)
        ins = ret[1] if isinstance(ret, tuple) else ret  # List[(ClientProxy, EvaluateIns)]
        for client, evalins in ins:
            mname = self._decide_model_for_client(client)
            evalins.parameters = ndarrays_to_parameters(self.params_map[mname])
            sig = self.sig_map[mname]
            evalins.config.update({"model_name": mname, "sig_count": sig["sig_count"], "sig_hash": sig["sig_hash"]})
        return ins

    def aggregate_fit(self, server_round: int, results: List[Tuple], failures):
        if not results:
            any_name = next(iter(self.params_map))
            return ndarrays_to_parameters(self.params_map[any_name]), {}

        buckets: Dict[str, List[Tuple[float, List[np.ndarray]]]] = {}
        for client_proxy, fit_res in results:
            arrs = parameters_to_ndarrays(fit_res.parameters)
            w = float(getattr(fit_res, "num_examples", 1))
            # 优先看 metrics 里的 model_name；其次根据缓存/assign_fn 决定
            mname = None
            if fit_res.metrics:
                mname = fit_res.metrics.get("model_name")
            if not mname:
                mname = self.cid2model.get(client_proxy.cid, self.assign_fn(client_proxy.cid))
            buckets.setdefault(mname, []).append((w, arrs))

        for mname, items in buckets.items():
            if self.aggregator == "median":
                agg = _aggregate_median(items)
            elif self.aggregator == "trimmed":
                agg = _aggregate_trimmed_mean(items, self.trim_ratio)
            else:
                agg = _weighted_mean(items)
            self.params_map[mname] = agg
            # 聚合后更新该族签名（层数/shape 变更时也能跟上）
            self.sig_map[mname] = _sig_from_arrays(agg)

        any_name = next(iter(self.params_map))
        return ndarrays_to_parameters(self.params_map[any_name]), {}
