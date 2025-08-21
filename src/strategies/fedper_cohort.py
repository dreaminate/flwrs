# src/strategies/fedper_cohort.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math, os, time
import numpy as np
import flwr as fl

from flwr.common import (
    FitIns, EvaluateIns, Parameters,
    ndarrays_to_parameters, parameters_to_ndarrays,
)

from privacy.dp import l2_clip, add_gaussian_noise
from privacy.audit import log_jsonl, cfg_hash

# ---- 可选: 从你自己的 accountant 导入; 若不存在, 用内置兜底 ----
try:
    from privacy.accountant import PrivacyLedger  # 需要提供 .step(bucket,sigma) 和 .acc[bucket].delta
except Exception:
    from types import SimpleNamespace

    class PrivacyLedger:
        """最小可用 zCDP 账本: 用 rho 累加, 按 delta 近似换算 epsilon.
           rho += 1/(2*sigma^2);  eps = rho + 2*sqrt(rho*log(1/delta))"""
        def __init__(self, delta_map: Optional[Dict[str, float]] = None):
            self.acc: Dict[str, SimpleNamespace] = {}
            if delta_map:
                for k, delta in delta_map.items():
                    self.acc[k] = SimpleNamespace(delta=float(delta), rho=0.0, eps=0.0, steps=0)

        def step(self, bucket: str, sigma: float):
            if bucket not in self.acc:
                self.acc[bucket] = SimpleNamespace(delta=1e-5, rho=0.0, eps=0.0, steps=0)
            st = self.acc[bucket]
            if sigma and sigma > 0:
                st.rho += 1.0 / (2.0 * (sigma ** 2))
            # zCDP -> (ε, δ)
            delta = max(1e-12, float(st.delta))
            st.eps = float(st.rho + 2.0 * math.sqrt(st.rho * math.log(1.0 / delta))) if st.rho > 0 else 0.0
            st.steps += 1
            return st.eps


# ============ 小工具 ============
def _sig_from_arrays(arrs: List[np.ndarray]) -> Dict[str, Any]:
    import hashlib
    shapes = [a.shape for a in arrs]
    dtypes = [str(a.dtype) for a in arrs]
    payload = f"{len(arrs)}|" + "|".join(f"{s}:{d}" for s, d in zip(shapes, dtypes))
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return {"sig_count": len(arrs), "sig_hash": h}

def _weighted_mean(items: List[Tuple[float, List[np.ndarray]]]) -> List[np.ndarray]:
    ws = sum(w for w, _ in items) or 1.0
    n_params = len(items[0][1])
    out: List[np.ndarray] = []
    for p in range(n_params):
        stacked = np.stack([arrs[p] * (w / ws) for w, arrs in items], axis=0)
        out.append(stacked.sum(axis=0))
    return out

def _aggregate_median(items: List[Tuple[float, List[np.ndarray]]]) -> List[np.ndarray]:
    n_params = len(items[0][1])
    out: List[np.ndarray] = []
    for p in range(n_params):
        stacked = np.stack([arrs[p] for _, arrs in items], axis=0)
        out.append(np.median(stacked, axis=0))
    return out

def _aggregate_trimmed_mean(items: List[Tuple[float, List[np.ndarray]]], trim_ratio: float) -> List[np.ndarray]:
    n_params = len(items[0][1])
    out: List[np.ndarray] = []
    lo = int(math.floor(len(items) * trim_ratio))
    hi = len(items) - lo
    for p in range(n_params):
        stacked = np.stack([arrs[p] for _, arrs in items], axis=0)
        sorted_ = np.sort(stacked, axis=0)
        trimmed = sorted_[lo:hi]
        out.append(trimmed.mean(axis=0))
    return out


# ============ DP 配置 ============
@dataclass
class DpCfg:
    clip: float = 0.0
    sigma: float = 0.0
    delta: float = 1e-5


# ============ 策略实现 ============
class FedPerCohort(fl.server.strategy.Strategy):
    """
    FedPer 的“分桶聚合”版：按 (model_name 或 数组签名) 分桶，
    每桶只聚合共享子集（例如 adapter.*），互不干扰。
    """

    def __init__(
        self,
        assign_fn,                                 # cid -> bucket_name
        aggregator: str = "weighted",              # "weighted" | "median" | "trimmed"
        trim_ratio: float = 0.1,
        fraction_fit: float = 0.5,
        fraction_evaluate: float = 0.3,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        accept_failures: bool = True,
        dp_cfg_map: Optional[Dict[str, DpCfg]] = None,
        dp_mode: str = "client",                   # "client" 或 "server"
        policy_id: str = "dpv1",
        share_prefix: str = "adapter.",            # 仅用于日志标注
    ):
        self.assign_fn = assign_fn
        self.aggregator = aggregator
        self.trim_ratio = trim_ratio
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.accept_failures = accept_failures

        self.dp_cfg_map: Dict[str, DpCfg] = dp_cfg_map or {}
        self.dp_mode = dp_mode
        self.policy_id = policy_id
        self.dp_sig = cfg_hash({"dp_cfg_map": {k: vars(v) for k, v in self.dp_cfg_map.items()},
                                "dp_mode": dp_mode, "policy_id": policy_id})

        # 每桶的全局共享子集参数
        self.params_map: Dict[str, List[np.ndarray]] = {}
        self.sig_map: Dict[str, Dict[str, Any]] = {}

        # zCDP 账本（无则用兜底）
        self.ledger = PrivacyLedger({k: v.delta for k, v in self.dp_cfg_map.items()}) if self.dp_cfg_map else PrivacyLedger({})

        # 审计日志
        os.makedirs("bench_results", exist_ok=True)
        self._dp_log_path = os.path.join("bench_results", "privacy_log.jsonl")

    # ---------- Strategy 接口 ----------
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        # 返回 None 让 Flower 随机选客户端给初始参数
        return None

    # ------ 采样工具 ------
    @staticmethod
    def _count_available(cm) -> int:
        try:
            return cm.num_available()
        except Exception:
            return len(cm.wait_for(min_num_clients=1, timeout=0.1))

    def _sample(self, cm, fraction: float, min_clients: int):
        num_available = self._count_available(cm)
        num_sample = max(min_clients, int(math.ceil(num_available * fraction)))
        return cm.sample(num_sample, min_num_clients=min_clients)

    # ------ 配置客户端：按桶下发各自全局参数 ------
    def _make_fit_config_for(self, bucket: str) -> Dict[str, Any]:
        dp = self.dp_cfg_map.get(bucket, DpCfg())
        cfg = {
            "model_name": bucket,
            "dp_mode": self.dp_mode,
            "dp_clip": float(dp.clip),
            "dp_sigma": float(dp.sigma),
            "dp_policy_id": self.policy_id,
            "dp_sig": self.dp_sig,
            "share_prefix": "adapter.",
        }
        sig = self.sig_map.get(bucket)
        if sig:
            cfg.update(sig)  # sig_count/sig_hash
        return cfg

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        clients = self._sample(client_manager, self.fraction_fit, self.min_fit_clients)
        instructions: List[Tuple[fl.server.client_proxy.ClientProxy, FitIns]] = []
        for c in clients:
            bucket = getattr(c, "bucket_name", None) or self.assign_fn(c.cid)
            arrs = self.params_map.get(bucket, [])
            fit_cfg = self._make_fit_config_for(bucket)
            ins = FitIns(parameters=ndarrays_to_parameters(arrs), config=fit_cfg)
            instructions.append((c, ins))
        return instructions

    # ------ 聚合训练更新：分桶聚合 ------
    def aggregate_fit(self, server_round: int, results, failures):
        buckets: Dict[str, List[Tuple[float, List[np.ndarray]]]] = {}
        train_stats: Dict[str, Dict[str, float]] = {}
        prev_map = {k: [a.copy() for a in v] for k, v in self.params_map.items()}

        for client, fit_res in results:
            arrs = parameters_to_ndarrays(fit_res.parameters)
            bucket = (fit_res.metrics or {}).get("model_name") or self.assign_fn(client.cid)
            w = float(getattr(fit_res, "num_examples", 1))
            if len(arrs) == 0:
                continue
            buckets.setdefault(bucket, []).append((w, arrs))
            if fit_res.metrics and "train_loss" in fit_res.metrics:
                st = train_stats.setdefault(bucket, {"sum_wloss": 0.0, "sum_w": 0.0})
                st["sum_wloss"] += w * float(fit_res.metrics["train_loss"])
                st["sum_w"] += w

        for bucket, items in buckets.items():
            dp = self.dp_cfg_map.get(bucket, DpCfg())
            cur_prev = prev_map.get(bucket, None)

            # 服务端 DP（可选）
            if self.dp_mode == "server" and (dp.clip > 0 or dp.sigma > 0) and cur_prev and len(cur_prev) > 0:
                dp_items: List[Tuple[float, List[np.ndarray]]] = []
                for w, arrs in items:
                    if len(arrs) != len(cur_prev):
                        dp_items.append((w, arrs))  # 长度不一致直接保留原样，避免形状错
                        continue
                    delta = [a - p for a, p in zip(arrs, cur_prev)]
                    delta, _ = l2_clip(delta, dp.clip)
                    delta = add_gaussian_noise(delta, dp.sigma, dp.clip)
                    arrs_dp = [p + d for p, d in zip(cur_prev, delta)]
                    dp_items.append((w, arrs_dp))
                items = dp_items

            # 聚合
            if self.aggregator == "median":
                agg = _aggregate_median(items)
            elif self.aggregator == "trimmed":
                agg = _aggregate_trimmed_mean(items, self.trim_ratio)
            else:
                agg = _weighted_mean(items)

            self.params_map[bucket] = agg
            self.sig_map[bucket] = _sig_from_arrays(agg)

            # 预算记账 + 服务器审计日志
            epsilon = self.ledger.step(bucket, self.dp_cfg_map.get(bucket, DpCfg()).sigma) if self.ledger else None
            log_jsonl(self._dp_log_path, {
                "ts": time.time(),
                "round": server_round,
                "bucket": bucket,
                "dp_mode": self.dp_mode,
                "policy_id": self.policy_id,
                "dp": {"clip": dp.clip, "sigma": dp.sigma, "delta": dp.delta},
                "epsilon": epsilon,
                "num_results": len(items),
            })

        # 打印各桶 train
        for bucket, st in sorted(train_stats.items()):
            if st["sum_w"] > 0:
                avg = st["sum_wloss"] / st["sum_w"]
                print(f"[round {server_round}] train[{bucket}] = {avg:.6f} (n={int(st['sum_w'])})")
                log_jsonl(os.path.join("bench_results", "bucket_metrics.jsonl"), {
                    "ts": time.time(), "round": server_round, "type": "train",
                    "bucket": bucket, "loss": avg, "n": int(st["sum_w"])
                })

        any_bucket = next(iter(self.params_map), None)
        if any_bucket is None:
            return ndarrays_to_parameters([]), {}
        return ndarrays_to_parameters(self.params_map[any_bucket]), {}

    # ------ 配置评估 ------
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        clients = self._sample(client_manager, self.fraction_evaluate, self.min_available_clients)
        instructions: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateIns]] = []
        for c in clients:
            bucket = getattr(c, "bucket_name", None) or self.assign_fn(c.cid)
            arrs = self.params_map.get(bucket, [])
            cfg = self._make_fit_config_for(bucket)
            ins = EvaluateIns(parameters=ndarrays_to_parameters(arrs), config=cfg)
            instructions.append((c, ins))
        return instructions

    # ------ 聚合评估：分桶打印 + 返回 overall ------
    def aggregate_evaluate(self, server_round: int, results, failures):
        per_bucket: Dict[str, Dict[str, float]] = {}
        total_wloss, total_w = 0.0, 0.0
        for client, eval_res in results:
            n = float(getattr(eval_res, "num_examples", 1))
            loss = float(getattr(eval_res, "loss", 0.0))
            bucket = (eval_res.metrics or {}).get("model_name") or self.assign_fn(client.cid)
            st = per_bucket.setdefault(bucket, {"sum_wloss": 0.0, "sum_w": 0.0})
            st["sum_wloss"] += n * loss
            st["sum_w"] += n
            total_wloss += n * loss
            total_w += n

        for bucket, st in sorted(per_bucket.items()):
            if st["sum_w"] > 0:
                avg = st["sum_wloss"] / st["sum_w"]
                print(f"[round {server_round}]  val[{bucket}] = {avg:.6f} (n={int(st['sum_w'])})")
                log_jsonl(os.path.join("bench_results", "bucket_metrics.jsonl"), {
                    "ts": time.time(), "round": server_round, "type": "val",
                    "bucket": bucket, "loss": avg, "n": int(st["sum_w"])
                })

        overall = (total_wloss / total_w) if total_w > 0 else None
        metrics = {f"val_{k}": (v["sum_wloss"] / v["sum_w"]) for k, v in per_bucket.items() if v["sum_w"] > 0}
        return (overall, metrics)

    def evaluate(self, server_round: int, parameters: Parameters):
        return None
