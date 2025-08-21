from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Optional, Any
import os, json, time, hashlib
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from privacy.dp import l2_clip, add_gaussian_noise, DPCfg
from privacy.accountant import CohortPrivacyLedger
from privacy.audit import log_jsonl, cfg_hash

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
    多模型同轮 + 指纹自动匹配 + (可选) DP + 审计日志/配置下发
    """

    def __init__(
        self,
        init_params_map: Dict[str, List[np.ndarray]],
        assign_fn: Callable[[str], str],
        aggregator: str = "weighted",
        trim_ratio: float = 0.1,
        dp_cfg_map: Optional[Dict[str, Dict[str, float]]] = None,
        dp_mode: str = "client",
        policy_id: str = "dpv1",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.params_map = init_params_map
        self.assign_fn = assign_fn
        self.aggregator = aggregator
        self.trim_ratio = trim_ratio

        # 签名与 cid→model 缓存
        self.sig_map: Dict[str, Dict[str, Any]] = {name: _sig_from_arrays(arrs) for name, arrs in init_params_map.items()}
        self.cid2model: Dict[str, str] = {}

        # DP 配置
        self.dp_mode = dp_mode  # "client" | "server"
        self.dp_cfg_map: Dict[str, DPCfg] = {}
        delta_map: Dict[str, float] = {}
        for name in init_params_map.keys():
            cfg = dp_cfg_map.get(name, {}) if dp_cfg_map else {}
            clip = float(cfg.get("clip", 1.0))
            sigma = float(cfg.get("sigma", 0.0))
            delta = float(cfg.get("delta", 1e-5))
            self.dp_cfg_map[name] = DPCfg(clip=clip, sigma=sigma, mode=self.dp_mode)
            delta_map[name] = delta
        self.ledger = CohortPrivacyLedger(delta_per_bucket=delta_map)

        # 审计
        os.makedirs("bench_results", exist_ok=True)
        self._dp_log_path = os.path.join("bench_results", "privacy_log.jsonl")
        self.policy_id = policy_id

    # ---------- Flower plumbing ----------
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        any_name = next(iter(self.params_map))
        return ndarrays_to_parameters(self.params_map[any_name])

    def _get_client_props(self, client_proxy) -> Dict[str, Any]:
        try:
            res = client_proxy.get_properties({})
            props = getattr(res, "properties", None) or res
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
        if cid in self.cid2model:
            return self.cid2model[cid]
        props = self._get_client_props(client)
        mname = self._match_by_signature(props) or self.assign_fn(cid)
        self.cid2model[cid] = mname
        return mname

    # ---------- Configure ----------
    def _attach_dp_cfg(self, dst_cfg: Dict[str, Any], bucket: str):
        sig = self.sig_map[bucket]
        dp = self.dp_cfg_map[bucket]
        # 计算配置签名（用于客户端侧核对/审计）
        dp_cfg_payload = {"mode": self.dp_mode, "clip": dp.clip, "sigma": dp.sigma, "bucket": bucket, "policy_id": self.policy_id}
        dst_cfg.update({
            "model_name": bucket,
            "sig_count": sig["sig_count"], "sig_hash": sig["sig_hash"],
            "dp_mode": self.dp_mode, "dp_clip": dp.clip, "dp_sigma": dp.sigma,
            "dp_policy_id": self.policy_id,
            "dp_sig": cfg_hash(dp_cfg_payload),
        })

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        ret = super().configure_fit(server_round, parameters, client_manager)
        ins = ret[1] if isinstance(ret, tuple) else ret
        for client, fitins in ins:
            mname = self._decide_model_for_client(client)
            fitins.parameters = ndarrays_to_parameters(self.params_map[mname])
            self._attach_dp_cfg(fitins.config, mname)
            # 训练默认
            fitins.config.setdefault("batch_size", 64)
            fitins.config.setdefault("local_epochs", 1)
        return ins

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        ret = super().configure_evaluate(server_round, parameters, client_manager)
        ins = ret[1] if isinstance(ret, tuple) else ret
        for client, evalins in ins:
            mname = self._decide_model_for_client(client)
            evalins.parameters = ndarrays_to_parameters(self.params_map[mname])
            self._attach_dp_cfg(evalins.config, mname)
            evalins.config.setdefault("batch_size", 128)
        return ins

    # ---------- Aggregate ----------
    def aggregate_fit(self, server_round: int, results: List[Tuple], failures):
        if not results:
            any_name = next(iter(self.params_map))
            return ndarrays_to_parameters(self.params_map[any_name]), {}

        # 备份上一轮
        prev_map = {name: [a.copy() for a in arrs] for name, arrs in self.params_map.items()}

        # 分桶 & 同时统计训练损失
        buckets: Dict[str, List[Tuple[float, List[np.ndarray]]]] = {}
        train_stats: Dict[str, Dict[str, float]] = {}  # {bucket: {"sum_wloss":..., "sum_w":..., "n":...}}
        for client_proxy, fit_res in results:
            arrs = parameters_to_ndarrays(fit_res.parameters)
            n = float(getattr(fit_res, "num_examples", 1))
            mname = None
            if fit_res.metrics:
                mname = fit_res.metrics.get("model_name")
            if not mname:
                mname = self.cid2model.get(client_proxy.cid, self.assign_fn(client_proxy.cid))
            buckets.setdefault(mname, []).append((n, arrs))

            # 训练损失（可选）
            tl = None
            if fit_res.metrics:
                tl = fit_res.metrics.get("train_loss")
            if tl is not None:
                st = train_stats.setdefault(mname, {"sum_wloss": 0.0, "sum_w": 0.0, "n": 0.0})
                st["sum_wloss"] += n * float(tl)
                st["sum_w"] += n
                st["n"] += n

        # 聚合 + 记账 + 日志
        for mname, items in buckets.items():
            dp = self.dp_cfg_map[mname]

            # 可选：服务端 DP
            if self.dp_mode == "server" and (dp.clip > 0 or dp.sigma > 0):
                dp_items: List[Tuple[float, List[np.ndarray]]] = []
                for w, arrs in items:
                    delta = [a - p for a, p in zip(arrs, prev_map[mname])]
                    delta, _ = l2_clip(delta, dp.clip)
                    delta = add_gaussian_noise(delta, dp.sigma, dp.clip)
                    arrs_dp = [p + d for p, d in zip(prev_map[mname], delta)]
                    dp_items.append((w, arrs_dp))
                items = dp_items

            # 聚合
            if self.aggregator == "median":
                agg = _aggregate_median(items)
            elif self.aggregator == "trimmed":
                agg = _aggregate_trimmed_mean(items, self.trim_ratio)
            else:
                agg = _weighted_mean(items)
            self.params_map[mname] = agg
            self.sig_map[mname] = _sig_from_arrays(agg)

            # 预算记账
            eps = self.ledger.step(mname, dp.sigma)

            # 服务器审计日志
            log_jsonl(self._dp_log_path, {
                "ts": time.time(),
                "round": server_round,
                "bucket": mname,
                "dp_mode": self.dp_mode,
                "policy_id": self.policy_id,
                "dp": {"clip": dp.clip, "sigma": dp.sigma},
                "epsilon": eps,
                "delta": self.ledger.acc[mname].delta,
                "num_results": len(items),
            })

        # ✅ 终端打印各桶训练损失（加权平均）
        for mname, st in sorted(train_stats.items()):
            if st["sum_w"] > 0:
                avg = st["sum_wloss"] / st["sum_w"]
                print(f"[round {server_round}] train[{mname}] = {avg:.6f} (n={int(st['sum_w'])})")
                # 可选：落盘一份
                log_jsonl(os.path.join("bench_results", "bucket_metrics.jsonl"), {
                    "ts": time.time(), "round": server_round, "type": "train",
                    "bucket": mname, "loss": avg, "n": int(st["sum_w"])
                })

        any_name = next(iter(self.params_map))
        return ndarrays_to_parameters(self.params_map[any_name]), {}

    def aggregate_evaluate(self, server_round: int, results: List[Tuple], failures):
        # 没有结果就沿用 Flower 默认行为
        if not results:
            return None

        # 分桶统计验证损失（用 EvaluateRes.loss 加权）
        per_bucket = {}  # {bucket: {"sum_wloss":..., "sum_w":...}}
        total_wloss, total_w = 0.0, 0.0
        for client_proxy, eval_res in results:
            n = float(getattr(eval_res, "num_examples", 1))
            loss = float(getattr(eval_res, "loss", None) or 0.0)
            mname = None
            if eval_res.metrics:
                mname = eval_res.metrics.get("model_name")
            if not mname:
                mname = self.cid2model.get(client_proxy.cid, self.assign_fn(client_proxy.cid))

            st = per_bucket.setdefault(mname, {"sum_wloss": 0.0, "sum_w": 0.0})
            st["sum_wloss"] += n * loss
            st["sum_w"] += n
            total_wloss += n * loss
            total_w += n

        # ✅ 打印各桶验证损失
        for mname, st in sorted(per_bucket.items()):
            if st["sum_w"] > 0:
                avg = st["sum_wloss"] / st["sum_w"]
                print(f"[round {server_round}]  val[{mname}] = {avg:.6f} (n={int(st['sum_w'])})")
                log_jsonl(os.path.join("bench_results", "bucket_metrics.jsonl"), {
                    "ts": time.time(), "round": server_round, "type": "val",
                    "bucket": mname, "loss": avg, "n": int(st["sum_w"])
                })

        # 返回总览给 Flower（它会打印 History (loss, distributed)）
        overall = (total_wloss / total_w) if total_w > 0 else None
        # 也把每桶的值塞进 metrics 里，便于后处理
        metrics = {f"val_{k}": (v["sum_wloss"] / v["sum_w"]) for k, v in per_bucket.items() if v["sum_w"] > 0}
        return (overall, metrics)
