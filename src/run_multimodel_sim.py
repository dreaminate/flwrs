# src/run_multimodel_sim.py
from __future__ import annotations
from typing import Dict, List, Tuple
import argparse
import numpy as np
import flwr as fl

from strategies.fedper_cohort import FedPerCohort, DpCfg
from strategies.multimodel_cohort import MultiModelCohort
from strategies.robust_dp_fedprox import RobustDPFedProx
from strategies.fedopt import FedOptLike

from models.registry import MODEL_BUILDERS
from clients.hetero_client import HeteroModelClient, pack_state
from models.adapters import inject_adapters, freeze_non_adapter, count_params


# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--strategy",
        type=str,
        default="multimodel",
        choices=["multimodel", "robust", "fedopt", "fedper", "fedper_cohort"],
    )
    # 默认三种时间序列模型
    p.add_argument("--models", type=str, default="lstm,tcn,ts_transformer")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--clients", type=int, default=30)
    p.add_argument(
        "--aggregator",
        type=str,
        default="trimmed",
        choices=["weighted", "median", "trimmed"],
    )
    p.add_argument("--trim_ratio", type=float, default=0.1)
    p.add_argument("--mu_prox", type=float, default=0.0)

    # ====== DP / 参与度 ======
    p.add_argument("--min_fit", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_examples", type=int, default=1000)
    p.add_argument("--num_val_examples", type=int, default=200)

    # 旧 robust 策略参数（中央 DP）
    p.add_argument("--dp_max_norm", type=float, default=1.0)
    p.add_argument("--dp_sigma", type=float, default=0.0)

    # FedOpt
    p.add_argument("--fedopt_variant", type=str, default="adam", choices=["adam", "yogi"])
    p.add_argument("--fedopt_lr", type=float, default=0.01)

    # ✅ 功能2：细粒度“客户端本地 DP”下发/记账
    p.add_argument("--dp_mode", type=str, default="client", choices=["client", "server", "none"])
    p.add_argument("--dp_clip_map", type=str, default="")   # 例: "lstm:1.0,tcn:0.5,ts_transformer:1.5"
    p.add_argument("--dp_sigma_map", type=str, default="")  # 例: "lstm:0.8,tcn:1.2,ts_transformer:0.6"  （倍数语义）
    p.add_argument("--dp_delta_map", type=str, default="")  # 例: "lstm:1e-5,tcn:1e-6"
    p.add_argument("--dp_policy_id", type=str, default="dpv1")  # 写入审计日志与配置签名

    # ✅ 功能3：LoRA/Adapter 注入与只共享前缀
    p.add_argument("--adapter_mode", type=str, default="none", choices=["none", "lora", "mlp"],
                   help="是否注入 Adapter/LoRA")
    p.add_argument("--adapter_rank", type=int, default=4, help="LoRA rank 或 MLP bottleneck")
    p.add_argument("--adapter_alpha", type=int, default=16, help="LoRA scaling alpha")
    p.add_argument("--adapter_dropout", type=float, default=0.0, help="Adapter/LoRA dropout")
    p.add_argument("--share_prefix", type=str, default="adapter.", help="要共享的参数名片段（FedPer 客户端只共享含此片段的权重）")
    p.add_argument("--freeze_non_adapter", action="store_true", help="冻结所有非 share_prefix 参数")
    return p.parse_args()


# ------------------------- 工具 -------------------------

def assign_fn_factory(model_names: List[str]):
    def _assign(cid: str) -> str:
        idx = int(cid) % len(model_names)
        return model_names[idx]
    return _assign


def build_init_params_map(model_names: List[str]) -> Dict[str, List[np.ndarray]]:
    # 用基础模型权重初始化（MultiModel 策略会按桶分发）
    params_map: Dict[str, List[np.ndarray]] = {}
    for name in model_names:
        model = MODEL_BUILDERS[name]()
        arrays = pack_state(model)
        params_map[name] = arrays
    return params_map


def _parse_map(spec: str, default: float, names: List[str]) -> Dict[str, float]:
    out = {n: default for n in names}
    if not spec:
        return out
    for kv in spec.split(","):
        if ":" in kv:
            k, v = kv.split(":", 1)
            k = k.strip(); v = v.strip()
            if k in out:
                out[k] = float(v)
    return out


def _choose_global_dp(clip_map: Dict[str, float], sigma_map: Dict[str, float], delta_map: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Flower 的 on_fit_config_fn（标准接口）无法按客户端/桶精准下发时，给一个“统一”的客户端本地 DP。
    这里用 clip 的中位数、sigma 的中位数、delta 的中位数作为全局默认。
    （如果你的自定义策略支持 per-client 配置，这里可以换成基于 client_id 的映射。）
    """
    clips = np.array(list(clip_map.values()), dtype=np.float64)
    sigmas = np.array(list(sigma_map.values()), dtype=np.float64)
    deltas = np.array(list(delta_map.values()), dtype=np.float64)
    return float(np.median(clips)), float(np.median(sigmas)), float(np.median(deltas))


# ✅ 功能3：在构建客户端模型时按需注入 Adapter/LoRA，并可冻结非 adapter 参数
def make_client_fn(strategy: str, model_names: List[str], args):
    assign_fn = assign_fn_factory(model_names)

    def _build_and_inject(mname: str):
        model = MODEL_BUILDERS[mname]()  # 你的模型构建器
        if args.adapter_mode != "none":
            inject_adapters(
                model,
                mode=args.adapter_mode,
                rank=args.adapter_rank,
                alpha=args.adapter_alpha,
                dropout=args.adapter_dropout,
            )
            if args.freeze_non_adapter:
                freeze_non_adapter(model, share_substr=args.share_prefix)
            shared, total = count_params(model, share_substr=args.share_prefix)
            print(f"[adapter] {mname}: shared={shared} / total={total} params ({shared/max(1,total):.2%})")
        return model

    # Flower 新签名（推荐）
    from flwr.common import Context
    def client_fn(context: Context):
        cid = str(context.node_id)
        mname = assign_fn(cid)
        model = _build_and_inject(mname)
        if strategy in ("fedper", "fedper_cohort"):
            # 仅在使用 FedPer/FedPerCohort 时导入
            from clients.fedper_client import FedPerClient
            return FedPerClient(model=model, model_name=mname, cid=cid, share_substr=args.share_prefix).to_client()
        else:
            return HeteroModelClient(model=model, model_name=mname, cid=cid).to_client()

    return client_fn


# ------------------------- 入口 -------------------------

def main():
    args = parse_args()
    model_names = [s.strip() for s in args.models.split(",") if s.strip()]

    # 解析 DP 映射（倍数语义：sigma 是 std/clip）
    clip_map  = _parse_map(args.dp_clip_map, 1.0, model_names)
    sigma_map = _parse_map(args.dp_sigma_map, 0.0, model_names)
    delta_map = _parse_map(args.dp_delta_map, 1e-5, model_names) if args.dp_delta_map else {n: 1e-5 for n in model_names}
    dp_cfg_map = {n: {"clip": clip_map[n], "sigma": sigma_map[n], "delta": delta_map[n]} for n in model_names}
    g_clip, g_sigma, g_delta = _choose_global_dp(clip_map, sigma_map, delta_map)

    # 选择参与比例
    if args.clients <= 2:
        frac_fit, frac_eval = 1.0, 1.0
    else:
        frac_fit, frac_eval = 0.33, 0.25

    # 统一的客户端配置下发（若策略不支持逐客户端配置，用这个兜底）
    def on_fit_config_fn(rnd: int):
        cfg = {
            "lr": args.lr,
            "num_examples": args.num_examples,
            "num_val_examples": args.num_val_examples,
        }
        if args.dp_mode.lower() == "client":
            cfg.update({
                "dp_mode": "client",
                "dp_clip": g_clip,
                "dp_sigma": g_sigma,           # 倍数语义；客户端内部会用 std = sigma * clip
                "dp_delta": g_delta,
                "dp_sig": f"localdp@clip{g_clip:.3g}_sigma{g_sigma:.3g}_delta{g_delta:.1e}",
                "dp_policy_id": args.dp_policy_id,
            })
        elif args.dp_mode.lower() == "server":
            cfg["dp_mode"] = "server"
        else:
            cfg["dp_mode"] = "none"
        return cfg

    # 只聚合指标，避免 per-client 明细外泄
    def fit_metrics_aggregation_fn(results):
        total = sum(num for _, num, _ in results) or 1
        avg_loss = sum(num * m.get("train_loss", 0.0) for _, num, m in results) / total
        dp_rate = sum(num * m.get("dp_applied", 0.0) for _, num, m in results) / total
        return {"loss": float(avg_loss), "dp_applied_rate": float(dp_rate)}

    def evaluate_metrics_aggregation_fn(results):
        total = sum(num for _, num, _ in results) or 1
        avg_vloss = sum(num * m.get("val_loss", 0.0) for _, num, m in results) / total
        return {"val_loss": float(avg_vloss)}

    # 构造策略
    if args.strategy == "multimodel":
        init_map = build_init_params_map(model_names)
        strategy = MultiModelCohort(
            init_params_map=init_map,
            assign_fn=assign_fn_factory(model_names),
            aggregator=args.aggregator,
            trim_ratio=args.trim_ratio,
            fraction_fit=frac_fit, fraction_evaluate=frac_eval,
            dp_cfg_map=dp_cfg_map,            # 每个模型桶的 clip/sigma/delta（供服务器侧 DP/记账用）
            dp_mode=args.dp_mode,             # "client" 或 "server"（服务器策略内部也会用）
            policy_id=args.dp_policy_id,      # ✅ 功能2：审计/配置签名的策略ID
            min_fit_clients=args.min_fit, min_available_clients=args.min_fit,
            accept_failures=True,
            on_fit_config_fn=on_fit_config_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

    elif args.strategy == "robust":
        strategy = RobustDPFedProx(
            dp_max_norm=args.dp_max_norm, dp_noise_sigma=args.dp_sigma,
            robust=args.aggregator, trim_ratio=args.trim_ratio,
            mu_prox=args.mu_prox,
            fraction_fit=0.5, fraction_evaluate=0.3,
            min_fit_clients=args.min_fit, min_available_clients=args.min_fit,
            accept_failures=True,
            on_fit_config_fn=on_fit_config_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

    elif args.strategy == "fedopt":
        strategy = FedOptLike(
            lr=args.fedopt_lr, variant=args.fedopt_variant,
            fraction_fit=0.5, fraction_evaluate=0.3,
            min_fit_clients=args.min_fit, min_available_clients=args.min_fit,
            accept_failures=True,
            on_fit_config_fn=on_fit_config_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

    elif args.strategy == "fedper":
        # 传统 FedPer：单桶单全局，要求同构，否则会 shape mismatch
        strategy = RobustDPFedProx(
            robust=args.aggregator, trim_ratio=args.trim_ratio,
            mu_prox=args.mu_prox,
            fraction_fit=0.5, fraction_evaluate=0.3,
            min_fit_clients=args.min_fit, min_available_clients=args.min_fit,
            accept_failures=True,
            on_fit_config_fn=on_fit_config_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

    elif args.strategy == "fedper_cohort":
        # ✅ FedPer 的“分桶聚合”版：各模型/签名各自聚合共享子集
        dp_cfg_map_dc = {k: DpCfg(clip=v["clip"], sigma=v["sigma"], delta=v["delta"]) for k, v in dp_cfg_map.items()}
        strategy = FedPerCohort(
            assign_fn=assign_fn_factory(model_names),
            aggregator=args.aggregator,
            trim_ratio=args.trim_ratio,
            fraction_fit=(1.0 if args.clients <= 2 else 0.5),
            fraction_evaluate=(1.0 if args.clients <= 2 else 0.5),
            min_fit_clients=args.min_fit, min_available_clients=args.min_fit,
            accept_failures=True,
            dp_cfg_map=dp_cfg_map_dc,
            dp_mode=args.dp_mode,                # "client" 或 "server"
            policy_id=args.dp_policy_id,
            share_prefix=args.share_prefix,
        )

    else:
        raise ValueError("unknown strategy")

    # 若自定义策略构造器未显式接收回调，这里做一次兜底绑定（不报错就行）
    for k, v in {
        "on_fit_config_fn": on_fit_config_fn,
        "fit_metrics_aggregation_fn": fit_metrics_aggregation_fn,
        "evaluate_metrics_aggregation_fn": evaluate_metrics_aggregation_fn,
    }.items():
        try:
            if getattr(strategy, k, None) is None:
                setattr(strategy, k, v)
        except Exception:
            pass

    client_fn = make_client_fn(args.strategy, model_names, args)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    # ====== 保存权重（按策略类型） ======
    if hasattr(strategy, "params_map"):
        from utils.common import save_params_map, save_shared_params_map
        if args.strategy == "multimodel":
            save_params_map(strategy.params_map, MODEL_BUILDERS, out_dir="ckpt")
        elif args.strategy in ("fedper", "fedper_cohort"):
            # 只保存共享子集（adapter）
            meta = {}
            for k, v in strategy.params_map.items():
                meta[k] = {
                    "strategy": args.strategy,
                    "share_prefix": getattr(strategy, "share_prefix", "adapter."),
                    "dp_mode": getattr(strategy, "dp_mode", ""),
                    "policy_id": getattr(strategy, "policy_id", ""),
                }
            save_shared_params_map(strategy.params_map, out_dir="ckpt_shared", meta=meta)

    print("[done] experiment finished.")


if __name__ == "__main__":
    main()
