from __future__ import annotations
from typing import Dict, List
import argparse
import os
import flwr as fl
import numpy as np
import torch

from strategies.multimodel_cohort import MultiModelCohort
from strategies.robust_dp_fedprox import RobustDPFedProx
from strategies.fedopt import FedOptLike

from models.registry import MODEL_BUILDERS
from clients.hetero_client import HeteroModelClient, pack_state, unpack_state
# ❌ 不在顶部导入 FedPerClient，避免未使用策略时报错；按需在 client_fn 内部导入

# ✅ Adapter/LoRA 注入工具（功能3）
from models.adapters import inject_adapters, freeze_non_adapter, count_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", type=str, default="multimodel",
                   choices=["multimodel", "robust", "fedopt", "fedper"])
    # 默认三种时间序列模型
    p.add_argument("--models", type=str, default="lstm,tcn,ts_transformer")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--clients", type=int, default=30)
    p.add_argument("--aggregator", type=str, default="trimmed",
                   choices=["weighted", "median", "trimmed"])
    p.add_argument("--trim_ratio", type=float, default=0.1)
    p.add_argument("--mu_prox", type=float, default=0.0)
    p.add_argument("--dp_max_norm", type=float, default=1.0)
    p.add_argument("--dp_sigma", type=float, default=0.0)
    p.add_argument("--fedopt_variant", type=str, default="adam", choices=["adam", "yogi"])
    p.add_argument("--fedopt_lr", type=float, default=0.01)
    p.add_argument("--min_fit", type=int, default=2)

    # ✅ 功能2：细粒度 DP 下发/记账
    p.add_argument("--dp_mode", type=str, default="client", choices=["client","server"])
    p.add_argument("--dp_clip_map", type=str, default="")   # 例: "lstm:1.0,tcn:0.5,ts_transformer:1.5"
    p.add_argument("--dp_sigma_map", type=str, default="")  # 例: "lstm:0.8,tcn:1.2,ts_transformer:0.6"
    p.add_argument("--dp_delta_map", type=str, default="")  # 例: "lstm:1e-5,tcn:1e-6"
    p.add_argument("--dp_policy_id", type=str, default="dpv1")  # 写入审计日志与配置签名

    # ✅ 功能3：LoRA/Adapter 注入与只共享前缀
    p.add_argument("--adapter_mode", type=str, default="none", choices=["none", "lora", "mlp"],
                   help="是否注入 Adapter/LoRA")
    p.add_argument("--adapter_rank", type=int, default=4, help="LoRA rank 或 MLP bottleneck")
    p.add_argument("--adapter_alpha", type=int, default=16, help="LoRA scaling alpha")
    p.add_argument("--adapter_dropout", type=float, default=0.0, help="Adapter/LoRA dropout")
    p.add_argument("--share_prefix", type=str, default="adapter.", help="要共享的参数名片段（FedPer 客户端会只共享含此片段的权重）")
    p.add_argument("--freeze_non_adapter", action="store_true", help="冻结所有非 share_prefix 参数")

    return p.parse_args()


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


# ✅ 功能3：在构建客户端模型时按需注入 Adapter/LoRA，并可冻结非 adapter 参数
def make_client_fn(strategy: str, model_names: list[str], args):
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
        if strategy == "fedper":
            # 仅在使用 FedPer 时导入，避免未用模块报错
            from clients.fedper_client import FedPerClient
            return FedPerClient(model=model, model_name=mname, cid=cid, share_substr=args.share_prefix).to_client()
        else:
            return HeteroModelClient(model=model, model_name=mname, cid=cid).to_client()

    return client_fn


def main():
    args = parse_args()
    model_names = [s.strip() for s in args.models.split(",") if s.strip()]

    # 解析 DP 映射
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

    clip_map  = _parse_map(args.dp_clip_map, 1.0, model_names)
    sigma_map = _parse_map(args.dp_sigma_map, 0.0, model_names)
    delta_map = _parse_map(args.dp_delta_map, 1e-5, model_names) if args.dp_delta_map else {n: 1e-5 for n in model_names}

    dp_cfg_map = {n: {"clip": clip_map[n], "sigma": sigma_map[n], "delta": delta_map[n]} for n in model_names}

    # 选择策略
    if args.strategy == "multimodel":
        init_map = build_init_params_map(model_names)
        if args.clients <= 2:
            frac_fit, frac_eval = 1.0, 1.0
        else:
            frac_fit, frac_eval = 0.33, 0.25
        strategy = MultiModelCohort(
            init_params_map=init_map,
            assign_fn=assign_fn_factory(model_names),
            aggregator=args.aggregator,
            trim_ratio=args.trim_ratio,
            fraction_fit=frac_fit, fraction_evaluate=frac_eval,
            dp_cfg_map=dp_cfg_map,            # 每个模型桶的 clip/sigma/delta
            dp_mode=args.dp_mode,             # "client" 或 "server"
            policy_id=args.dp_policy_id,      # ✅ 功能2：审计/配置签名的策略ID
            min_fit_clients=args.min_fit, min_available_clients=args.min_fit,
            accept_failures=True,
        )
    elif args.strategy == "robust":
        strategy = RobustDPFedProx(
            dp_max_norm=args.dp_max_norm, dp_noise_sigma=args.dp_sigma,
            robust=args.aggregator, trim_ratio=args.trim_ratio,
            mu_prox=args.mu_prox,
            fraction_fit=0.5, fraction_evaluate=0.3,
            min_fit_clients=args.min_fit, min_available_clients=args.min_fit,
            accept_failures=True,
        )
    elif args.strategy == "fedopt":
        strategy = FedOptLike(
            lr=args.fedopt_lr, variant=args.fedopt_variant,
            fraction_fit=0.5, fraction_evaluate=0.3,
            min_fit_clients=args.min_fit, min_available_clients=args.min_fit,
            accept_failures=True,
        )
    elif args.strategy == "fedper":
        # FedPer 侧仅客户端控制“只共享 share_prefix”；服务端可用通用稳健策略
        strategy = RobustDPFedProx(
            robust=args.aggregator, trim_ratio=args.trim_ratio,
            fraction_fit=0.5, fraction_evaluate=0.3,
            min_fit_clients=args.min_fit, min_available_clients=args.min_fit,
            accept_failures=True,
        )
    else:
        raise ValueError("unknown strategy")

    # ✅ 使用带 args 的 client_fn（此前少传了 args 会报错）
    client_fn = make_client_fn(args.strategy, model_names, args)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    if hasattr(strategy, "params_map"):
        from utils.common import save_params_map
        save_params_map(strategy.params_map, MODEL_BUILDERS, out_dir="ckpt")

    print("[done] experiment finished.")


if __name__ == "__main__":
    main()
