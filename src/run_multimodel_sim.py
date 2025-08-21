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
from clients.fedper_client import FedPerClient  # 可选

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
    return p.parse_args()

def assign_fn_factory(model_names: List[str]):
    def _assign(cid: str) -> str:
        idx = int(cid) % len(model_names)
        return model_names[idx]
    return _assign

def build_init_params_map(model_names: List[str]) -> Dict[str, List[np.ndarray]]:
    params_map: Dict[str, List[np.ndarray]] = {}
    for name in model_names:
        model = MODEL_BUILDERS[name]()
        arrays = pack_state(model)
        params_map[name] = arrays
    return params_map

def make_client_fn(strategy: str, model_names: List[str]):
    assign_fn = assign_fn_factory(model_names)
    if strategy == "fedper":
        def client_fn(cid: str):
            mname = assign_fn(cid)
            model = MODEL_BUILDERS[mname]()
            return FedPerClient(model=model, model_name=mname, cid=cid)
    else:
        def client_fn(cid: str):
            mname = assign_fn(cid)
            model = MODEL_BUILDERS[mname]()
            return HeteroModelClient(model=model, model_name=mname, cid=cid)
    return client_fn

def main():
    args = parse_args()
    model_names = [s.strip() for s in args.models.split(",") if s.strip()]

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
        strategy = RobustDPFedProx(
            robust=args.aggregator, trim_ratio=args.trim_ratio,
            fraction_fit=0.5, fraction_evaluate=0.3,
            min_fit_clients=args.min_fit, min_available_clients=args.min_fit,
            accept_failures=True,
        )
    else:
        raise ValueError("unknown strategy")

    fl.simulation.start_simulation(
        client_fn=make_client_fn(args.strategy, model_names),
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
