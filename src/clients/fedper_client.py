# src/clients/fedper_client.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import hashlib, time, os
import numpy as np
import torch
import flwr as fl
from flwr.common import Parameters, parameters_to_ndarrays
from privacy.dp import sanitize_update
from privacy.audit import log_jsonl

DEFAULT_SHARE_SUBSTR = "adapter."   # 只共享包含该片段的参数

def _match_keys(state: Dict[str, torch.Tensor], share_substr: str) -> List[str]:
    keys = sorted(state.keys())
    return [k for k in keys if share_substr in k]  # 只要包含即可（不要求开头）

def _sig_from_state(state: Dict[str, torch.Tensor], share_substr: str) -> Dict[str, Any]:
    keys = _match_keys(state, share_substr)
    shapes = [tuple(state[k].shape) for k in keys]
    dtypes = [str(state[k].dtype).split(".")[-1] for k in keys]
    payload = f"{len(keys)}|" + "|".join(f"{s}:{d}" for s, d in zip(shapes, dtypes))
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return {"sig_count": len(keys), "sig_hash": h}

def pack_shared(model: torch.nn.Module, share_substr: str) -> List[np.ndarray]:
    state = model.state_dict()
    keys = _match_keys(state, share_substr)
    return [state[k].detach().cpu().numpy() for k in keys]

def unpack_shared(model: torch.nn.Module, arrays: List[np.ndarray], share_substr: str) -> None:
    state = model.state_dict()
    keys = _match_keys(state, share_substr)
    assert len(keys) == len(arrays), f"shared shape mismatch: {len(keys)} vs {len(arrays)}"
    for k, arr in zip(keys, arrays):
        state[k] = torch.tensor(arr, dtype=state[k].dtype)
    model.load_state_dict(state, strict=False)

def _to_arrays(parameters_or_arrays) -> List[np.ndarray]:
    if isinstance(parameters_or_arrays, list):
        return parameters_or_arrays
    if isinstance(parameters_or_arrays, Parameters):
        return parameters_to_ndarrays(parameters_or_arrays)
    return parameters_or_arrays  # type: ignore

def _get_model_specs(model: torch.nn.Module):
    seq_len = getattr(model, "seq_len", 32)
    in_dim  = getattr(model, "in_dim", 16)
    out_dim = getattr(model, "out_dim", 1)
    return seq_len, in_dim, out_dim

def user_train_one_round(model: torch.nn.Module, model_name: str, cid: str, config: Dict[str, Any]) -> Dict[str, float]:
    model.train()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config.get("lr", 1e-3)))
    T, F, O = _get_model_specs(model)
    x = torch.randn(256, T, F); y = torch.randn(256, O)
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    opt.zero_grad(); loss.backward(); opt.step()
    return {"train_loss": float(loss.item())}

def user_evaluate(model: torch.nn.Module, model_name: str, cid: str, config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    model.eval()
    T, F, O = _get_model_specs(model)
    with torch.no_grad():
        x = torch.randn(128, T, F); y = torch.randn(128, O)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
    return float(loss.item()), {"val_loss": float(loss.item())}

class FedPerClient(fl.client.NumPyClient):
    """
    个性化联邦（Adapter/LoRA 版）：
    - 只共享 state_dict 中键名包含 share_substr 的参数（默认 'adapter.'）
    - 非 adapter 参数建议冻结（在构建模型时处理）
    """
    def __init__(self, model: torch.nn.Module, model_name: str, cid: str, share_substr: str = DEFAULT_SHARE_SUBSTR):
        self.model = model
        self.model_name = model_name
        self.cid = cid
        self.share_substr = share_substr

    def get_properties(self, config):
        sig = _sig_from_state(self.model.state_dict(), self.share_substr)
        return {"model_name": self.model_name, "sig_count": sig["sig_count"], "sig_hash": sig["sig_hash"]}

    def get_parameters(self, config):
        return pack_shared(self.model, self.share_substr)

    def fit(self, parameters, config):
        arrays_in = _to_arrays(parameters)
        expected_sig = {k: config[k] for k in ["sig_count", "sig_hash"] if k in config}

        # 尝试加载（不匹配就跳过）
        try:
            st = self.model.state_dict()
            local_sig = _sig_from_state(st, self.share_substr)
            if (expected_sig.get("sig_count") == local_sig.get("sig_count")
                and expected_sig.get("sig_hash") == local_sig.get("sig_hash")
                and isinstance(arrays_in, list)
                and len(arrays_in) == local_sig.get("sig_count", -1)):
                unpack_shared(self.model, arrays_in, self.share_substr)
            else:
                print(f"[warn] adapter signature mismatch or len mismatch, skip loading. expected={expected_sig}, local={local_sig}, got_len={len(arrays_in) if isinstance(arrays_in, list) else '??'}")
        except Exception as e:
            print(f"[warn] unpack shared failed: {e}. Skip.")

        # 训练（仅 adapter 参与梯度）
        train_metrics = user_train_one_round(self.model, self.model_name, self.cid, config)
        new_params = pack_shared(self.model, self.share_substr)

        # delta 范数（审计）
        prev = arrays_in if isinstance(arrays_in, list) else []
        delta_norm = None
        if prev and len(prev) == len(new_params):
            delta = [n - p for n, p in zip(new_params, prev)]
            delta_norm = float(np.sqrt(sum(float((d.astype(np.float64) ** 2).sum()) for d in delta)))

        # 本地 DP
        dp_mode  = str(config.get("dp_mode", "client"))
        dp_clip  = float(config.get("dp_clip", 0.0))
        dp_sigma = float(config.get("dp_sigma", 0.0))
        dp_sig   = str(config.get("dp_sig", ""))
        policy_id = str(config.get("dp_policy_id", ""))
        applied = False
        est_post_clip = min(delta_norm, dp_clip) if (delta_norm is not None and dp_clip > 0) else delta_norm
        if dp_mode == "client" and (dp_clip > 0 or dp_sigma > 0) and prev and len(prev) == len(new_params):
            new_params = sanitize_update(prev, new_params, dp_clip, dp_sigma)
            applied = True

        # 客户端审计日志
        log_jsonl(os.path.join("bench_results", "clients", f"cid_{self.cid}_privacy.jsonl"), {
            "ts": time.time(),
            "cid": self.cid,
            "model_name": self.model_name,
            "policy_id": policy_id,
            "dp_sig": dp_sig,
            "dp_mode": dp_mode,
            "dp": {"clip": dp_clip, "sigma": dp_sigma},
            "delta_norm": delta_norm,
            "est_post_clip_norm": est_post_clip,
            "applied": applied,
            "train_metrics": train_metrics,
            "share_substr": self.share_substr,
        })

        sig = _sig_from_state(self.model.state_dict(), self.share_substr)
        metrics = {"model_name": self.model_name, **sig, **train_metrics}
        num_examples = int(config.get("num_examples", 1000))
        return new_params, num_examples, metrics

    def evaluate(self, parameters, config):
        arrays = _to_arrays(parameters)
        try:
            st = self.model.state_dict()
            local_sig = _sig_from_state(st, self.share_substr)
            expected_sig = {k: config[k] for k in ["sig_count", "sig_hash"] if k in config}
            if (expected_sig.get("sig_count") == local_sig.get("sig_count")
                and expected_sig.get("sig_hash") == local_sig.get("sig_hash")
                and isinstance(arrays, list)
                and len(arrays) == local_sig.get("sig_count", -1)):
                unpack_shared(self.model, arrays, self.share_substr)
        except Exception as e:
            print(f"[warn] eval unpack shared failed: {e}. Skip.")
        loss, metrics = user_evaluate(self.model, self.model_name, self.cid, config)
        metrics.update(_sig_from_state(self.model.state_dict(), self.share_substr))
        return loss, int(config.get("num_val_examples", 200)), metrics
                                             