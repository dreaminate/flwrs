# src/clients/hetero_client.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import hashlib, time, os, math
import numpy as np
import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays, Parameters

from privacy.dp import sanitize_update           # 本地 DP：对 (new-prev) 裁剪+加噪，再还原
from privacy.audit import log_jsonl              # 客户端审计日志


# ===================== 小工具 =====================

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _gaussian_sigma_from_epsilon(epsilon: float, delta: float, sensitivity: float = 1.0) -> float:
    """
    高斯机制近似标定：
        sigma = S * sqrt(2 * ln(1.25/delta)) / epsilon
    用于“发送前一次性更新”的本地 DP。
    """
    epsilon = max(float(epsilon), 1e-12)
    delta = max(float(delta), 1e-12)
    return (float(sensitivity) * math.sqrt(2.0 * math.log(1.25 / delta))) / epsilon


# ===================== 指纹 =====================

def _model_signature_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    keys = sorted(state.keys())
    shapes = [tuple(state[k].shape) for k in keys]
    dtypes = [str(state[k].dtype).split(".")[-1] for k in keys]
    payload = f"{len(keys)}|" + "|".join(f"{s}:{d}" for s, d in zip(shapes, dtypes))
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return {"sig_count": len(keys), "sig_hash": h}


# ===================== 打包 / 解包 =====================

def pack_state(model: torch.nn.Module) -> List[np.ndarray]:
    state = model.state_dict()
    keys = sorted(state.keys())
    return [state[k].detach().cpu().numpy() for k in keys]

def unpack_state(model: torch.nn.Module, arrays: List[np.ndarray]) -> None:
    state = model.state_dict()
    keys = sorted(state.keys())
    assert len(keys) == len(arrays), f"shape mismatch: {len(keys)} vs {len(arrays)}"
    new_state = {}
    for k, arr in zip(keys, arrays):
        new_state[k] = torch.tensor(arr, dtype=state[k].dtype)
    model.load_state_dict(new_state, strict=True)


# --------- 安全转换 / 防御式加载 ----------

def _to_arrays(parameters_or_arrays) -> List[np.ndarray]:
    if isinstance(parameters_or_arrays, list):
        return parameters_or_arrays
    if isinstance(parameters_or_arrays, Parameters):
        return parameters_to_ndarrays(parameters_or_arrays)
    if isinstance(parameters_or_arrays, tuple):
        return list(parameters_or_arrays)  # 兜底
    return parameters_or_arrays  # type: ignore

def _maybe_load_params(model: torch.nn.Module, arrays: List[np.ndarray], expected_sig: Dict[str, Any] | None = None) -> bool:
    keys = sorted(model.state_dict().keys())
    if not isinstance(arrays, list) or len(arrays) != len(keys):
        print(f"[warn] param length mismatch: expect={len(keys)}, got={len(arrays) if isinstance(arrays, list) else '??'}. Skip loading.")
        return False
    if expected_sig is not None:
        st = model.state_dict()
        local_sig = _model_signature_from_state(st)
        if (local_sig.get("sig_count") != expected_sig.get("sig_count")
            or local_sig.get("sig_hash") != expected_sig.get("sig_hash")):
            print(f"[warn] signature mismatch: local={local_sig}, expected={expected_sig}. Skip loading.")
            return False
    try:
        unpack_state(model, arrays)
        return True
    except AssertionError as e:
        print(f"[warn] unpack failed: {e}. Skip loading.")
        return False


# ---------- dummy 数据规格 ----------

def _get_model_specs(model: torch.nn.Module):
    seq_len = getattr(model, "seq_len", 32)
    in_dim  = getattr(model, "in_dim", 16)
    out_dim = getattr(model, "out_dim", 1)
    return seq_len, in_dim, out_dim


# ---------- 你接入真实数据的位置 ----------

def user_train_one_round(model: torch.nn.Module, model_name: str, cid: str, config: Dict[str, Any]) -> Dict[str, float]:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-3)))
    loss_fn = torch.nn.MSELoss()
    T, F, O = _get_model_specs(model)
    x = torch.randn(256, T, F); y = torch.randn(256, O)
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    return {"train_loss": float(loss.item())}

def user_evaluate(model: torch.nn.Module, model_name: str, cid: str, config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    model.eval()
    T, F, O = _get_model_specs(model)
    with torch.no_grad():
        x = torch.randn(128, T, F); y = torch.randn(128, O)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
    return float(loss.item()), {"val_loss": float(loss.item())}


# ===================== 客户端实现 =====================

class HeteroModelClient(fl.client.NumPyClient):
    """
    横向/异构模型客户端（共享完整 state_dict）：
    - 发送前本地 DP（客户端级）：支持 dp_sigma 或 (dp_epsilon, dp_delta, sensitivity) 自动换算。
    - 完整的签名与长度校验；日志自动建目录。
    """
    def __init__(self, model: torch.nn.Module, model_name: str, cid: str):
        self.model = model
        self.model_name = model_name
        self.cid = cid

    def get_properties(self, config):
        return {"model_name": self.model_name, **_model_signature_from_state(self.model.state_dict())}

    def get_parameters(self, config):
        return pack_state(self.model)

    def fit(self, parameters, config):
        arrays_in = _to_arrays(parameters)
        expected_sig = {k: config[k] for k in ["sig_count", "sig_hash"] if k in config}
        _maybe_load_params(self.model, arrays_in, expected_sig=expected_sig)

        # 1) 本地训练
        train_metrics = user_train_one_round(self.model, self.model_name, self.cid, config)
        new_params = pack_state(self.model)

        # 2) 计算 delta 范数（仅在长度匹配时）
        prev: List[np.ndarray] = []
        if isinstance(arrays_in, list) and len(arrays_in) == len(new_params):
            prev = arrays_in
        delta_norm = None
        if prev:
            delta = [n - p for n, p in zip(new_params, prev)]
            delta_norm = float(np.sqrt(sum(float((d.astype(np.float64) ** 2).sum()) for d in delta)))

        # 3) 本地 DP（发送前裁剪+加噪）
        dp_mode  = str(config.get("dp_mode", "client"))         # "client" / "none" / "server"
        dp_clip  = float(config.get("dp_clip", 0.0))             # L2 裁剪半径（更新向量）
        dp_sigma = float(config.get("dp_sigma", 0.0))            # 若 >0 优先用它
        dp_eps   = float(config.get("dp_epsilon", 0.0))          # 可选：给 epsilon/delta/sensitivity → sigma
        dp_delta = float(config.get("dp_delta", 0.0))
        dp_sens  = float(config.get("dp_sensitivity", 1.0))
        dp_sig   = str(config.get("dp_sig", ""))
        policy_id = str(config.get("dp_policy_id", ""))

        if dp_sigma <= 0.0 and (dp_eps > 0.0 and dp_delta > 0.0):
            try:
                dp_sigma = _gaussian_sigma_from_epsilon(dp_eps, dp_delta, dp_sens)
            except Exception:
                dp_sigma = 0.0  # 保守兜底：失败则不应用噪声

        applied = False
        est_post_clip = min(delta_norm, dp_clip) if (delta_norm is not None and dp_clip > 0) else delta_norm

        if (
            dp_mode.lower() == "client"
            and (dp_clip > 0.0 or dp_sigma > 0.0)
            and prev  # 只有能计算 delta 时才能做 sanitize_update
        ):
            try:
                new_params = sanitize_update(prev, new_params, dp_clip, dp_sigma)
                applied = True
            except Exception as e:
                print(f"[warn] sanitize_update failed: {e}. Send raw params (no DP).")
                applied = False

        # 4) 客户端审计日志
        log_path = os.path.join("bench_results", "clients", f"cid_{self.cid}_privacy.jsonl")
        _ensure_dir(log_path)
        log_jsonl(
            log_path,
            {
                "ts": time.time(),
                "cid": self.cid,
                "model_name": self.model_name,
                "policy_id": policy_id,
                "dp_sig": dp_sig,
                "dp_mode": dp_mode,
                "dp": {
                    "clip": dp_clip,
                    "sigma": dp_sigma,
                    "epsilon": dp_eps,
                    "delta": dp_delta,
                    "sensitivity": dp_sens,
                },
                "delta_norm": delta_norm,
                "est_post_clip_norm": est_post_clip,
                "applied": applied,
                "train_metrics": train_metrics,
            },
        )

        # 5) 返回
        sig = _model_signature_from_state(self.model.state_dict())
        metrics = {
            "model_name": self.model_name,
            **sig,
            **train_metrics,
            "dp_applied": float(applied),
            "dp_clip": float(dp_clip),
            "dp_sigma": float(dp_sigma),
        }
        num_examples = int(config.get("num_examples", 1000))
        return new_params, num_examples, metrics

    def evaluate(self, parameters, config):
        arrays = _to_arrays(parameters)
        expected_sig = {k: config[k] for k in ["sig_count", "sig_hash"] if k in config}
        _maybe_load_params(self.model, arrays, expected_sig=expected_sig)

        loss, metrics = user_evaluate(self.model, self.model_name, self.cid, config)
        metrics.update(_model_signature_from_state(self.model.state_dict()))
        return loss, int(config.get("num_val_examples", 200)), metrics
