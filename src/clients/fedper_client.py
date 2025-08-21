# src/clients/fedper_client.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import hashlib, time, os, math
import numpy as np
import torch
import flwr as fl
from flwr.common import Parameters, parameters_to_ndarrays

from privacy.dp import sanitize_update  # 期望实现：基于 prev/new, 对 (new-prev) 做 L2 裁剪 + 高斯噪声
from privacy.audit import log_jsonl

DEFAULT_SHARE_SUBSTR = "adapter."   # 只共享包含该片段的参数


# --------------------------- 小工具 ---------------------------

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _gaussian_sigma_from_epsilon(
    epsilon: float, delta: float, sensitivity: float = 1.0
) -> float:
    """
    单次高斯机制近似标定：sigma = S * sqrt(2 ln(1.25/delta)) / epsilon
    （这里用于“发送前本地 DP”的一次性更新保护）
    """
    epsilon = max(float(epsilon), 1e-12)
    delta = max(float(delta), 1e-12)
    return (float(sensitivity) * math.sqrt(2.0 * math.log(1.25 / delta))) / epsilon


# --------------------------- 共享参数选择/打包 ---------------------------

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
        # 保持 dtype，一般 float32/bfloat16
        state[k] = torch.tensor(arr, dtype=state[k].dtype)
    model.load_state_dict(state, strict=False)

def _to_arrays(parameters_or_arrays) -> List[np.ndarray]:
    if isinstance(parameters_or_arrays, list):
        return parameters_or_arrays
    if isinstance(parameters_or_arrays, Parameters):
        return parameters_to_ndarrays(parameters_or_arrays)
    # 兜底（某些模拟场景可能传 tuple）
    if isinstance(parameters_or_arrays, tuple):
        return list(parameters_or_arrays)  # type: ignore
    return parameters_or_arrays  # type: ignore

def _get_model_specs(model: torch.nn.Module):
    seq_len = getattr(model, "seq_len", 32)
    in_dim  = getattr(model, "in_dim", 16)
    out_dim = getattr(model, "out_dim", 1)
    return seq_len, in_dim, out_dim


# --------------------------- 示例训练/评估（你后续替换为真实数据管道） ---------------------------

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


# --------------------------- 客户端实现 ---------------------------

class FedPerClient(fl.client.NumPyClient):
    """
    个性化联邦（Adapter/LoRA 版）：
    - 只共享 state_dict 中键名包含 share_substr 的参数（默认 'adapter.'）
    - 非 adapter 参数建议冻结（在构建模型时处理）
    - 本地 DP（发送前裁剪+加噪）支持两种配置：
        1) 直接给定 dp_sigma
        2) 指定 dp_epsilon/dp_delta/sensitivity，由客户端换算为 sigma
    """
    def __init__(self, model: torch.nn.Module, model_name: str, cid: str, share_substr: str = DEFAULT_SHARE_SUBSTR):
        self.model = model
        self.model_name = model_name
        self.cid = cid
        self.share_substr = share_substr

    # ------ Flower 必需接口 ------

    def get_properties(self, config):
        sig = _sig_from_state(self.model.state_dict(), self.share_substr)
        return {"model_name": self.model_name, "sig_count": sig["sig_count"], "sig_hash": sig["sig_hash"]}

    def get_parameters(self, config):
        return pack_shared(self.model, self.share_substr)

    def fit(self, parameters, config):
        arrays_in = _to_arrays(parameters)
        expected_sig = {k: config[k] for k in ["sig_count", "sig_hash"] if k in config}

        # 1) 尝试加载来自服务器的共享 adapter 权重（签名不匹配则跳过）
        try:
            st = self.model.state_dict()
            local_sig = _sig_from_state(st, self.share_substr)
            if (
                expected_sig.get("sig_count") == local_sig.get("sig_count")
                and expected_sig.get("sig_hash") == local_sig.get("sig_hash")
                and isinstance(arrays_in, list)
                and len(arrays_in) == local_sig.get("sig_count", -1)
            ):
                unpack_shared(self.model, arrays_in, self.share_substr)
            else:
                print(
                    f"[warn] adapter signature mismatch or len mismatch, skip loading. "
                    f"expected={expected_sig}, local={local_sig}, got_len={len(arrays_in) if isinstance(arrays_in, list) else '??'}"
                )
        except Exception as e:
            print(f"[warn] unpack shared failed: {e}. Skip.")

        # 2) 本地训练（仅 adapter 参与梯度）
        train_metrics = user_train_one_round(self.model, self.model_name, self.cid, config)
        new_params = pack_shared(self.model, self.share_substr)

        # 3) 计算 delta 范数（用于审计/可视化）
        prev = arrays_in if isinstance(arrays_in, list) else []
        delta_norm = None
        if prev and len(prev) == len(new_params):
            delta = [n - p for n, p in zip(new_params, prev)]
            delta_norm = float(np.sqrt(sum(float((d.astype(np.float64) ** 2).sum()) for d in delta)))

        # 4) 本地 DP（发送前裁剪+加噪）
        #    支持两种配置：优先使用 dp_sigma；若 <=0 则尝试用 epsilon/delta/sensitivity 换算。
        dp_mode  = str(config.get("dp_mode", "client"))        # "client"/"none"/"server"
        dp_clip  = float(config.get("dp_clip", 0.0))            # L2 裁剪半径（更新向量）
        dp_sigma = float(config.get("dp_sigma", 0.0))           # 高斯噪声标准差（直接模式）
        dp_eps   = float(config.get("dp_epsilon", 0.0))         # 若给定 epsilon，则换算 sigma
        dp_delta = float(config.get("dp_delta", 0.0))
        dp_sens  = float(config.get("dp_sensitivity", 1.0))
        dp_sig   = str(config.get("dp_sig", ""))                # 自定义策略签名/版本
        policy_id = str(config.get("dp_policy_id", ""))

        # 若未直接给 sigma，但给了 (epsilon, delta)，则换算一次
        if dp_sigma <= 0.0 and (dp_eps > 0.0 and dp_delta > 0.0):
            try:
                dp_sigma = _gaussian_sigma_from_epsilon(dp_eps, dp_delta, dp_sens)
            except Exception:
                # 保守兜底：不抛异常、不应用 DP
                dp_sigma = 0.0

        applied = False
        est_post_clip = min(delta_norm, dp_clip) if (delta_norm is not None and dp_clip > 0) else delta_norm

        if (
            dp_mode.lower() == "client"
            and (dp_clip > 0.0 or dp_sigma > 0.0)
            and prev
            and len(prev) == len(new_params)
        ):
            try:
                # 这里期望 sanitize_update(prev, new, clip, sigma) 内部执行：
                #   delta = new - prev
                #   delta <- clip_L2(delta, clip)
                #   delta <- delta + N(0, sigma^2 I)
                #   return prev + delta
                new_params = sanitize_update(prev, new_params, dp_clip, dp_sigma)
                applied = True
            except Exception as e:
                print(f"[warn] sanitize_update failed: {e}. Send raw params (no DP).")
                applied = False

        # 5) 客户端隐私审计日志
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
                "share_substr": self.share_substr,
            },
        )

        # 6) 返回
        sig = _sig_from_state(self.model.state_dict(), self.share_substr)
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
        try:
            st = self.model.state_dict()
            local_sig = _sig_from_state(st, self.share_substr)
            expected_sig = {k: config[k] for k in ["sig_count", "sig_hash"] if k in config}
            if (
                expected_sig.get("sig_count") == local_sig.get("sig_count")
                and expected_sig.get("sig_hash") == local_sig.get("sig_hash")
                and isinstance(arrays, list)
                and len(arrays) == local_sig.get("sig_count", -1)
            ):
                unpack_shared(self.model, arrays, self.share_substr)
        except Exception as e:
            print(f"[warn] eval unpack shared failed: {e}. Skip.")

        loss, m = user_evaluate(self.model, self.model_name, self.cid, config)
        m.update(_sig_from_state(self.model.state_dict(), self.share_substr))
        return loss, int(config.get("num_val_examples", 200)), m
