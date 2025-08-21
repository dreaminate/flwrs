# src/clients/fedper_client.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import hashlib
import numpy as np
import torch
import flwr as fl
from flwr.common import Parameters, parameters_to_ndarrays  # 安全转换

# 只联邦含此前缀的层
BACKBONE_PREFIX = "backbone."

# ---------- 工具：取 backbone 相关键 ----------
def _backbone_keys(model: torch.nn.Module) -> List[str]:
    return [k for k in sorted(model.state_dict().keys()) if k.startswith(BACKBONE_PREFIX)]

# ---------- 指纹：基于 backbone 的层顺序/shape/dtype ----------
def _backbone_signature_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    keys = [k for k in sorted(state.keys()) if k.startswith(BACKBONE_PREFIX)]
    shapes = [tuple(state[k].shape) for k in keys]
    dtypes = [str(state[k].dtype).split(".")[-1] for k in keys]
    payload = f"{len(keys)}|" + "|".join(f"{s}:{d}" for s, d in zip(shapes, dtypes))
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return {"sig_count": len(keys), "sig_hash": h}

# ---------- 打包/解包（仅 backbone） ----------
def pack_backbone(model: torch.nn.Module) -> List[np.ndarray]:
    state = model.state_dict()
    keys = _backbone_keys(model)
    return [state[k].detach().cpu().numpy() for k in keys]

def unpack_backbone(model: torch.nn.Module, arrays: List[np.ndarray]) -> None:
    state = model.state_dict()
    keys = _backbone_keys(model)
    assert len(keys) == len(arrays), f"backbone shape mismatch: {len(keys)} vs {len(arrays)}"
    for k, arr in zip(keys, arrays):
        state[k] = torch.tensor(arr, dtype=state[k].dtype)
    model.load_state_dict(state, strict=False)

# ---------- 安全转换 / 防御式加载（仅 backbone） ----------
def _to_arrays(parameters_or_arrays) -> List[np.ndarray]:
    if isinstance(parameters_or_arrays, list):
        return parameters_or_arrays
    if isinstance(parameters_or_arrays, Parameters):
        return parameters_to_ndarrays(parameters_or_arrays)
    return parameters_or_arrays  # type: ignore

def _maybe_load_backbone(
    model: torch.nn.Module,
    arrays: List[np.ndarray],
    expected_sig: Dict[str, Any] | None = None,
) -> bool:
    keys = _backbone_keys(model)
    if not isinstance(arrays, list) or len(arrays) != len(keys):
        print(f"[warn] backbone param length mismatch: expect={len(keys)}, got={len(arrays) if isinstance(arrays, list) else '??'}. Skip.")
        return False
    if expected_sig is not None:
        local_sig = _backbone_signature_from_state(model.state_dict())
        if (local_sig.get("sig_count") != expected_sig.get("sig_count")
            or local_sig.get("sig_hash") != expected_sig.get("sig_hash")):
            print(f"[warn] backbone signature mismatch: local={local_sig}, expected={expected_sig}. Skip.")
            return False
    try:
        unpack_backbone(model, arrays)
        return True
    except AssertionError as e:
        print(f"[warn] unpack backbone failed: {e}. Skip.")
        return False

# ---------- 仅用于示例训练/评估生成 dummy 时序 ----------
def _get_model_specs(model: torch.nn.Module):
    seq_len = getattr(model, "seq_len", 32)
    in_dim  = getattr(model, "in_dim", 16)
    out_dim = getattr(model, "out_dim", 1)
    return seq_len, in_dim, out_dim

# ========= 你接入真实数据/训练的两处 =========
def user_train_one_round(model: torch.nn.Module, model_name: str, cid: str, config: Dict[str, Any]) -> Dict[str, float]:
    """
    TODO: 用你的 DataLoader/Lightning 替换这里。保持返回指标字典即可。
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    T, F, O = _get_model_specs(model)
    x = torch.randn(256, T, F); y = torch.randn(256, O)
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    opt.zero_grad(); loss.backward(); opt.step()
    return {"train_loss": float(loss.item())}

def user_evaluate(model: torch.nn.Module, model_name: str, cid: str, config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    TODO: 用你的验证逻辑替换。返回 (loss, metrics)。
    """
    model.eval()
    T, F, O = _get_model_specs(model)
    with torch.no_grad():
        x = torch.randn(128, T, F); y = torch.randn(128, O)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
    return float(loss.item()), {"val_loss": float(loss.item())}
# ============================================

class FedPerClient(fl.client.NumPyClient):
    """
    个性化联邦：只联邦 backbone；head 本地保留。
    - 自动上报 backbone“指纹”（sig_count/sig_hash）
    - 训练/评估阶段都做防御式加载：不匹配就跳过，绝不崩
    """
    def __init__(self, model: torch.nn.Module, model_name: str, cid: str):
        self.model = model
        self.model_name = model_name
        self.cid = cid

    # ⭐ 关键：上报模型名 + backbone 签名，供服务器按指纹自动匹配
    def get_properties(self, config):
        sig = _backbone_signature_from_state(self.model.state_dict())
        return {"model_name": self.model_name, "sig_count": sig["sig_count"], "sig_hash": sig["sig_hash"]}

    # 只共享 backbone
    def get_parameters(self, config):
        return pack_backbone(self.model)

    def fit(self, parameters, config):
        arrays = _to_arrays(parameters)
        expected_sig = {k: config[k] for k in ["sig_count", "sig_hash"] if k in config}
        _maybe_load_backbone(self.model, arrays, expected_sig=expected_sig)

        train_metrics = user_train_one_round(self.model, self.model_name, self.cid, config)
        new_params = pack_backbone(self.model)

        sig = _backbone_signature_from_state(self.model.state_dict())
        metrics = {"model_name": self.model_name, **sig, **train_metrics}
        num_examples = int(config.get("num_examples", 1000))
        return new_params, num_examples, metrics

    def evaluate(self, parameters, config):
        arrays = _to_arrays(parameters)
        expected_sig = {k: config[k] for k in ["sig_count", "sig_hash"] if k in config}
        _maybe_load_backbone(self.model, arrays, expected_sig=expected_sig)

        loss, metrics = user_evaluate(self.model, self.model_name, self.cid, config)
        metrics.update(_backbone_signature_from_state(self.model.state_dict()))
        return loss, int(config.get("num_val_examples", 200)), metrics
