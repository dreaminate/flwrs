from __future__ import annotations
from typing import List, Tuple, Dict, Any
import hashlib
import numpy as np
import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays, Parameters  # 用于安全转换

# ========== 指纹计算 ==========
def _model_signature_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    # 以“层顺序 + 形状 + dtype”生成稳定哈希
    keys = sorted(state.keys())
    shapes = [tuple(state[k].shape) for k in keys]
    dtypes = [str(state[k].dtype).split(".")[-1] for k in keys]
    payload = f"{len(keys)}|" + "|".join(f"{s}:{d}" for s, d in zip(shapes, dtypes))
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return {"sig_count": len(keys), "sig_hash": h}

# ========== 权重打包/解包 ==========
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
        t = torch.tensor(arr, dtype=state[k].dtype)
        new_state[k] = t
    model.load_state_dict(new_state, strict=True)

# --------- 工具：安全转换 / 防御式加载 ----------
def _to_arrays(parameters_or_arrays) -> List[np.ndarray]:
    if isinstance(parameters_or_arrays, list):
        return parameters_or_arrays
    if isinstance(parameters_or_arrays, Parameters):
        return parameters_to_ndarrays(parameters_or_arrays)
    return parameters_or_arrays  # type: ignore

def _maybe_load_params(model: torch.nn.Module, arrays: List[np.ndarray],
                       expected_sig: Dict[str, Any] | None = None) -> bool:
    keys = sorted(model.state_dict().keys())
    if not isinstance(arrays, list) or len(arrays) != len(keys):
        print(f"[warn] param length mismatch: expect={len(keys)}, got={len(arrays) if isinstance(arrays, list) else '??'}. Skip loading.")
        return False
    # 如服务器发来 expected_sig，可再次校验
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

# --------- 读取模型规格（dummy 生成数据用） ----------
def _get_model_specs(model: torch.nn.Module):
    seq_len = getattr(model, "seq_len", 32)
    in_dim  = getattr(model, "in_dim", 16)
    out_dim = getattr(model, "out_dim", 1)
    return seq_len, in_dim, out_dim

# ========== 你要替换的两处（接入你的训练/评估） ==========
def user_train_one_round(model: torch.nn.Module, model_name: str, cid: str, config: Dict[str, Any]) -> Dict[str, float]:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    T, F, O = _get_model_specs(model)
    x = torch.randn(256, T, F); y = torch.randn(256, O)
    opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
    return {"train_loss": float(loss.item())}

def user_evaluate(model: torch.nn.Module, model_name: str, cid: str, config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    model.eval()
    T, F, O = _get_model_specs(model)
    with torch.no_grad():
        x = torch.randn(128, T, F); y = torch.randn(128, O)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
    return float(loss.item()), {"val_loss": float(loss.item())}
# =======================================================

class HeteroModelClient(fl.client.NumPyClient):
    def __init__(self, model: torch.nn.Module, model_name: str, cid: str):
        self.model = model
        self.model_name = model_name
        self.cid = cid

    # ⭐ 关键：向服务器上报“模型名 + 指纹”
    def get_properties(self, config):
        st = self.model.state_dict()
        sig = _model_signature_from_state(st)
        return {
            "model_name": self.model_name,
            "sig_count": sig["sig_count"],
            "sig_hash": sig["sig_hash"],
        }

    def get_parameters(self, config):
        return pack_state(self.model)

    def fit(self, parameters, config):
        arrays = _to_arrays(parameters)
        expected_sig = {k: config[k] for k in ["sig_count", "sig_hash"] if k in config}
        _maybe_load_params(self.model, arrays, expected_sig=expected_sig)

        train_metrics = user_train_one_round(self.model, self.model_name, self.cid, config)
        new_params = pack_state(self.model)
        metrics = {"model_name": self.model_name, **_model_signature_from_state(self.model.state_dict())}
        metrics.update(train_metrics)
        num_examples = int(config.get("num_examples", 1000))
        return new_params, num_examples, metrics

    def evaluate(self, parameters, config):
        arrays = _to_arrays(parameters)
        expected_sig = {k: config[k] for k in ["sig_count", "sig_hash"] if k in config}
        _maybe_load_params(self.model, arrays, expected_sig=expected_sig)

        loss, metrics = user_evaluate(self.model, self.model_name, self.cid, config)
        metrics.update(_model_signature_from_state(self.model.state_dict()))
        return loss, int(config.get("num_val_examples", 200)), metrics
