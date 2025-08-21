from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import flwr as fl

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

# --------- 读取模型上的默认输入规格（若无则回退） ----------
def _get_model_specs(model: torch.nn.Module):
    seq_len = getattr(model, "seq_len", 32)
    in_dim  = getattr(model, "in_dim", 16)
    out_dim = getattr(model, "out_dim", 1)
    return seq_len, in_dim, out_dim

# ========== 你要替换的两处（接入你的训练/评估） ==========
def user_train_one_round(model: torch.nn.Module, model_name: str, cid: str, config: Dict[str, Any]) -> Dict[str, float]:
    """
    TODO: 替换为你的本地训练（Lightning 或纯 PyTorch），返回指标字典。
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    T, F, O = _get_model_specs(model)
    x = torch.randn(256, T, F)
    y = torch.randn(256, O)

    opt.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
    return {"train_loss": float(loss.item())}

def user_evaluate(model: torch.nn.Module, model_name: str, cid: str, config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """TODO: 替换为你的验证评估，返回 (loss, metrics)。"""
    model.eval()
    T, F, O = _get_model_specs(model)
    with torch.no_grad():
        x = torch.randn(128, T, F)
        y = torch.randn(128, O)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
    return float(loss.item()), {"val_loss": float(loss.item())}
# =======================================================

class HeteroModelClient(fl.client.NumPyClient):
    def __init__(self, model: torch.nn.Module, model_name: str, cid: str):
        self.model = model
        self.model_name = model_name
        self.cid = cid

    def get_parameters(self, config):
        return pack_state(self.model)

    def fit(self, parameters, config):
        server_model_name = config.get("model_name", self.model_name)
        if server_model_name != self.model_name:
            server_model_name = self.model_name  # 保护：以本端为准避免结构不匹配
        unpack_state(self.model, parameters)

        train_metrics = user_train_one_round(self.model, server_model_name, self.cid, config)
        new_params = pack_state(self.model)

        metrics = {"model_name": server_model_name}
        metrics.update(train_metrics)
        num_examples = int(config.get("num_examples", 1000))
        return new_params, num_examples, metrics

    def evaluate(self, parameters, config):
        unpack_state(self.model, parameters)
        loss, metrics = user_evaluate(self.model, self.model_name, self.cid, config)
        return loss, int(config.get("num_val_examples", 200)), metrics
