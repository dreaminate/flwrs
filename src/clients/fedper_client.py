from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import flwr as fl

BACKBONE_PREFIX = "backbone."   # 只联邦含此前缀的权重

def pack_backbone(model: torch.nn.Module) -> List[np.ndarray]:
    state = model.state_dict()
    keys = [k for k in sorted(state.keys()) if k.startswith(BACKBONE_PREFIX)]
    return [state[k].detach().cpu().numpy() for k in keys]

def unpack_backbone(model: torch.nn.Module, arrays: List[np.ndarray]) -> None:
    state = model.state_dict()
    keys = [k for k in sorted(state.keys()) if k.startswith(BACKBONE_PREFIX)]
    assert len(keys) == len(arrays), f"backbone shape mismatch: {len(keys)} vs {len(arrays)}"
    for k, arr in zip(keys, arrays):
        state[k] = torch.tensor(arr, dtype=state[k].dtype)
    model.load_state_dict(state, strict=False)

def _get_model_specs(model: torch.nn.Module):
    seq_len = getattr(model, "seq_len", 32)
    in_dim  = getattr(model, "in_dim", 16)
    out_dim = getattr(model, "out_dim", 1)
    return seq_len, in_dim, out_dim

def user_train_one_round(model: torch.nn.Module, model_name: str, cid: str, config: Dict[str, Any]) -> Dict[str, float]:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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
    def __init__(self, model: torch.nn.Module, model_name: str, cid: str):
        self.model = model
        self.model_name = model_name
        self.cid = cid

    def get_parameters(self, config):  # 只共享 backbone
        return pack_backbone(self.model)

    def fit(self, parameters, config):
        unpack_backbone(self.model, parameters)  # 更新 backbone，head 本地保留
        train_metrics = user_train_one_round(self.model, self.model_name, self.cid, config)
        new_params = pack_backbone(self.model)
        metrics = {"model_name": self.model_name}
        num_examples = int(config.get("num_examples", 1000))
        return new_params, num_examples, metrics

    def evaluate(self, parameters, config):
        unpack_backbone(self.model, parameters)
        loss, metrics = user_evaluate(self.model, self.model_name, self.cid, config)
        return loss, int(config.get("num_val_examples", 200)), metrics
