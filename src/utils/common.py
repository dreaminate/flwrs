from __future__ import annotations
from typing import List, Dict, Iterable
import os
import numpy as np
import torch

def pack_state(model: torch.nn.Module, include: Iterable[str] | None = None, exclude_prefixes: Iterable[str] | None = None) -> List[np.ndarray]:
    state = model.state_dict()
    keys = sorted(state.keys())
    if include is not None:
        include = set(include)
        keys = [k for k in keys if k in include]
    if exclude_prefixes is not None:
        prefixes = tuple(exclude_prefixes)
        keys = [k for k in keys if not k.startswith(prefixes)]
    return [state[k].detach().cpu().numpy() for k in keys]

def unpack_state(model: torch.nn.Module, arrays: List[np.ndarray], template_model: torch.nn.Module | None = None) -> None:
    state = model.state_dict()
    keys = sorted(state.keys())
    assert len(keys) == len(arrays), f"shape mismatch: {len(keys)} vs {len(arrays)}"
    new_state = {}
    for k, arr in zip(keys, arrays):
        t = torch.tensor(arr, dtype=state[k].dtype)
        new_state[k] = t
    model.load_state_dict(new_state, strict=True)

def save_params_map(params_map: Dict[str, List[np.ndarray]], builders: Dict[str, callable], out_dir: str = "ckpt"):
    os.makedirs(out_dir, exist_ok=True)
    for name, arrays in params_map.items():
        model = builders[name]()
        unpack_state(model, arrays)
        path = os.path.join(out_dir, f"global_{name}.pth")
        torch.save(model.state_dict(), path)
        print(f"[ckpt] saved: {path}")
