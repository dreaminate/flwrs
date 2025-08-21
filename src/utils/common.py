# src/utils/common.py
from __future__ import annotations
from typing import Dict, List
import os, json
import numpy as np
import torch

# ====== 整模打包/解包（给 MultiModel 等策略用） ======
def pack_state(model: torch.nn.Module) -> List[np.ndarray]:
    st = model.state_dict()
    keys = sorted(st.keys())
    return [st[k].detach().cpu().numpy() for k in keys]

def unpack_state(model: torch.nn.Module, arrays: List[np.ndarray]) -> None:
    st = model.state_dict()
    keys = sorted(st.keys())
    assert len(keys) == len(arrays), f"shape mismatch: {len(keys)} vs {len(arrays)}"
    new_st = {}
    for k, arr in zip(keys, arrays):
        new_st[k] = torch.tensor(arr, dtype=st[k].dtype)
    model.load_state_dict(new_st, strict=True)

def save_params_map(params_map: Dict[str, List[np.ndarray]], model_builders: Dict[str, callable], out_dir: str = "ckpt") -> None:
    """保存整模权重（要求 arrays 数量与 state_dict 完全一致）。"""
    os.makedirs(out_dir, exist_ok=True)
    for name, arrays in params_map.items():
        model = model_builders[name]()
        unpack_state(model, arrays)  # 若不是整模，这里会 assert
        # 存成 PyTorch 权重
        path = os.path.join(out_dir, f"{name}.pt")
        torch.save(model.state_dict(), path)
        # 也存一份 .npz 方便对比
        np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **{f"arr_{i}": a for i, a in enumerate(arrays)})

# ====== 共享子集保存（给 FedPerCohort/FedPer 的 adapter 子集用） ======
def save_shared_params_map(params_map: Dict[str, List[np.ndarray]], out_dir: str = "ckpt_shared", meta: Dict[str, dict] | None = None) -> None:
    """保存“子集参数列表”，不尝试还原到整模。
    仅把数组按顺序写入 .npz，并附带 shapes/dtypes 以及可选 meta。
    """
    os.makedirs(out_dir, exist_ok=True)
    meta = meta or {}
    for bucket, arrays in params_map.items():
        if not arrays:
            continue
        fn = os.path.join(out_dir, f"{bucket}_shared.npz")
        payload = {f"arr_{i}": a for i, a in enumerate(arrays)}
        payload["shapes"] = np.array([a.shape for a in arrays], dtype=object)
        payload["dtypes"] = np.array([str(a.dtype) for a in arrays], dtype=object)
        np.savez_compressed(fn, **payload)
        # 额外写一份 json 元信息（可选）
        info = {
            "bucket": bucket,
            "num_arrays": len(arrays),
            "shapes": [list(a.shape) for a in arrays],
            "dtypes": [str(a.dtype) for a in arrays],
        }
        info.update(meta.get(bucket, {}))
        with open(os.path.join(out_dir, f"{bucket}_shared.meta.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
