from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class DPCfg:
    clip: float = 1.0     # L2 裁剪阈值（对“更新向量”的 L2）
    sigma: float = 0.0    # 高斯噪声倍数（噪声标准差 = sigma * clip）
    mode: str = "client"  # "client" | "server"

def l2_clip(arrs: List[np.ndarray], max_norm: float) -> Tuple[List[np.ndarray], float]:
    """把一组张量当作一个大向量做 L2 裁剪。"""
    if max_norm <= 0:
        return arrs, 0.0
    total = 0.0
    for a in arrs:
        total += np.sum(a.astype(np.float64) ** 2)
    norm = float(np.sqrt(total) + 1e-12)
    scale = min(1.0, max_norm / norm)
    if scale == 1.0:
        return arrs, norm
    return [a * scale for a in arrs], norm

def add_gaussian_noise(arrs: List[np.ndarray], sigma: float, clip: float) -> List[np.ndarray]:
    if sigma <= 0:
        return arrs
    std = sigma * clip
    rng = np.random.default_rng()
    return [a + rng.normal(0.0, std, size=a.shape).astype(a.dtype) for a in arrs]

def sanitize_update(prev: List[np.ndarray], new: List[np.ndarray], clip: float, sigma: float) -> List[np.ndarray]:
    """对“参数更新”执行 裁剪+加噪： new = prev + noise(clip(new-prev))."""
    delta = [n - p for n, p in zip(new, prev)]
    delta, _ = l2_clip(delta, clip)
    delta = add_gaussian_noise(delta, sigma, clip)
    out = [p + d for p, d in zip(prev, delta)]
    return out
