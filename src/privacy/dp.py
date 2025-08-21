# src/privacy/dp.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import math

__all__ = [
    "DPCfg",
    "l2_norm",
    "l2_clip",
    "add_gaussian_noise_abs",
    "sanitize_update",
    "gaussian_sigma_mult_from_epsilon",
    "clip_and_add_noise_to_ndarrays",
]

# ===================== 配置结构 =====================

@dataclass
class DPCfg:
    clip: float = 1.0           # L2 裁剪半径（对“更新向量”的 L2）
    sigma_mult: float = 0.0     # 噪声强度倍数：std = sigma_mult * clip
    mode: str = "client"        # "client" | "server"


# ===================== 基础工具 =====================

def l2_norm(arrs: List[np.ndarray]) -> float:
    """把一组张量当作一个大向量的 L2 范数。"""
    if not arrs:
        return 0.0
    flat = np.concatenate([a.ravel().astype(np.float64, copy=False) for a in arrs], axis=0)
    return float(np.linalg.norm(flat, ord=2))

def l2_clip(arrs: List[np.ndarray], max_norm: float) -> Tuple[List[np.ndarray], float]:
    """对一组张量整体做 L2 裁剪；返回(裁剪后张量, 原始范数)。"""
    if max_norm <= 0.0 or not arrs:
        return arrs, l2_norm(arrs)
    norm = l2_norm(arrs)
    if norm <= max_norm or norm == 0.0:
        return arrs, norm
    scale = max_norm / (norm + 1e-12)
    clipped = [(a * scale).astype(a.dtype, copy=False) for a in arrs]
    return clipped, norm

def add_gaussian_noise(
    arrs: List[np.ndarray],
    std: float,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """对每个数组加独立高斯噪声（绝对标准差 std）。"""
    if std <= 0.0 or not arrs:
        return arrs
    rng = rng or np.random.default_rng()
    return [a + rng.normal(loc=0.0, scale=std, size=a.shape).astype(a.dtype) for a in arrs]


# ===================== 发送前“裁剪+加噪”核心 =====================

def sanitize_update(
    prev: List[np.ndarray],
    new: List[np.ndarray],
    clip: float,
    sigma_mult: float,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    对“参数更新”执行 裁剪 + 加噪：
      delta = new - prev
      delta <- clip_L2(delta, clip)
      delta <- delta + N(0, (sigma_mult * clip)^2 I)
      return prev + delta
    """
    assert len(prev) == len(new), f"length mismatch: {len(prev)} vs {len(new)}"
    if not prev:
        return new
    # 计算更新
    delta = [n - p for n, p in zip(new, prev)]
    # L2 裁剪
    delta, _ = l2_clip(delta, clip)
    # 高斯噪声（倍数语义）
    std = max(float(sigma_mult), 0.0) * max(float(clip), 0.0)
    delta = add_gaussian_noise_abs(delta, std=std, rng=rng)
    # 还原参数
    return [p + d for p, d in zip(prev, delta)]


# ===================== 从 (ε, δ) 计算噪声“倍数” =====================

def gaussian_sigma_mult_from_epsilon(
    epsilon: float,
    delta: float,
    clipping_norm: float,
    sensitivity: Optional[float] = None,
) -> float:
    """
    高斯机制近似标定，把 (ε, δ) -> sigma_mult：
      std_abs = S * sqrt(2 ln(1.25/δ)) / ε
      sigma_mult = std_abs / clipping_norm
    通常选择 S = clipping_norm（即更新向量的敏感度就是裁剪半径），
    此时 sigma_mult = sqrt(2 ln(1.25/δ)) / ε
    """
    epsilon = max(float(epsilon), 1e-12)
    delta = max(float(delta), 1e-12)
    clipping_norm = max(float(clipping_norm), 1e-12)
    S = float(sensitivity) if sensitivity is not None else clipping_norm
    std_abs = (S * math.sqrt(2.0 * math.log(1.25 / delta))) / epsilon
    return float(std_abs / clipping_norm)


# ===================== 直接对“参数数组”裁剪+加噪（便捷接口） =====================

def clip_and_add_noise_to_ndarrays(
    arrs: List[np.ndarray],
    clipping_norm: float = 1.0,
    *,
    # 方式一：直接给“倍数”
    sigma_mult: Optional[float] = None,
    # 方式二：给 (ε, δ[, sensitivity])，内部换算为“倍数”
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    sensitivity: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    先 L2 裁剪到 clipping_norm，再加 N(0, (sigma_mult * clipping_norm)^2) 噪声。
    - 若给定 sigma_mult，优先使用；
    - 否则需给 (epsilon, delta [, sensitivity]) 以换算。
    """
    if not arrs:
        return arrs
    clipped, _ = l2_clip(arrs, clipping_norm)
    if sigma_mult is None:
        if epsilon is None or delta is None:
            return clipped  # 没给噪声参数，就只裁剪
        sigma_mult = gaussian_sigma_mult_from_epsilon(
            epsilon=epsilon, delta=delta, clipping_norm=clipping_norm, sensitivity=sensitivity
        )
    std = max(float(sigma_mult), 0.0) * max(float(clipping_norm), 0.0)
    return add_gaussian_noise_abs(clipped, std=std, rng=rng)
