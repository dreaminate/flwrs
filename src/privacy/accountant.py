from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import math

def zcdp_to_epsilon(z: float, delta: float) -> float:
    """zCDP (ρ) → (ε, δ) 的保守换算。"""
    if z <= 0:
        return 0.0
    delta = max(delta, 1e-12)
    return float(z + 2.0 * math.sqrt(z * math.log(1.0 / delta)))

@dataclass
class ZCDPAccountant:
    delta: float = 1e-5
    z: float = 0.0  # ρ
    def step_gaussian(self, sigma: float) -> float:
        # 高斯机制（敏感度=1）一次查询的 ρ = 1/(2σ^2)
        if sigma > 0:
            self.z += 1.0 / (2.0 * (sigma ** 2))
        return self.current_epsilon()
    def current_epsilon(self) -> float:
        return zcdp_to_epsilon(self.z, self.delta)

class CohortPrivacyLedger:
    """按模型桶跟踪 DP 预算。"""
    def __init__(self, delta_per_bucket: Dict[str, float]):
        self.acc: Dict[str, ZCDPAccountant] = {
            name: ZCDPAccountant(delta=delta) for name, delta in delta_per_bucket.items()
        }
    def step(self, bucket: str, sigma: float) -> float | None:
        if bucket in self.acc:
            return self.acc[bucket].step_gaussian(sigma)
        return None
    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "epsilon": acc.current_epsilon(),
                "delta": acc.delta,
                "z": acc.z,
            }
            for name, acc in self.acc.items()
        }
