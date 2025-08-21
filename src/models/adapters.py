# src/models/adapters.py
from __future__ import annotations
from typing import Iterable, Tuple, Optional, Callable
import re
import torch
import torch.nn as nn

# --------- LoRA 封装：Linear / Conv1d ----------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        in_f, out_f = base.in_features, base.out_features
        # 冻结原权重
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        # 适配器：x -> (x @ A) -> (.. @ B)
        self.adapter = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_f, rank, bias=False),
            nn.Linear(rank, out_f, bias=False),
        )
        # 初始化：A~N(0,0.01), B=0
        nn.init.normal_(self.adapter[1].weight, std=0.01)
        nn.init.zeros_(self.adapter[2].weight)
        self.scaling = float(alpha) / float(rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.adapter(x)

class LoRAConv1d(nn.Module):
    """在 Conv1d 输出通道上做 1x1 的低秩适配（channel 方向），不依赖 kernel 大小。"""
    def __init__(self, base: nn.Conv1d, rank: int = 4, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Conv1d)
        in_c, out_c = base.in_channels, base.out_channels
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.adapter = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(in_c, rank, kernel_size=1, bias=False),
            nn.Conv1d(rank, out_c, kernel_size=1, bias=False),
        )
        nn.init.normal_(self.adapter[1].weight, std=0.01)
        nn.init.zeros_(self.adapter[2].weight)
        self.scaling = float(alpha) / float(rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.adapter(x)

# --------- MLP Adapter（通用残差瓶颈），可用于 Linear/Conv1d 输出 ----------
class MLPAdapter(nn.Module):
    """对最后维度/通道维做瓶颈 MLP：y + up(act(down(y)))"""
    def __init__(self, dim: int, bottleneck: int = 16, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, dim, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        z = self.drop(self.down(y))
        z = self.up(self.act(z))
        return y + z

class MLPAdapterWrap(nn.Module):
    """把 Linear/Conv1d 的输出接一个 MLPAdapter（以 Linear 输出为例）"""
    def __init__(self, base: nn.Module, out_dim: int, bottleneck: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.adapter = MLPAdapter(out_dim, bottleneck=bottleneck, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return self.adapter(y)

# --------- 主注入函数 ----------
def inject_adapters(
    model: nn.Module,
    mode: str = "lora",                # "lora" | "mlp"
    rank: int = 4,
    alpha: int = 16,
    dropout: float = 0.0,
    target_types: Tuple[type, ...] = (nn.Linear, nn.Conv1d),
    exclude_name_regex: str = r"(head|classifier|proj_out)",
) -> nn.Module:
    """
    遍历模型，把目标层替换为带 adapter 的包装层。
    - 所有 adapter 参数的 state_dict 路径都会包含 ".adapter." 片段，便于 FedPer 只共享它们。
    """
    pattern = re.compile(exclude_name_regex) if exclude_name_regex else None

    def should_exclude(name_path: str) -> bool:
        return bool(pattern and pattern.search(name_path))

    def replace(parent: nn.Module, name: str, child: nn.Module, name_path: str):
        if should_exclude(name_path):
            return
        if isinstance(child, nn.Linear) and nn.Linear in target_types:
            wrapped = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout) if mode == "lora" \
                      else MLPAdapterWrap(child, out_dim=child.out_features, bottleneck=rank, dropout=dropout)
            # 把适配器挂在名为 adapter 的子模块下，确保键名含 ".adapter."
            wrapper = nn.Sequential()
            wrapper.add_module("base", wrapped.base if hasattr(wrapped, "base") else child)
            wrapper.add_module("adapter", wrapped.adapter if hasattr(wrapped, "adapter") else nn.Identity())
            # 用自定义 forward
            class _Wrap(nn.Sequential):
                def forward(self, x):
                    y = self._modules["base"](x)
                    a = self._modules["adapter"]
                    return y + a(x) if isinstance(a, (nn.Sequential, nn.Module)) else y
            new_mod = _Wrap()
            new_mod.add_module("base", wrapper.base)
            new_mod.add_module("adapter", wrapper.adapter)
            setattr(parent, name, new_mod)
        elif isinstance(child, nn.Conv1d) and nn.Conv1d in target_types:
            if mode == "lora":
                wrapped = LoRAConv1d(child, rank=rank, alpha=alpha, dropout=dropout)
                new_mod = nn.Sequential()
                new_mod.add_module("base", wrapped.base)
                new_mod.add_module("adapter", wrapped.adapter)
                class _Wrap1d(nn.Sequential):
                    def forward(self, x):
                        return self._modules["base"](x) + self._modules["adapter"](x) * (wrapped.scaling if hasattr(wrapped, "scaling") else 1.0)
                setattr(parent, name, _Wrap1d())
                getattr(parent, name).add_module("base", wrapped.base)
                getattr(parent, name).add_module("adapter", wrapped.adapter)
            else:
                # MLPAdapter 对 Conv1d：把通道维 flatten 到最后一维再还原（近似，简单稳妥）
                out_c = child.out_channels
                wrapper = nn.Sequential()
                wrapper.add_module("base", child)
                wrapper.add_module("adapter", nn.Sequential(
                    nn.Flatten(start_dim=1, end_dim=1),  # (B,C,L) -> (B*C, L) 不改变 L；仅作为占位
                ))
                class _WrapC1d(nn.Module):
                    def __init__(self, base: nn.Conv1d, out_c: int, drop: float):
                        super().__init__()
                        self.base = base
                        for p in self.base.parameters():
                            p.requires_grad = False
                        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
                        self.down = nn.Conv1d(out_c, rank, kernel_size=1, bias=False)
                        self.up = nn.Conv1d(rank, out_c, kernel_size=1, bias=False)
                        nn.init.normal_(self.down.weight, std=0.01); nn.init.zeros_(self.up.weight)
                    def forward(self, x):
                        y = self.base(x)
                        return y + self.up(self.drop(self.down(y)))
                setattr(parent, name, _WrapC1d(wrapper.base, out_c, dropout))

    def walk(module: nn.Module, prefix: str = ""):
        for child_name, child in list(module.named_children()):
            name_path = f"{prefix}.{child_name}" if prefix else child_name
            replace(module, child_name, child, name_path)
            # 如果被替换为 nn.Sequential/_Wrap，内部不再递归；否则继续深入
            if hasattr(module, child_name):
                new_child = getattr(module, child_name)
                if new_child is child:
                    walk(child, name_path)

    walk(model)
    return model

# 冻结非 adapter 参数
def freeze_non_adapter(model: nn.Module, share_substr: str = "adapter.") -> None:
    for name, p in model.named_parameters():
        if share_substr in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

def count_params(model: nn.Module, share_substr: str = "adapter.") -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    shared = sum(p.numel() for n, p in model.named_parameters() if share_substr in n)
    return shared, total
