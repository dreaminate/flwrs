from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Positional Encoding (Sine)
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]

# -----------------------------
# 1) LSTM Forecaster
# -----------------------------
class LSTMForecaster(nn.Module):
    """
    输入: (B,T,F) → LSTM → 取最后一个隐藏态 → Linear → (B,out_dim)
    """
    def __init__(
        self,
        in_dim: int = 16,
        seq_len: int = 32,
        hidden_size: int = 128,
        num_layers: int = 2,
        out_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim, self.seq_len, self.out_dim = in_dim, seq_len, out_dim
        self.backbone = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F)
        z, (h_n, _) = self.backbone(x)       # h_n: (num_layers, B, H)
        last = h_n[-1]                        # (B,H)
        y = self.head(last)
        return y

# -----------------------------
# 2) TCN Forecaster (causal, dilated)
# -----------------------------
class _CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, k, dilation=1):
        padding = (k - 1) * dilation
        super().__init__(in_ch, out_ch, k, padding=padding, dilation=dilation)

    def forward(self, x):
        out = super().forward(x)
        # Trim to ensure causality
        trim = self.padding[0]
        if trim > 0:
            out = out[:, :, :-trim]
        return out

class _TCNBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = _CausalConv1d(ch, ch, k, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = _CausalConv1d(ch, ch, k, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B,C,T)
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x + res

class TCNForecaster(nn.Module):
    """
    输入: (B,T,F) → 1x1升维 → 多个Dilated残差块 → 取最后时刻通道向量 → Linear
    """
    def __init__(
        self,
        in_dim: int = 16,
        seq_len: int = 32,
        channels: int = 128,
        levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        out_dim: int = 1,
    ):
        super().__init__()
        self.in_dim, self.seq_len, self.out_dim = in_dim, seq_len, out_dim
        self.backbone = nn.ModuleDict()
        self.backbone["proj"] = nn.Conv1d(in_dim, channels, kernel_size=1)
        blocks = []
        for i in range(levels):
            dilation = 2 ** i
            blocks.append(_TCNBlock(channels, kernel_size, dilation, dropout))
        self.backbone["tcn"] = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F) → (B,F,T)
        x = x.transpose(1, 2)
        z = self.backbone["proj"](x)
        z = self.backbone["tcn"](z)          # (B,C,T)
        last = z[:, :, -1]                   # (B,C)
        y = self.head(last)
        return y

# -----------------------------
# 3) Transformer Forecaster
# -----------------------------
class TransformerForecaster(nn.Module):
    """
    输入: (B,T,F) → 线性映射到 d_model + PosEnc → TransformerEncoder → 取最后token → Linear
    """
    def __init__(
        self,
        in_dim: int = 16,
        seq_len: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        out_dim: int = 1,
    ):
        super().__init__()
        self.in_dim, self.seq_len, self.out_dim = in_dim, seq_len, out_dim
        self.backbone = nn.ModuleDict()
        self.backbone["in_proj"] = nn.Linear(in_dim, d_model)
        self.backbone["pos"] = PositionalEncoding(d_model, max_len=seq_len+512)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.backbone["enc"] = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F)
        z = self.backbone["in_proj"](x)
        z = self.backbone["pos"](z)
        z = self.backbone["enc"](z)          # (B,T,D)
        last = z[:, -1, :]                   # (B,D)
        y = self.head(last)
        return y

# -----------------------------
# 4) N-BEATS Lite（简化多层全连接）
# -----------------------------
class NBeatsLite(nn.Module):
    """
    简化版 N-BEATS：直接把 (T,F) flatten → MLP 堆叠。
    适合作为轻量基线；不做 backcast/forecast 分解。
    """
    def __init__(
        self,
        in_dim: int = 16,
        seq_len: int = 32,
        width: int = 256,
        depth: int = 4,
        out_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim, self.seq_len, self.out_dim = in_dim, seq_len, out_dim
        inp = in_dim * seq_len
        layers = []
        for i in range(depth):
            layers += [nn.Linear(inp if i == 0 else width, width), nn.ReLU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F) → (B, T*F)
        z = x.reshape(x.size(0), -1)
        z = self.backbone(z)
        y = self.head(z)
        return y
