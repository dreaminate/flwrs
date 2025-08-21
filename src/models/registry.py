from __future__ import annotations
from typing import Dict, Callable
import torch.nn as nn

from .timeseries import (
    LSTMForecaster,
    TCNForecaster,
    TransformerForecaster,
    NBeatsLite,
)

# 统一默认形状：真实项目里请改成你的 (seq_len, in_dim, out_dim)
DEFAULT_SEQ_LEN = 32
DEFAULT_IN_DIM  = 16
DEFAULT_OUT_DIM = 1

def make_lstm() -> nn.Module:
    return LSTMForecaster(in_dim=DEFAULT_IN_DIM, seq_len=DEFAULT_SEQ_LEN, out_dim=DEFAULT_OUT_DIM)

def make_tcn() -> nn.Module:
    return TCNForecaster(in_dim=DEFAULT_IN_DIM, seq_len=DEFAULT_SEQ_LEN, out_dim=DEFAULT_OUT_DIM)

def make_ts_transformer() -> nn.Module:
    return TransformerForecaster(in_dim=DEFAULT_IN_DIM, seq_len=DEFAULT_SEQ_LEN, out_dim=DEFAULT_OUT_DIM)

def make_nbeatslite() -> nn.Module:
    return NBeatsLite(in_dim=DEFAULT_IN_DIM, seq_len=DEFAULT_SEQ_LEN, out_dim=DEFAULT_OUT_DIM)

# 为兼容之前的名字：把 "tft" 映射到 Transformer，"nbeats" 映射到简化版 N-BEATS
MODEL_BUILDERS: Dict[str, Callable[[], nn.Module]] = {
    "lstm": make_lstm,
    "tcn": make_tcn,
    "ts_transformer": make_ts_transformer,
    "nbeatslite": make_nbeatslite,
    "tft": make_ts_transformer,   # 兼容名
    "nbeats": make_nbeatslite,    # 兼容名
}
