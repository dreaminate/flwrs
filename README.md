
---

# Federated Multi-Model Cohort (MPC) — README

> 端到端的联邦学习多模型模拟框架。支持 FedPer / FedPerCohort / FedOpt / MultiModel / 鲁棒 DP-FedProx；内置 LoRA/MLP 适配器注入、(ε,δ)-DP 与 zCDP 记账、批量基准实验编排。

---

## ✨ 主要特性

* **批量实验编排**：`experiments.yaml` 列出多实验；`run_bench.py` 顺序执行并汇总到 `bench_results/`
* **多策略对比**：`multimodel`、`robust`（DP-FedProx）、`fedopt`、`fedper`、`fedper_cohort`
* **个性化与部分共享**：FedPer 客户端仅共享指定前缀参数（如 `adapter.` / `encoder.`）
* **差分隐私全链路**：客户端/服务器端 L2 裁剪 + 高斯噪声；zCDP ↔ (ε,δ) 换算与分桶记账
* **适配器模块化**：一键向 LSTM/TCN/**TS-Transformer (`ts_transformer`)**/N-BEATS 注入 LoRA/MLP Adapter
* **统一模型注册**：`registry.py` 为时间序列模型提供一致构建接口
* **完善审计**：JSONL 运行日志、配置哈希与匿名化工具

> 提示：注册表里的 Transformer 键名是 **`ts_transformer`**（也支持 `tft` 别名）。

---

## 📦 目录结构

```
src/
├── run_bench.py                # 基准运行器（批量读取 experiments.yaml）
├── run_multimodel_sim.py       # 多模型联邦模拟主程序（CLI 入口）
├── strategies/
│   ├── fedper_cohort.py        # FedPer 分桶聚合 + 服务器/客户端 DP
│   ├── fedopt.py               # FedOptLike：FedAvg 后接 Adam/Yogi
│   ├── multimodel_cohort.py    # 同轮多模型分桶聚合（可选 DP 与审计）
│   └── robust_dp_fedprox.py    # 鲁棒 DP-FedProx（中位数/截尾均值等）
├── clients/
│   ├── fedper_client.py        # 仅共享特定前缀参数的个性化客户端 + 本地 DP
│   └── hetero_client.py        # 异构模型整模共享 + 签名校验 + 更新范数 + 客户端 DP
├── models/
│   ├── adapters.py             # LoRA/MLP Adapter 与 inject_adapters()
│   ├── registry.py             # LSTM/TCN/TS-Transformer/N-BEATS 构造注册表
│   └── timeseries.py           # 时间序列模型实现
├── privacy/
│   ├── accountant.py           # zCDP ↔ (ε,δ) 换算，按模型桶记账
│   └── dp.py                   # DP 配置、L2 裁剪、添加高斯噪声、σ 估算
├── utils/
│   ├── audit.py                # JSONL 日志、配置哈希、字符串匿名化
│   ├── common.py               # 参数打包/解包、整模/子集权重保存
│   └── make_anchor.py          # 生成锚点样本（输出至 src/data/anchor/）
└── data/                       # 数据与输出（可自定义）
bench_results/                  # run_bench 的结果汇总
experiments.yaml                # 批量实验配置
```

---

## 🛠 环境与安装

```powershell
# 建议 Python ≥ 3.10
pip install torch numpy pandas pyyaml rich
pip install flwr              # Flower 联邦框架
# 可选：ray（并行/分布式模拟）、scipy、tqdm 等
```

> 实际依赖以你的 `requirements.txt` 为准。

---

## 🖥️ 命令行用法（`run_multimodel_sim.py --help`）

* **策略与训练**：
  `--strategy {multimodel,robust,fedopt,fedper,fedper_cohort}`
  `--models lstm,tcn,ts_transformer,nbeats...`（逗号分隔）
  `--rounds` / `--clients` / `--min_fit` / `--lr` / `--num_examples` / `--num_val_examples`

* **聚合与正则**：
  `--aggregator {weighted,median,trimmed}`，`--trim_ratio`，`--mu_prox`

* **FedOpt**：
  `--fedopt_variant {adam,yogi}`，`--fedopt_lr`

* **差分隐私（DP）**：
  `--dp_mode {client,server,none}`，`--dp_max_norm`，`--dp_sigma`，
  `--dp_clip_map` / `--dp_sigma_map` / `--dp_delta_map`，`--dp_policy_id`

* **适配器/共享**：
  `--adapter_mode {none,lora,mlp}`，`--adapter_rank`，`--adapter_alpha`，`--adapter_dropout`，
  `--share_prefix`，`--freeze_non_adapter`

---

## 🚀 快速开始（PowerShell 一行可复制）

### ✅ 冒烟测试（FedPer-Cohort + TS-Transformer，无 DP）

```powershell
python .\src\run_multimodel_sim.py --strategy fedper_cohort --models ts_transformer --rounds 1 --clients 2 --min_fit 2 --lr 1e-3 --aggregator weighted --dp_mode none
```

### FedOpt（Yogi）服务器端优化

```powershell
python .\src\run_multimodel_sim.py --strategy fedopt --models lstm --rounds 30 --clients 50 --min_fit 30 --lr 5e-4 --aggregator weighted --dp_mode none --fedopt_variant yogi --fedopt_lr 0.01
```

> 小贴士：请使用**修复版** `src/strategies/fedopt.py`（首轮初始化动量并规范返回），否则可能报 `NoneType` 错。你可以直接采用这里的替换实现。

### 鲁棒 DP-FedProx（中位数 + 服务器端 DP）

```powershell
python .\src\run_multimodel_sim.py --strategy robust --models tcn --rounds 20 --clients 40 --min_fit 20 --lr 1e-3 --mu_prox 0.01 --aggregator median --dp_mode server --dp_max_norm 1.5 --dp_sigma 0.6 --dp_policy_id robust_demo
```

### 多模型同轮分桶 + 截尾均值

```powershell
python .\src\run_multimodel_sim.py --strategy multimodel --models lstm,tcn,ts_transformer --rounds 15 --clients 60 --min_fit 30 --lr 1e-3 --aggregator trimmed --trim_ratio 0.1 --dp_mode none
```

### FedPer（非分桶）+ 客户端 DP + 共享 `encoder.`

> **重要**：客户端 DP 的裁剪/噪声请用 **map 方式** 设置；`--dp_max_norm/--dp_sigma` 在 `client` 模式下**不生效**。

```powershell
python .\src\run_multimodel_sim.py --strategy fedper --models ts_transformer --rounds 10 --clients 20 --min_fit 10 --lr 1e-3 --aggregator weighted --dp_mode client --dp_clip_map ts_transformer:1.0 --dp_sigma_map ts_transformer:0.8 --share_prefix encoder. --freeze_non_adapter
```

---

## 🧠 运行方式解读（这些命令分别做什么）

### 1) 冒烟测试（FedPer-Cohort + TS-Transformer，无 DP）

* **做什么**：`fedper_cohort` + 单模型 `ts_transformer`，1 轮、2 客户端、加权聚合，不加 DP。
* **意义**：最快验证**端到端链路**（本地训练→上传→分桶聚合→日志）。
* **机制**：FedPer 的“**分桶**”版，会按模型/签名把客户端分桶并只聚合共享子集（本例等价于整模共享）。
* **何时用**：首次跑通环境/接口。

### 2) FedOpt（Yogi）服务器端优化

* **做什么**：先 **FedAvg** 出目标参数，再在**服务器端**用 **Yogi/Adam** 对全局参数来一步优化（`--fedopt_lr` 控制步长）。
* **意义**：缓解 non-IID 震荡，**更平滑**地更新全局。
* **关键参数**：`--fedopt_variant yogi|adam`，`--fedopt_lr`。
* **取舍**：步长太大可能发散，太小更新慢；一般从 `1e-2~1e-3` 试。

### 3) 鲁棒 DP-FedProx（中位数 + 服务器端 DP）

* **做什么**：中位数/截尾均值抗异常；**服务器端 DP** 对聚合更新裁剪+加噪；本地加入 **FedProx** 正则。
* **意义**：兼顾**隐私**与**鲁棒性**。
* **关键参数**：`--aggregator median|trimmed`、`--trim_ratio`、`--dp_mode server`、`--dp_max_norm`、`--dp_sigma`、`--mu_prox`。
* **取舍**：`dp_sigma` 大→精度掉；`mu_prox` 大→本地学习受限。

### 4) 多模型同轮分桶 + 截尾均值

* **做什么**：`multimodel` 同轮推进多模型（如 `lstm,tcn,ts_transformer`），按指纹分桶，各桶独立聚合；聚合器用 **截尾均值**。
* **意义**：并行推进**异构模型**，便于**对比实验**。
* **关键参数**：`--models`、`--aggregator trimmed`、`--trim_ratio`（建议 0.05\~0.2）。
* **取舍**：单桶有效样本过少会不稳。

### 5) FedPer（非分桶）+ 客户端 DP + 共享 `encoder.`

* **做什么**：**只上传**包含 `encoder.` 的参数，其余保持个性化；上传前做**本地 DP**。
* **意义**：在**个性化/隐私**与**协作**间折中。
* **关键参数**：`--dp_mode client`、`--dp_clip_map/--dp_sigma_map`、`--share_prefix`。
* **取舍**：共享子集太小→全局进步慢；`dp_sigma` 过大→性能降。

> **选型速查**
> 先跑通：冒烟（FedPer-Cohort，单模型，无 DP）
> 更稳更平滑：FedOpt（Yogi/Adam）
> 隐私与鲁棒：鲁棒 DP-FedProx（中位/截尾 + 服务器端 DP）
> 多模型并行：MultiModel（同轮分桶）
> 强个性化：FedPer（共享 `encoder.`/`adapter.`；可加本地 DP）

---

## 🔀 参数可排列组合吗？（组合规则与禁配清单）

### 速查表

| 策略 (`--strategy`) | `--models` 多模型 | `--aggregator` median/trimmed |            `--dp_mode server` | `--dp_mode client` | `--dp_*_map` 分桶DP | `--share_prefix` 只共享前缀 | `--adapter_mode` LoRA/MLP |
| ----------------- | -------------: | ----------------------------: | ----------------------------: | -----------------: | ----------------: | ---------------------: | ------------------------: |
| `multimodel`      |           ✅ 推荐 |                   ✅（按**桶**生效） |                             ✅ |    ✅（作为**全局默认**下发） |           ✅（按模型桶） |      ❌（除非用 FedPer 客户端） |                         ✅ |
| `fedper_cohort`   |    ✅（按**桶**聚合） |                   ✅（按**桶**生效） |                             ✅ |    ✅（作为**全局默认**下发） |           ✅（按模型桶） |             ✅（只上传命中前缀） |                         ✅ |
| `fedper`          |    ❌（**仅单模型**） |                             ✅ |                             ✅ |                  ✅ |                 ❌ |             ✅（只上传命中前缀） |                         ✅ |
| `robust`          |    ❌（**仅单模型**） |                             ✅ | ✅（`--dp_max_norm/--dp_sigma`） |                  ✅ |                 ❌ |                ❌（整模共享） |                         ✅ |
| `fedopt`          |    ❌（**仅单模型**） |                 ❌（内部先 FedAvg） |                             ✅ |                  ✅ |                 ❌ |                ❌（整模共享） |                         ✅ |

**关键注意事项**

1. **单/多模型**：`fedopt`/`robust`/`fedper` 只能**单模型**；`multimodel`/`fedper_cohort` 支持**多模型分桶**。
2. **前缀共享**：只有 `fedper`/`fedper_cohort` 真正“只上传前缀”；其他策略里 `--share_prefix` 不改变上传集合（但 `--freeze_non_adapter` 仍会冻结训练）。
3. **DP 互斥**：`--dp_mode` 三选一（`client`/`server`/`none`），**不能同时开**。
4. **分桶 DP**：`--dp_*_map`（如 `--dp_clip_map`）只在 `multimodel`/`fedper_cohort` 生效；其余策略忽略。
5. **鲁棒聚合参与数**：`median` ≥3 更稳；`trimmed` 建议 `min_fit ≥ 10` 且 `trim_ratio ≤ 0.2`。
6. **FedOpt 聚合器**：`fedopt` 无视 `--aggregator`，内部是 FedAvg → 服务器端 Adam/Yogi。
7. **LoRA/MLP**：`--adapter_mode` 对所有策略都能注入；只有 `fedper`/`fedper_cohort` + `--share_prefix adapter.` 才会“只上传适配器”。
8. **客户端 DP 提醒**：`client` 模式下，请用 `--dp_clip_map/--dp_sigma_map` 设置；`--dp_max_norm/--dp_sigma` 对客户端不生效。

**组合建议**

* 先跑通：`fedper_cohort` + 单模型 + `aggregator=weighted` + `dp_mode=none`
* 更稳：`fedopt`（Yogi/Adam）+ 单模型
* 隐私与鲁棒：`robust` + `aggregator=median/trimmed` + `dp_mode=server` + `dp_max_norm/sigma`
* 多模型并行：`multimodel` + `models=lstm,tcn,ts_transformer` + `aggregator=trimmed` + 可选分桶 DP *map*
* 强个性化：`fedper`/`fedper_cohort` + `--share_prefix adapter.` + `--freeze_non_adapter` + 可选 `dp_mode=client`

---

## 🧩 DP Map 传参示例（分模型/分桶）

```powershell
python .\src\run_multimodel_sim.py --strategy multimodel --models lstm,ts_transformer --rounds 10 --clients 20 --min_fit 10 --dp_mode server --dp_clip_map '{"lstm":1.0,"ts_transformer":0.7}' --dp_sigma_map '{"lstm":0.8,"ts_transformer":0.6}' --dp_delta_map '{"lstm":1e-5,"ts_transformer":1e-6}'
```

**或 k\:v 逗号分隔（依据你代码解析）**

```powershell
python .\src\run_multimodel_sim.py --strategy multimodel --models lstm,ts_transformer --rounds 10 --clients 20 --min_fit 10 --dp_mode server --dp_clip_map lstm:1.0,ts_transformer:0.7 --dp_sigma_map lstm:0.8,ts_transformer:0.6 --dp_delta_map lstm:1e-5,ts_transformer:1e-6
```

---

## 🧪 批量基准：`run_bench.py`

`experiments.yaml`（与 CLI 字段一致）：

```yaml
defaults:
  rounds: 10
  clients: 20
  min_fit: 10
  lr: 0.001
  aggregator: weighted
  dp_mode: none
  adapter_mode: none

experiments:
  - name: fedper_lora_only_adapter
    strategy: fedper_cohort
    models: ts_transformer
    adapter_mode: lora
    adapter_rank: 8
    adapter_alpha: 16
    adapter_dropout: 0.05
    share_prefix: adapter.
    freeze_non_adapter: true

  - name: fedopt_yogi_lstm
    strategy: fedopt
    models: lstm
    fedopt_variant: yogi
    fedopt_lr: 0.01

  - name: robust_dp_tcn
    strategy: robust
    models: tcn
    aggregator: median
    mu_prox: 0.01
    dp_mode: server
    dp_max_norm: 1.5
    dp_sigma: 0.6
    dp_policy_id: robust_demo

  - name: multimodel_trimmed
    strategy: multimodel
    models: lstm,tcn,ts_transformer
    aggregator: trimmed
    trim_ratio: 0.1
```

运行与输出：

```powershell
python .\src\run_bench.py --config .\experiments.yaml --out .\bench_results
# 结果在 .\bench_results\
```

---

## 🧱 扩展模型：如何添加你自己的 model

> 目标：让新模型名可以直接出现在 `--models your_model` 中，并能参与 `multimodel/fedper_cohort` 分桶、适配器注入、FedPer 前缀共享、DP 记账与保存。

### 1) 在 `timeseries.py` 实现模型（建议接口）

* 参考 `LSTMForecaster/TCNForecaster/TransformerForecaster/NBeatsLite`，前向：输入 `(B,T,F)` → 输出 `(B,out_dim)`
* 构造函数建议包含：`in_dim`, `seq_len`, `out_dim`
* 为兼容 `inject_adapters()`，尽量用标准 `nn.Linear/nn.Conv` 作为可注入点。

### 2) 在 `models/registry.py` 注册构建器

```python
# models/registry.py
from .timeseries import MyFancyForecaster
DEFAULT_SEQ_LEN = 32
DEFAULT_IN_DIM  = 16
DEFAULT_OUT_DIM = 1

def make_myfancy():
    return MyFancyForecaster(in_dim=DEFAULT_IN_DIM, seq_len=DEFAULT_SEQ_LEN, out_dim=DEFAULT_OUT_DIM)

MODEL_BUILDERS.update({
    "myfancy": make_myfancy,
    "fancy": make_myfancy,  # 可选别名
})
```

> 之后即可用：`--models myfancy`，或与其他模型并行：`--models lstm,myfancy,ts_transformer`（仅限 `multimodel/fedper_cohort`）。

### 3) 适配器与 FedPer 前缀（可选）

* 只上传适配器：`--adapter_mode lora --share_prefix adapter. --freeze_non_adapter`
* 只上传编码器：为编码器层加名 `encoder.*`，运行时用 `--share_prefix encoder. --freeze_non_adapter`

### 4) 形状与分桶

* **单模型策略**（`fedopt/robust/fedper`）要求**所有客户端模型同形状**（参数维度一致）。
* **多模型策略**（`multimodel/fedper_cohort`）按**模型键名**分桶；**每个桶内**形状需一致。

### 5) 冒烟测试你的新模型

```powershell
# 单模型 + FedPer-Cohort（无 DP）
python .\src\run_multimodel_sim.py --strategy fedper_cohort --models myfancy --rounds 1 --clients 2 --min_fit 2 --lr 1e-3 --aggregator weighted --dp_mode none

# 多模型并行
python .\src\run_multimodel_sim.py --strategy multimodel --models lstm,myfancy,ts_transformer --rounds 2 --clients 6 --min_fit 3 --lr 1e-3 --aggregator trimmed --trim_ratio 0.1 --dp_mode none
```

> 只共享适配器：加 `--adapter_mode lora --share_prefix adapter. --freeze_non_adapter`
> 客户端 DP：加 `--dp_mode client --dp_clip_map myfancy:1.0 --dp_sigma_map myfancy:0.8`

---

## 📊 审计与记账

* **日志**：`utils/audit.py` 输出 JSONL（轮次、聚合、DP 参数、指标）+ 配置哈希
* **DP**：`privacy/dp.py` 提供 `l2_clip()`、`add_gaussian_noise()`、`estimate_sigma()`
* **记账**：`privacy/accountant.py` 支持 zCDP 与 (ε,δ) 互换，并按「模型桶」累计隐私预算

---

## 🧯 故障排查

* `KeyError: 'transformer'`：请用 **`ts_transformer`**（或 `tft`）作为模型名；或在注册表中添加 `transformer` 的别名映射。
* `ModuleNotFoundError` / 找不到脚本：请先 `cd` 到仓库根目录；必要时用绝对路径。
* 共享子集为空：确认已注入适配器且前缀与 `--share_prefix` 一致；查看日志“共享参数计数/占比”。
* DP 噪声过大导致精度下降：增大 `--dp_max_norm` 或减小 `--dp_sigma`；结合 zCDP 记账评估 `rounds × sample_fraction`。
* `TypeError: unexpected keyword`：以 `--help` 为准对齐参数；不同策略/Flower 版本签名可能不同。
* `ValueError: not enough values to unpack`：某些 Flower 版本 evaluate 返回二元组；本仓库已做兼容，如你定制了聚合回调，注意分别处理二元/三元返回。

---

## 🧷 最佳实践

* 先用 **`aggregator=weighted` + `dp_mode=none`** 验证流程与收敛，再逐步叠加 DP/鲁棒聚合/个性化
* 截尾均值 `trim_ratio` 建议从 **0.1** 起试；中位数在**客户端数少**时可能不稳
* FedPer 常用：`--share_prefix adapter.` + `--freeze_non_adapter`（仅上传适配器层）
* 审计务必记录：`(ε,δ)`、`sigma`、`clip`、`rounds`、`sample_fraction`、`aggregator`、共享占比等

---

## 🧭 路线图 / Future Plan

* **DP 调度器（Noise/Clip Scheduler）**

  * 像学习率调度器一样，按轮动态调整 `σ/clip`。
  * 计划支持：`--dp_sched {const,cosine,linear,warmup}` 与 `--dp_sched_warmup`（前若干轮 warmup）。
  * 与 `privacy/accountant.py` 联动，实现**目标 ε/δ 驱动**、训练中**预算累计**与**早停**。
* **服务器端 DP 调度**：在 `robust_dp_fedprox.py`、`multimodel_cohort.py`、`fedper_cohort.py` 中按 `server_round` 缩放各桶 `sigma`。
* **安全聚合（Secure Aggregation, 成对掩码）**：隐藏单客户端更新，降低反推风险；与 DP 可叠加。
* **分群/分桶自适应配置**：按客户端统计/指标动态下发 `clip/sigma`、适配器 rank、参与率等。
* **Opacus/TFF 选件**：可切换为 Opacus（PyTorch）或 TFF（TensorFlow）的 DP 组件与调度器。
* **鲁棒聚合增强**：Krum、Bulyan、Coordinate-wise median/trimmed 等可插拔。
* **可视化**：ε-轮次曲线、隐私-精度帕累托前沿、分桶对比图。

---

## 🧮 DP 调度器（预览接口）

> 本节为预览设计，合并后即可使用；与现有 CLI 兼容。

```powershell
# 客户端 DP 调度（余弦退火），均值归一化保证总预算稳定
python .\src\run_multimodel_sim.py ^
  --strategy fedper_cohort --models ts_transformer ^
  --rounds 30 --clients 30 --min_fit 15 ^
  --dp_mode client --dp_clip_map ts_transformer:1.0 --dp_sigma_map ts_transformer:0.9 ^
  --dp_sched cosine --dp_sched_warmup 0.1
```

> 思路：在 `on_fit_config_fn(rnd)` 中按轮缩放 `sigma`；若设目标 `(ε*, δ)`，在 `privacy/accountant.py` 里累计 zCDP，达到阈值提前停止。

---

## 📚 FedAvg / Adam / Yogi 对比与选型

| 算法          | 思想                                             | 优点                  | 可能问题          | 适用建议              |
| ----------- | ---------------------------------------------- | ------------------- | ------------- | ----------------- |
| **FedAvg**  | 各客户端本地训练后，按样本数加权**平均参数**                       | 简单稳定、通信少            | 非 IID 时易抖/收敛慢 | 作为基线；数据较均衡时足够     |
| **FedAdam** | 用 `g = target - params` 作“梯度”，服务器端 **Adam** 更新 | 自适应动量，收敛更平滑/通常更快    | 步长不当可能发散；超参敏感 | 想加速/平滑、分布差异中等     |
| **FedYogi** | 与 Adam 类似，但二阶动量更新抑制无界增长                        | 对高方差/非 IID 更稳健，不易发散 | 仍需调 `lr`；可能略慢 | non-IID 明显或震荡大时优先 |

**本项目实现**：`--strategy fedopt` 先做 **FedAvg** 得目标参数，再用 **Adam/Yogi** 在服务器端更新（`--fedopt_variant adam|yogi`，`--fedopt_lr` 控制步长）。
**实践建议**：先用 FedAvg 对齐流程 → 试 Yogi（更稳）→ 再试 Adam（更快）。`--fedopt_lr` 从 `1e-3~1e-2` 起试。

---

## 📝 许可证

根据你的项目选择（MIT/Apache-2.0/BSD-3-Clause 等）。

---
