# Federated Multi-Model Cohort (MPC) — README

> 端到端的联邦学习多模型模拟框架。支持 FedPer / FedPerCohort / FedOpt / MultiModel / 鲁棒 DP-FedProx 等策略；内置 LoRA/MLP 适配器注入、(ε,δ)-DP 与 zCDP 记账、批量基准实验编排。

---

## ✨ 主要特性

* **批量实验编排**：`experiments.yaml` 列出多实验；`run_bench.py` 顺序执行并汇总到 `bench_results/`
* **多策略对比**：`multimodel`、`robust`（DP-FedProx）、`fedopt`、`fedper`、`fedper_cohort`
* **个性化与部分共享**：FedPer 客户端仅共享指定前缀参数（如 `adapter.` / `encoder.`）
* **差分隐私全链路**：客户端/服务器端 L2 裁剪 + 高斯噪声；zCDP ↔ (ε,δ) 换算与分桶记账
* **适配器模块化**：一键向 LSTM/TCN/Transformer/N-BEATS 注入 LoRA/MLP Adapter
* **统一模型注册**：`registry.py` 为时间序列模型提供一致构建接口
* **完善审计**：JSONL 运行日志、配置哈希与匿名化工具

---

## 📦 目录结构

```
src/
├── run_bench.py                # 基准运行器（批量读取 experiments.yaml）
├── run_multimodel_sim.py       # 多模型联邦模拟主程序（CLI 入口）
├── strategies/
│   ├── fedper_cohort.py        # FedPer 按签名/模型分桶，聚合共享子集，支持服务器/客户端 DP
│   ├── fedopt.py               # FedOptLike：FedAvg 后接服务器端 Adam/Yogi
│   ├── multimodel_cohort.py    # 同轮多模型分桶聚合，可选 DP 与审计
│   └── robust_dp_fedprox.py    # Delta 裁剪+加噪 的鲁棒 DP-FedProx（加权/中位数/截尾均值）
├── clients/
│   ├── fedper_client.py        # 仅共享特定前缀参数的个性化客户端 + 本地 DP
│   └── hetero_client.py        # 异构模型整模共享 + 参数签名校验 + 更新范数统计 + 客户端 DP
├── models/
│   ├── adapters.py             # LoRA/MLP Adapter 与 inject_adapters()
│   ├── registry.py             # LSTM/TCN/Transformer/N-BEATS 构造注册表
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

```bash
# 建议 Python ≥ 3.10
pip install torch numpy pandas pyyaml rich
pip install flwr              # Flower 联邦框架
# 可选：ray（并行/分布式模拟）、scipy、tqdm 等
```

> 实际依赖以你的 `requirements.txt` 为准。

---

## 🖥️ 命令行用法（`run_multimodel_sim.py --help`）

**策略与训练**

* `--strategy {multimodel,robust,fedopt,fedper,fedper_cohort}`：选择联邦策略
* `--models MODELS`：模型名列表（逗号分隔，如 `lstm,tcn,transformer,nbeats`）
* `--rounds ROUNDS`：联邦轮数
* `--clients CLIENTS`：客户端总数（模拟数）
* `--min_fit MIN_FIT`：每轮最少参与训练的客户端数
* `--lr LR`：客户端本地学习率
* `--num_examples` / `--num_val_examples`：示例/验证数据规模（若启用内置数据）

**聚合与正则**

* `--aggregator {weighted,median,trimmed}`：聚合器（加权/中位数/截尾）
* `--trim_ratio TRIM_RATIO`：截尾比例（0.05\~0.2 常用）
* `--mu_prox MU_PROX`：FedProx 正则系数（`robust` 策略）

**FedOpt（服务器端优化）**

* `--fedopt_variant {adam,yogi}`
* `--fedopt_lr FEDOPT_LR`：服务器端优化器学习率

**差分隐私（DP）**

* `--dp_mode {client,server,none}`：DP 发生在客户端/服务器端/关闭
* `--dp_max_norm DP_MAX_NORM`：L2 裁剪阈值（全局）
* `--dp_sigma DP_SIGMA`：高斯噪声 σ（全局）
* `--dp_clip_map DP_CLIP_MAP`：分模型/分桶裁剪阈值（字符串；见示例）
* `--dp_sigma_map DP_SIGMA_MAP`：分模型/分桶噪声 σ
* `--dp_delta_map DP_DELTA_MAP`：分模型/分桶 δ
* `--dp_policy_id DP_POLICY_ID`：DP 策略标识（用于审计/记账）

**适配器（LoRA/MLP）与共享子集**

* `--adapter_mode {none,lora,mlp}`：是否注入适配器
* `--adapter_rank ADAPTER_RANK`：LoRA rank / MLP bottleneck
* `--adapter_alpha ADAPTER_ALPHA`：LoRA scaling α
* `--adapter_dropout ADAPTER_DROPOUT`：适配器 dropout
* `--share_prefix SHARE_PREFIX`：只共享包含该前缀的参数名（如 `adapter.`、`encoder.`）
* `--freeze_non_adapter`：冻结所有非 `share_prefix` 参数（典型 FedPer 设置）

---

## 🚀 快速开始

> 下面的命令可直接复制执行。Windows PowerShell 用 `^` 续行；macOS/Linux 用 `\` 续行，或写成一行。

### 1) FedPer-Cohort + LoRA（只共享适配器）

**Windows**

```bat
python src\run_multimodel_sim.py ^
  --strategy fedper_cohort ^
  --models transformer ^
  --rounds 10 --clients 20 --min_fit 10 ^
  --lr 1e-3 ^
  --aggregator weighted ^
  --dp_mode none ^
  --adapter_mode lora --adapter_rank 8 --adapter_alpha 16 --adapter_dropout 0.05 ^
  --share_prefix adapter. --freeze_non_adapter
```

**Linux/macOS**

```bash
python src/run_multimodel_sim.py \
  --strategy fedper_cohort \
  --models transformer \
  --rounds 10 --clients 20 --min_fit 10 \
  --lr 1e-3 \
  --aggregator weighted \
  --dp_mode none \
  --adapter_mode lora --adapter_rank 8 --adapter_alpha 16 --adapter_dropout 0.05 \
  --share_prefix adapter. --freeze_non_adapter
```

### 2) FedOpt（Yogi）服务器端优化

```bash
python src/run_multimodel_sim.py \
  --strategy fedopt \
  --models lstm \
  --rounds 30 --clients 50 --min_fit 30 \
  --lr 5e-4 \
  --aggregator weighted \
  --dp_mode none \
  --fedopt_variant yogi --fedopt_lr 0.01
```

### 3) 鲁棒 DP-FedProx（中位数 + 服务器端 DP）

```bash
python src/run_multimodel_sim.py \
  --strategy robust \
  --models tcn \
  --rounds 20 --clients 40 --min_fit 20 \
  --lr 1e-3 --mu_prox 0.01 \
  --aggregator median \
  --dp_mode server --dp_max_norm 1.5 --dp_sigma 0.6 \
  --dp_policy_id robust_demo
```

### 4) 多模型同轮分桶 + 截尾均值

```bash
python src/run_multimodel_sim.py \
  --strategy multimodel \
  --models lstm,tcn,transformer \
  --rounds 15 --clients 60 --min_fit 30 \
  --lr 1e-3 \
  --aggregator trimmed --trim_ratio 0.1 \
  --dp_mode none
```

### 5) FedPer（非分桶）+ 客户端 DP + 共享 `encoder.`

```bash
python src/run_multimodel_sim.py \
  --strategy fedper \
  --models transformer \
  --rounds 10 --clients 20 --min_fit 10 \
  --lr 1e-3 \
  --aggregator weighted \
  --dp_mode client --dp_max_norm 1.0 --dp_sigma 0.8 \
  --share_prefix encoder. --freeze_non_adapter
```

---

## 🧩 DP Map 传参示例（分模型/分桶）

> 你的 CLI 接收**字符串**，常见两种写法，按代码解析选择其一。

**A) JSON 字符串（最稳妥）**

```bash
python src/run_multimodel_sim.py \
  --strategy multimodel \
  --models lstm,transformer \
  --rounds 10 --clients 20 --min_fit 10 \
  --dp_mode server \
  --dp_clip_map '{"lstm":1.0,"transformer":0.7}' \
  --dp_sigma_map '{"lstm":0.8,"transformer":0.6}' \
  --dp_delta_map '{"lstm":1e-5,"transformer":1e-6}'
```

**B) 逗号分隔 k=v**

```bash
python src/run_multimodel_sim.py \
  --strategy multimodel \
  --models lstm,transformer \
  --rounds 10 --clients 20 --min_fit 10 \
  --dp_mode server \
  --dp_clip_map lstm=1.0,transformer=0.7 \
  --dp_sigma_map lstm=0.8,transformer=0.6 \
  --dp_delta_map lstm=1e-5,transformer=1e-6
```

---

## 🧪 批量基准：`run_bench.py`

`experiments.yaml` 示例（字段名与 CLI **一致**，`run_bench.py` 会拼接成命令行）：

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
    models: transformer
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
    models: lstm,tcn,transformer
    aggregator: trimmed
    trim_ratio: 0.1
```

运行：

```bash
python src/run_bench.py --config experiments.yaml --out bench_results
```

输出：`bench_results/` 下包含每个实验的日志、指标与汇总。

---

## 📊 审计与记账

* **日志**：`utils/audit.py` 以 JSONL 记录轮次事件、聚合、DP 参数、指标等，并写入配置哈希（复现实验）
* **DP**：`privacy/dp.py` 提供 `l2_clip()`、`add_gaussian_noise()`、`estimate_sigma()`
* **记账**：`privacy/accountant.py` 支持 zCDP 与 (ε,δ) 互换，并按「模型桶」累计隐私预算

---

## 🧯 故障排查

* **`ImportError: cannot import name 'add_gaussian_noise'`**
  确认 `privacy/dp.py` 中存在函数并正确导入：`from privacy.dp import add_gaussian_noise`

* **`TypeError: ... got an unexpected keyword argument 'on_fit_config_fn'`**
  Flower 版本/策略构造签名不匹配。以 `run_multimodel_sim.py --help` 为准修正入参；必要时调整 Flower 版本。

* **共享子集为空**
  确认已注入适配器且参数名前缀与 `--share_prefix` 一致；查看日志中的“共享参数计数/占比”。

* **DP 噪声过大导致精度骤降**
  适度增大 `dp_max_norm` 或降低 `dp_sigma`；结合 zCDP 记账评估 `rounds × sample_fraction` 的累计影响。

---

## 🧷 最佳实践

* 先用 **`aggregator=weighted` + `dp_mode=none`** 验证流程与收敛，再逐步叠加 DP/鲁棒聚合/个性化。
* 截尾均值 `trim_ratio` 建议从 **0.1** 起试；中位数在**客户端数较少**时可能不稳定。
* FedPer 模式常用：`--share_prefix adapter.` + `--freeze_non_adapter`，只上传适配器层。
* 保存 **配置哈希**、`(ε,δ)`、`sigma`、`clip`、`rounds`、`sample_fraction`、`aggregator`、适配器占比等关键信息。

---

## 📝 许可证

根据实际项目选择（MIT/Apache-2.0/BSD-3-Clause 等）。

---

如果你想把 **当前实验** 精确地写进一份 `experiments.yaml`，直接把你打算的参数发我，我按这套 CLI 格式生成可用模板。
