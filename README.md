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

### 鲁棒 DP-FedProx（中位数 + 服务器端 DP）

```powershell
python .\src\run_multimodel_sim.py --strategy robust --models tcn --rounds 20 --clients 40 --min_fit 20 --lr 1e-3 --mu_prox 0.01 --aggregator median --dp_mode server --dp_max_norm 1.5 --dp_sigma 0.6 --dp_policy_id robust_demo
```

### 多模型同轮分桶 + 截尾均值

```powershell
python .\src\run_multimodel_sim.py --strategy multimodel --models lstm,tcn,ts_transformer --rounds 15 --clients 60 --min_fit 30 --lr 1e-3 --aggregator trimmed --trim_ratio 0.1 --dp_mode none
```

### FedPer（非分桶）+ 客户端 DP + 共享 `encoder.`

```powershell
python .\src\run_multimodel_sim.py --strategy fedper --models ts_transformer --rounds 10 --clients 20 --min_fit 10 --lr 1e-3 --aggregator weighted --dp_mode client --dp_max_norm 1.0 --dp_sigma 0.8 --share_prefix encoder. --freeze_non_adapter
```

---

## 🧩 DP Map 传参示例（分模型/分桶）

**JSON 字符串（推荐）**

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

## 📊 审计与记账

* **日志**：`utils/audit.py` 输出 JSONL（轮次、聚合、DP 参数、指标）+ 配置哈希
* **DP**：`privacy/dp.py` 提供 `l2_clip()`、`add_gaussian_noise()`、`estimate_sigma()`
* **记账**：`privacy/accountant.py` 支持 zCDP 与 (ε,δ) 互换，并按「模型桶」累计隐私预算

---

## 🧯 故障排查

* `KeyError: 'transformer'`：请用 **`ts_transformer`**（或 `tft`）作为模型名；或在注册表中添加 `transformer` 的别名映射。
* `ModuleNotFoundError`/找不到脚本：请先 `cd` 到仓库根目录；必要时用绝对路径。
* 共享子集为空：确认已注入适配器且前缀与 `--share_prefix` 一致；查看日志“共享参数计数/占比”。
* DP 噪声过大导致精度下降：增大 `--dp_max_norm` 或减小 `--dp_sigma`；结合 zCDP 记账评估 `rounds × sample_fraction`。
* `TypeError: unexpected keyword`：以 `--help` 为准对齐参数；不同策略/Flower 版本签名可能不同。

---

## 🧷 最佳实践

* 先用 **`aggregator=weighted` + `dp_mode=none`** 验证流程与收敛，再逐步叠加 DP/鲁棒聚合/个性化
* 截尾均值 `trim_ratio` 建议从 **0.1** 起试；中位数在**客户端数少**时可能不稳
* FedPer 常用：`--share_prefix adapter.` + `--freeze_non_adapter`（仅上传适配器层）
* 审计务必记录：`(ε,δ)`、`sigma`、`clip`、`rounds`、`sample_fraction`、`aggregator`、共享占比等

---

## 📝 许可证

根据你的项目选择（MIT/Apache-2.0/BSD-3-Clause 等）。
