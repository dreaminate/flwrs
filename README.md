# Federated Multi-Model Cohort (MPC) — README

> 端到端的联邦学习多模型模拟框架，支持 FedPer/FedOpt 类/鲁棒 DP-FedProx/多模型分桶等策略，内置 LoRA/MLP 适配器注入、(ε,δ)-DP 隐私记账与基准实验批量运行。

---

## ✨ 主要特性

* **批量实验编排**：`experiments.yaml` 描述多组实验；`run_bench.py` 顺序执行并汇总至 `bench_results/`
* **多策略对比**：`FedPerCohort`、`MultiModelCohort`、`FedOptLike`、`RobustDPFedProx`
* **个性化与部分共享**：FedPer 客户端仅共享指定前缀参数（如 `adapter.`）
* **差分隐私全链路**：本地/服务器端 L2 裁剪 + 高斯噪声；zCDP↔(ε,δ) 换算与账本
* **适配器模块化**：一键向 LSTM/TCN/Transformer/N-BEATS 注入 LoRA/MLP Adapter
* **统一模型注册**：`registry.py` 为各时间序列模型提供一致的构建接口
* **完整审计**：配置哈希、匿名化与 JSONL 运行日志

---

## 📦 目录结构

```
src/
├── run_bench.py                # 基准运行器（批量读取 experiments.yaml）
├── run_multimodel_sim.py       # 多模型模拟主程序（命令行入口）
├── strategies/
│   ├── fedper_cohort.py        # FedPerCohort 分桶 + 子集共享 + DP
│   ├── fedopt.py               # FedOptLike（服务器端 Adam/Yogi）
│   ├── multimodel_cohort.py    # MultiModelCohort 同轮多模型分桶聚合
│   └── robust_dp_fedprox.py    # 鲁棒 DP-FedProx（中值/截尾均值等）
├── clients/
│   ├── fedper_client.py        # 仅共享指定前缀参数的个性化客户端 + 本地DP
│   └── hetero_client.py        # 异构模型客户端（整模共享 + 签名校验）
├── models/
│   ├── adapters.py             # LoRA/MLP Adapter 与 inject_adapters()
│   ├── registry.py             # LSTM/TCN/Transformer/N-BEATS 统一注册
│   └── timeseries.py           # 时间序列模型实现
├── privacy/
│   ├── accountant.py           # zCDP↔(ε,δ) 换算与分桶记账
│   └── dp.py                   # L2 裁剪、加噪、σ 估算、DP 配置
├── utils/
│   ├── audit.py                # JSONL 日志、配置哈希、字符串匿名化
│   ├── common.py               # 参数打包/解包与共享子集保存
│   └── make_anchor.py          # 生成锚点样本（src/data/anchor）
└── data/                       # 你的数据与输出（可自定义）
bench_results/                  # 基准汇总结果（由 run_bench 产出）
experiments.yaml                # 批量实验配置
```

---

## 🛠 环境与安装

建议 Python ≥ 3.10。常用依赖（按需增减）：

```bash
pip install torch numpy pandas pyyaml rich
pip install flwr  # Flower 联邦框架
# 可选：ray（分布式/并行模拟）、scipy、tqdm 等
```

> 实际依赖以你的 `requirements.txt` 为准；如策略构造签名与 Flower 版本不匹配，请参考下文「故障排查」。

---

## 🚀 快速开始

### 1) 编写 `experiments.yaml`

```yaml
# experiments.yaml
defaults:
  rounds: 10
  clients: 20
  sample_fraction: 0.5         # 每轮参与客户端比例
  local_epochs: 1
  batch_size: 64
  adapter:
    enable: true
    type: lora                  # lora 或 mlp
    share_prefix: "adapter."    # FedPer 共享前缀
  dp:
    client:
      enable: true
      l2_clip: 1.0
      sigma: 0.8                # 本地高斯噪声
    server:
      enable: true
      l2_clip: 1.5
      sigma: 0.5
    accountant:
      delta: 1e-5
      mechanism: "zcdp"         # 记账方式
  aggregator:
    type: "trimmed_mean"        # mean|median|trimmed_mean
    trim_ratio: 0.1

experiments:
  - name: fedper_lstm_lora
    strategy: "fedper_cohort"
    model: "lstm"
    model_args:
      input_size: 64
      hidden_size: 128
      num_layers: 2

  - name: fedopt_transformer
    strategy: "fedopt_like"
    model: "transformer"
    server_opt:
      type: "adam"              # adam 或 yogi
      lr: 0.01
    aggregator:
      type: "mean"

  - name: robust_dp_fedprox_tcn
    strategy: "robust_dp_fedprox"
    model: "tcn"
    fedprox_mu: 0.01
    aggregator:
      type: "median"
```

> 字段名示例仅作参考，请以 `run_multimodel_sim.py --help` 与源码参数为准。

### 2) 执行批量基准

```bash
# 从仓库根目录执行
python src/run_bench.py --config experiments.yaml --out bench_results
```

完成后在 `bench_results/` 下可见每个实验的配置、指标与日志汇总。

### 3) 运行单次实验（示例）

```bash
python src/run_multimodel_sim.py \
  --strategy fedper_cohort \
  --model transformer \
  --rounds 10 --clients 20 --sample-fraction 0.5 \
  --local-epochs 1 --batch-size 64 \
  --adapter.enable true --adapter.type lora --adapter.share-prefix "adapter." \
  --dp.client.enable true --dp.client.l2-clip 1.0 --dp.client.sigma 0.8 \
  --dp.server.enable true --dp.server.l2-clip 1.5 --dp.server.sigma 0.5 \
  --aggregator.type trimmed_mean --aggregator.trim-ratio 0.1
```

> 参数开关/命名以实际实现为准；推荐先查看 `--help`。

---

## 🧩 关键模块说明

### FedPer 客户端（`clients/fedper_client.py`）

* **仅共享** 参数名带指定前缀（如 `adapter.`）的层；其余为本地个性化参数。
* 发送前可执行 **本地 DP**：L2 裁剪 + 高斯噪声。
* 自动统计与上报共享子集的参数数量与范数。

### 异构模型客户端（`clients/hetero_client.py`）

* 整模权重共享；带 **参数签名校验** 与 **更新范数** 统计。
* 按客户端级配置执行本地 DP。

### 分桶与聚合策略

* **FedPerCohort**（`strategies/fedper_cohort.py`）：
  按模型/签名分桶，仅聚合共享子集；支持服务器端/客户端 DP；落地隐私日志。
* **MultiModelCohort**（`strategies/multimodel_cohort.py`）：
  同轮训练多个模型，自动指纹匹配与分桶聚合，可选 DP。
* **FedOptLike**（`strategies/fedopt.py`）：
  先加权平均，再用服务器端 Adam/Yogi 对全局参数做一步优化。
* **RobustDPFedProx**（`strategies/robust_dp_fedprox.py`）：
  Delta 裁剪 + 高斯噪声 + 鲁棒聚合（均值/中位数/截尾均值）+ FedProx 正则。

### 适配器与模型注册

* **adapters.py**：提供 LoRA/MLP Adapter 封装与 `inject_adapters(model, ...)`。
  统计并返回共享参数（前缀匹配）的规模与占比。
* **registry.py**：统一注册 LSTM/TCN/Transformer/N-BEATS，
  通过字符串标识与 `model_args` 进行构建，供策略按需实例化。
* **timeseries.py**：各时间序列模型的具体实现（精简/教学友好）。

### 差分隐私与记账

* **dp.py**：

  * `l2_clip(tensor, max_norm)`：按组或全局裁剪
  * `add_gaussian_noise(tensor, sigma)`：加噪
  * `estimate_sigma(eps, delta, ...)`：根据 (ε,δ) 反推噪声倍数（如启用）
* **accountant.py**：

  * 支持 zCDP 与 (ε,δ) 的换算
  * **按模型桶** 聚合与记录 **累计隐私预算**

### 审计与通用工具

* **audit.py**：

  * 运行配置哈希（用于可复现标识）
  * 匿名化工具（对客户端 ID/字符串做一致哈希）
  * JSONL 事件流（轮次、聚合、DP 参数、指标等）
* **common.py**：

  * 参数打包/解包（整模或共享子集）
  * 权重保存（本地快照/共享快照）

---

## 📊 输出与目录约定

* `bench_results/`：`run_bench.py` 的整体实验汇总（含每个实验的配置、曲线、表格/JSON）
* `logs/`（如有）：按实验名/时间戳划分的 JSONL 审计日志
* `src/data/anchor/`：`make_anchor.py` 生成的锚点样本
* 其他：按你在 `run_multimodel_sim.py` 中的设定进行保存（如模型快照）

---

## 🔐 DP 速查

* **本地 DP**：客户端在发送前对“共享参数”执行 `clip → noise`
* **服务器端 DP**：服务器对收到的更新再做 `clip → noise → 聚合`
* **zCDP 记账**：多轮复用时，zCDP 可加和，再换算为目标 (ε,δ)
* 建议显式记录：`{clip, sigma, mechanism, delta, rounds, sample_fraction}` 并写入审计日志

---

## 🧪 可复现性

* 每次运行会输出 **配置哈希**（来自 `audit.py`）
* 建议固定随机种子、样本抽样比例、客户端顺序与数据切分
* 将 `experiments.yaml` 与生成的 JSONL 一并归档

---

## 🧯 常见问题（FAQ / Troubleshooting）

1. **`ImportError: cannot import name 'add_gaussian_noise'`**

   * 请确认 `privacy/dp.py` 中存在该函数，且上层正确 `from privacy.dp import add_gaussian_noise`。
   * 若文件/模块层级调整，请更新相对导入路径。

2. **`TypeError: ... got an unexpected keyword argument 'on_fit_config_fn'`**

   * 你的策略构造函数不接受此关键字，或 Flower 版本与示例签名不一致。
   * 解决：查看 `strategies/*.py` 实际 `__init__` 参数；使用 `run_multimodel_sim.py --help` 支持的 CLI；必要时升级/降级 Flower，使签名匹配。

3. **聚合器选择与鲁棒性**

   * 极端/脏数据较多时，优先 `median` 或 `trimmed_mean`；
   * 截尾比例 `trim_ratio` 不宜过大（常用 0.05\~0.2）。

4. **适配器共享为空**

   * 确认已注入 LoRA/MLP Adapter，且共享前缀（如 `adapter.`）与实际参数名一致；
   * 在日志中检查“共享参数计数/占比”。

5. **DP 噪声过大，精度显著下降**

   * 逐步调小 `sigma` 或增大 `l2_clip`；
   * 在相同 ε 目标下，增加本地 epoch 或 rounds 并不总是更优，需结合记账评估。

---

## 🧷 最佳实践小贴士

* **先通再专**：先用 `aggregator=mean`、不加 DP 验证流程完整与收敛，再叠加 DP/鲁棒聚合/个性化。
* **分桶对齐**：多模型/多签名分桶时，保证每桶最少客户端数，以免中值/截尾均值退化。
* **精确审计**：把 rounds、采样率、(ε,δ)、clip、sigma、聚合器、适配器占比一并写入 JSONL。
* **实验自描述**：`experiments.yaml` 中尽量写清策略、模型超参、DP/聚合设置，便于复现实验。

---

## 📝 许可证

请根据你的项目实际选择（MIT/Apache-2.0/BSD-3-Clause 等）。

---

## 🙌 致谢

* Flower 等联邦学习生态
* 社区时间序列建模工作（LSTM/TCN/Transformer/N-BEATS 等）

---

### 附：最小可用 `experiments.yaml`（极简示例）

```yaml
defaults:
  rounds: 5
  clients: 5
  sample_fraction: 1.0
  local_epochs: 1
  batch_size: 32
  adapter:
    enable: false
  dp:
    client: { enable: false }
    server: { enable: false }
  aggregator:
    type: "mean"

experiments:
  - name: quickstart_lstm
    strategy: "fedper_cohort"
    model: "lstm"
    model_args:
      input_size: 16
      hidden_size: 32
      num_layers: 1
