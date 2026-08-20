# Second-pass MNIST Merge Suite

自动化回答下面四个问题：

1. `Dirichlet alpha` 变小时，RegMean 相对 simple average 的优势会不会变强？
2. RegMean 需要多少 `calibration data` 才开始有效？
3. RegMean 恢复的是哪些类别能力？
4. RegMean 恢复的是哪些 client-local 能力？

## 文件

- `core_merge_lab.py`
  - 训练、切分、merge、评估的核心逻辑
- `report_utils_v2.py`
  - CSV / Markdown / SVG 汇总工具
- `run_second_pass_experiments.py`
  - 自动跑 4 组实验
- `cnn_merge_lab.py`
  - CIFAR 上的小型 CNN merge 实验核心逻辑
- `run_cnn_experiments.py`
  - CIFAR CNN 版批量实验入口
- `run_mnist_advanced_experiments.py`
  - MNIST + MLP 的 5 组高级验证实验入口
- `run_gap_diagnostic_experiments.py`
  - MNIST + MLP 的 gap 定位与 gap 缩小实验入口
- `plot_gap_diagnostic_results.py`
  - 单独为 gap diagnostics 结果生成图和 `.md` 摘要

## 四组实验

### 1. Alpha sweep

固定训练预算，扫描：

- `alpha = 5.0`
- `alpha = 1.0`
- `alpha = 0.5`
- `alpha = 0.1`

目标：验证 `non-IID` 越强时，RegMean 是否越优于 simple average。

### 2. Calibration sweep

固定代表性 setting：

- `Dirichlet alpha = 0.5`
- `epochs = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]`

扫描：

- `16`
- `32`
- `64`
- `128`
- `256`
- `512`

目标：验证 RegMean 对 calibration budget 的敏感性。

### 3. Per-class recovery

固定：

- `Dirichlet alpha = 0.5`
- `equal epochs`

输出：

- `class_recovery.csv`
- `class_recovery.svg`

目标：看 RegMean 恢复了哪些类别。

### 4. Per-client recovery

固定：

- `Dirichlet alpha = 0.5`
- `heterogeneous epochs`

输出：

- `client_recovery.csv`
- `client_recovery.svg`

目标：看 RegMean 恢复了哪些 client-local 能力。

## 怎么跑

运行全部 4 组：

```bash
python3 run_second_pass_experiments.py
```

只跑其中几组：

```bash
python3 run_second_pass_experiments.py \
  --groups alpha_sweep calibration_sweep
```

只跑一个 seed 做快速检查：

```bash
python3 run_second_pass_experiments.py \
  --seeds 7
```

## 结果目录

默认输出到：

- `outputs/second_pass_suite`

重点看这些文件：

- `summary_rows.csv`
- `aggregate_summary.csv`
- `aggregate_summary.json`
- `report.md`
- `alpha_sweep_accuracy.svg`
- `alpha_sweep_regmean_delta.svg`
- `calibration_sweep_accuracy.svg`
- `per_class_recovery/dirichlet_medium/class_recovery.csv`
- `per_class_recovery/dirichlet_medium/class_recovery.svg`
- `per_client_recovery/dirichlet_budget_hetero/client_recovery.csv`
- `per_client_recovery/dirichlet_budget_hetero/client_recovery.svg`

## 依赖

至少需要：

```bash
pip install torch torchvision
```

## 建议使用顺序

第一次建议这样跑：

1. 先快速验证

```bash
python3 run_second_pass_experiments.py --seeds 7 --groups alpha_sweep
```

2. 再跑完整四组

```bash
python3 run_second_pass_experiments.py
```

3. 最后优先看：

- `report.md`
- `alpha_sweep_regmean_delta.svg`
- `class_recovery.svg`
- `client_recovery.svg`

## CNN 版完整库

如果你想从 `MNIST + MLP` 往更真实的深层视觉模型过渡，可以直接跑这套 `CNN` 版：

```bash
python3 run_cnn_experiments.py --seeds 7
```

默认支持：

- 数据：
  - `cifar10`
  - `cifar100`
- 模型：
  - `SmallCNN`
- 实验组：
  - `cnn_alpha_sweep`
  - `cnn_calibration_sweep`
  - `cnn_feature_skew`

### 只跑 alpha sweep

```bash
python3 run_cnn_experiments.py \
  --seeds 7 \
  --groups cnn_alpha_sweep
```

### 切到 CIFAR-100

```bash
python3 run_cnn_experiments.py \
  --dataset-name cifar100 \
  --seeds 7
```

### CNN 输出目录

- `outputs/cnn_merge_suite`

优先看：

- `outputs/cnn_merge_suite/report.md`
- `outputs/cnn_merge_suite/cnn_alpha_sweep_accuracy.svg`
- `outputs/cnn_merge_suite/cnn_alpha_sweep_regmean_delta.svg`
- `outputs/cnn_merge_suite/cnn_calibration_sweep_accuracy.svg`

## MNIST 高级 5 组实验

如果你想继续沿用最开始的 `MNIST + MLP`，现在也可以直接跑这 5 组更贴近你当前问题的验证实验：

- `few_shot_recovery`
- `extreme_non_iid`
- `gap_decomposition`
- `layerwise_ablation`
- `weighting_ablation`

运行全部：

```bash
python3 run_mnist_advanced_experiments.py --seeds 7
```

只跑其中几组：

```bash
python3 run_mnist_advanced_experiments.py \
  --seeds 7 \
  --groups few_shot_recovery gap_decomposition
```

输出目录：

- `outputs/mnist_advanced_suite`

这 5 组实验的作用分别是：

- `few_shot_recovery`
  - 看 merge 后少量全局样本能否把 `non-IID` 的 RegMean 推近 `IID` 水平
- `extreme_non_iid`
  - 用 pathological label skew 放大 `simple average` 的失败模式
- `gap_decomposition`
  - 区分 gap 到底来自 merge、输出层错位，还是表征本身不够好
- `layerwise_ablation`
  - 看是不是只 merge 最后层/隐藏层会更稳
- `weighting_ablation`
  - 比较 `equal / sample / epoch / val-loss` 权重设计

## Gap 定位实验

如果你想专门回答“离 centralized 的差距到底来自哪里，以及怎么缩小”，可以直接跑：

```bash
python3 run_gap_diagnostic_experiments.py --seeds 7
```

默认会产出 5 组：

- `gap_sources`
  - 看 gap 更像 head misalignment、merge gap，还是 deeper representation gap
- `function_space`
  - 计算 merged model 和 centralized 的 logits KL、prediction agreement
- `representation_space`
  - 计算隐藏层和 penultimate feature 到 centralized 的 cosine 相似度
- `gap_reduction`
  - 直接比较几种缩 gap 的候选改法
- `systematic_calibration`
  - 围绕 `sample_weighted_avg` 和 `regmean_hidden_layers` 主线，系统比较 bias-only、temperature、vector scaling、last-layer tune、hidden+head tune

输出目录：

- `outputs/gap_diagnostics_suite`

重点看：

- `report.md`
- `diagnostics.json`
- `aggregate_summary.csv`

如果你已经有结果，只想补图和更适合阅读的摘要：

```bash
python3 plot_gap_diagnostic_results.py \
  --results-dir outputs/gap_diagnostics_suite
```

它会生成：

- `diagnostics_report.md`
- `gap_sources_accuracy.svg`
- `gap_reduction_accuracy.svg`
- `function_space_symmetric_kl.svg`
- `function_space_agreement.svg`
- `representation_space_cosine.svg`
- `representation_space_deep_drop.svg`
- `systematic_calibration_regmean_hidden_layers.svg`
- `systematic_calibration_sample_weighted.svg`
