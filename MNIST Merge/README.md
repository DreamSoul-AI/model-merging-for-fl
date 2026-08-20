# MNIST Merge Lab

“最小可验证实验平台”：

- 模型先固定为 `MNIST + 小 MLP`
- 先验证 `simple average / weighted average / RegMean`
- 先支持最关键的异构来源：
  - 训练步数不同
  - 数据分布不同
- 现在也支持批量实验、汇总表和自动出图

## 现在已经包含的能力

- `SmallMLP`: `784 -> 128 -> 128 -> 10`
- 数据切分：
  - `iid`
  - `pair-label`
  - `dirichlet`
- 本地训练：
  - 所有 client 从同一个初始化开始
  - 可给每个 client 指定不同 epoch
- 合并方法：
  - `simple_avg`
  - `sample_weighted_avg`
  - `epoch_weighted_avg`
  - `regmean_equal`
  - `regmean_epoch_weighted`
- 评估指标：
  - `accuracy`
  - `loss`
  - `macro_accuracy`
  - `worst_class_accuracy`
  - 每类准确率
  - 每个 merged model 在各 client 本地数据上的准确率

## 批量实验包

新增两个脚本：

- `batch_experiments.py`
- `plot_batch_results.py`

它们会把实验整理成这几层产物：

- 每次运行一个 `results.json`
- 一个展开后的 `summary_rows.csv`
- 一个聚合后的 `aggregate_summary.csv`
- 一个便于阅读的 `report.md`
- 两张对比图：
  - `accuracy_comparison.svg`
  - `regmean_vs_simple_delta.svg`

### 一键跑核心实验矩阵

```bash
python3 batch_experiments.py
```

默认会跑：

- `phase0_iid_equal`
- `phase1_iid_budget_hetero`
- `phase2_dirichlet_medium`
- `phase4_dirichlet_budget_hetero`

默认随机种子：

- `7 17 27`

输出目录默认是：

- `outputs/mnist_merge_lab/batch`

### 跑完整 preset 集合

```bash
python3 batch_experiments.py --suite full
```

### 只跑指定实验

```bash
python3 batch_experiments.py \
  --experiments phase1_iid_budget_hetero phase4_dirichlet_budget_hetero \
  --seeds 7 17 27
```

### 重新生成汇总表和图片

如果你已经有一批 JSON 结果，不想重新训练，可以直接重建报告：

```bash
python3 plot_batch_results.py \
  --batch-dir outputs/mnist_merge_lab/batch
```

## 推荐的最小实验

### 1. Phase 0: IID + 相同训练预算

```bash
python3 mnist_merge_lab.py \
  --split-mode iid \
  --num-clients 10 \
  --client-epochs 5 5 5 5 5 5 5 5 5 5
```

预期：`simple_avg` 和 `regmean_equal` 不会差太多，主要用于确认训练、merge、评估流程都通了。

### 2. Phase 1: IID + 不同训练步数

```bash
python3 mnist_merge_lab.py \
  --split-mode iid \
  --num-clients 10 \
  --client-epochs 1 2 3 5 8 10 15 20 30 50
```

预期：`simple_avg` 会被低训练预算 client 拖累，`epoch_weighted_avg` 和 `regmean_epoch_weighted` 通常会更稳。

### 3. Phase 2: Label skew + 相同训练步数

```bash
python3 mnist_merge_lab.py \
  --split-mode dirichlet \
  --dirichlet-alpha 0.5 \
  --num-clients 10 \
  --client-epochs 5 5 5 5 5 5 5 5 5 5
```

预期：看 `macro_accuracy`、`worst_class_accuracy` 和每类准确率，能更容易观察到 simple average 是否丢失局部类别能力。

## RegMean 实现对应关系

代码里每个线性层都按下面这个形式合并：

- 对每个 client 收集 calibration set 上的 `C_HH` 和 `C_ZH`
- 使用闭式解：

```text
A = (sum_k alpha_k C_ZH_k) (sum_k alpha_k C_HH_k + lambda I)^(-1)
```

然后把 `A` 拆回 `weight` 和 `bias`。

## 结果输出

运行后会在这里生成结果：

- `outputs/mnist_merge_lab/results.json`

里面会包含：

- 每个 client 的训练配置和训练集指标
- 每种 merge 方法在测试集上的结果
- 每种 merge 方法在每个 client 本地分片上的表现

如果用批量脚本，则主要看：

- `outputs/mnist_merge_lab/batch/report.md`
- `outputs/mnist_merge_lab/batch/aggregate_summary.csv`
- `outputs/mnist_merge_lab/batch/accuracy_comparison.svg`
- `outputs/mnist_merge_lab/batch/regmean_vs_simple_delta.svg`

## 依赖

当前环境里还没有装这些包，运行前需要先准备：

```bash
pip install torch torchvision
```

## 推荐使用顺序

第一次建议这样走：

1. 先跑一个最小 sanity check

```bash
python3 mnist_merge_lab.py \
  --split-mode iid \
  --num-clients 10 \
  --client-epochs 5 5 5 5 5 5 5 5 5 5
```

2. 再直接跑核心批量实验

```bash
python3 batch_experiments.py
```

3. 最后优先看这三个结果

- `report.md`
- `aggregate_summary.csv`
- `regmean_vs_simple_delta.svg`


