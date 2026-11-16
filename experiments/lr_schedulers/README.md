# 实验 02：MLP 的学习率调度对比

## 想回答的问题

在相同网络和训练预算下，学习率始终固定，与先升温再退火相比，收敛过程和最终测试表现有什么不同？

灵感来自视频 [小小学习率竟然能这么显著影响模型训练结果](https://www.bilibili.com/video/BV1NMVn6vE7o)。这里独立实现三组受控对比，不将视频中的数值当作本次实验结果。

## 从原始 MLP 增加了什么

- 保留原有 `code/MLP.py`，在 `code/MLP_lr_compare.py` 中统一重跑三组，避免把不同时期的历史结果直接拼接比较。
- 在训练循环中加入线性 Warmup + Cosine 公式，以及 PyTorch OneCycleLR。
- 每组重置初始化种子，并重新建立同种子的 DataLoader 打乱生成器，使初始权重与每轮 batch 顺序一致。
- TensorBoard 记录每次更新实际使用的学习率；计算全部结束后，依次显示指标对比和学习率曲线。
- 输出配置、每轮指标、最终汇总和逐步学习率 CSV，选取完整实验的配置与小体积结果提交到 Git。

## 实验条件

| 项目 | 配置 |
| --- | --- |
| 数据 | MNIST，训练 60,000，测试 10,000 |
| 预处理 | ToTensor + Normalize(0.1307, 0.3081)，无数据增强 |
| 模型 | Flatten → Linear(784, 128) → ReLU → Linear(128, 10)，101,770 参数 |
| 损失 | CrossEntropyLoss，直接接收 logits |
| 优化器 | SGD，momentum=0，weight_decay=0 |
| 训练预算 | 每组 5 轮，batch size 128；每轮 469 次、总共 2,345 次更新 |
| 随机种子 | 42，每组重新设置，DataLoader 也使用独立的同种子生成器 |
| 评估 | 每轮结束后固定模型，完整训练/测试集；按样本数加权平均 loss |
| 模型保存 | 固定最后一轮，不用测试集选择最佳 checkpoint |

MNIST 的像素转换和标准化只做一次，缓存为 CPU Tensor，再按 batch 送到 GPU。这与原始 `ToTensor + Normalize` 的数值处理相同；抽样逐元素比较一致。三组共用同一份预处理数据，减少重复的 PIL 转换，不改变网络和学习率以外的组间条件。

| 策略 | 起始 LR | 最高 LR | 最后一次更新 LR | 升温占比 | 动量 |
| --- | --- | --- | --- | --- | --- |
| fixed | 0.1 | 0.1 | 0.1 | 无 | 固定 0 |
| warmup_cosine | 0.01 | 0.1 | 0.0001 | 前 234 次更新为线性升温；第 234 步到峰值，step 从 0 起 | 固定 0 |
| onecycle | 0.01 | 约 0.1 | 0.0001 | 前 30%，cos 升温 | 固定 0，`cycle_momentum=False` |

两种调度使用相同的起点、峰值上界与终点，但升温比例和升温曲线不同；这是对这两套配置的比较。OneCycleLR 的峰值处于离散更新序列中，可能略低于设定的 0.1。

## 实现中要注意的地方

Warmup + Cosine 在 `optimizer.step()` 前按当前更新编号计算学习率。OneCycleLR 每个 batch 调用一次 `scheduler.step()`，位于 `optimizer.step()` 后。日志在参数更新前读取优化器中的学习率，因此表示当前 batch 实际使用的数值，不把下一步学习率误画成当前值。

OneCycleLR 默认还会改变动量；本实验明确关闭，以免同时改变学习率和动量。设置 `div_factor=10`、`final_div_factor=100`，使其起点为 0.01，终点为 0.0001；这两个参数也不同于默认值。这里比较的是只调学习率的 OneCycleLR 配置，不代表完整的 OneCycle 动量策略。

## 复现

```powershell
conda activate ptl-dl
cd code
python MLP_lr_compare.py
```

默认使用 CUDA。三组结果放在同一个 `code/outputs/lr_compare_<时间>/`，包括 `config.json`、`summary.csv`、`epoch_metrics.csv`、`learning_rates.csv` 和三组模型权重；TensorBoard 日志位于 `code/runs/lr_compare_<时间>/<策略>/`。

绘图在三组计算全部完成后开始：先显示 4 个指标面板，关闭后显示 3 条学习率曲线。Matplotlib 不保存图片；`show_plots=False` 可以只计算与保存数值。

## 结果

实验日期：2026-11-16。完整运行编号：`lr_compare_20261116_094346_581571`。代码版本为 [48722f2](https://github.com/zixiang793/PyTorch/commit/48722f2)，即先提交代码，再运行并整理本次结果。

环境：Python 3.10.22、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、TensorBoard 2.21.0；显卡为 NVIDIA GeForce RTX 4060 Laptop GPU。本次批量运行关闭了绘图，训练参数与默认配置相同。

| 策略 | 训练准确率 | 测试准确率 | 测试 loss | 相对基线 |
| --- | --- | --- | --- | --- |
| fixed | 98.04% | **97.24%** | 0.0947 | — |
| warmup_cosine | 97.28% | **96.80%** | 0.1119 | -0.44 个百分点 |
| onecycle | 97.29% | **96.75%** | 0.1111 | -0.49 个百分点 |

本次固定学习率在最终测试准确率和 loss 上均优于两个调度组。Warmup + Cosine 比基线低 0.44 个百分点，OneCycleLR 低 0.49 个百分点。**这是调度方案的受控比较，不把未出现的准确率提升写成“改进效果”。**

- [完整配置 config.json](config.json)：数据规模、调度参数、版本与运行编号。
- [最终汇总 summary.csv](summary.csv)：三组最终指标、首尾学习率、更新次数与运行耗时。
- [逐轮指标 epoch_metrics.csv](epoch_metrics.csv)：三组各 5 轮，共 15 行。

每组实际更新 2,345 次，固定组 LR 始终为 0.1，两组调度从 0.01 开始，在最后一次更新降到 0.0001。完整逐步学习率与模型权重保存在本地对应的 `outputs/` 中；Git 中保留上述小体积数值记录。

### 逐轮测试准确率

| Epoch | fixed | warmup_cosine | onecycle |
| --- | --- | --- | --- |
| 1 | 94.55% | 93.64% | 91.93% |
| 2 | 96.03% | 95.71% | 95.24% |
| 3 | 96.75% | 96.47% | 96.30% |
| 4 | 97.20% | 96.76% | 96.73% |
| 5 | 97.24% | 96.80% | 96.75% |

### 本次观察

固定组较早进入较高的准确率区间；两个调度组在开始阶段使用较小学习率，并在第 5 轮前逐渐降到很低的学习率。这会改变有限步数内的有效更新幅度，可以作为短预算下收敛较慢的一个解释，但仅凭这次实验不能单独确定因果。

没有根据测试结果再调整最高学习率、升温比例或训练轮数。下一步若比较更长预算或其他调度参数，应预先确定实验条件，并引入验证集和多随机种子。


## 如何理解比较结果

单次实验使用 seed=42 和固定 5 轮，只说明这些配置在这次训练预算下的表现。学习率调度不保证一定比固定学习率更好；升温也会消耗有限的训练步数。判断时同时看收敛速度、最终 loss 与准确率，不只挑一个最好看的数值。

测试集按用户的教学需求每轮展示，但不参与梯度计算、模型选择或本次调参。正式比较更多方案时，应从训练数据划分验证集，保留测试集做最终评估；多随机种子可以进一步检查结果是否稳定。

## 参考

- [PyTorch OneCycleLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)：更新频率、参数与动量调度。
- [PyTorch 调整学习率](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)：优化器与 scheduler 的调用顺序。
