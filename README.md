# PyTorch 与深度学习学习记录

用代码复现学到的概念，再用可重复的实验检查效果。

- **`code/`**：自己的 PyTorch 实操代码，优先保留清晰的中文注释和顺序展开的训练流程。
- **`experiments/`**：选定实验的配置、数值结果和分析，跟随 Git 提交，方便在 GitHub 上回看。
- **CS231n**：结合斯坦福 [CS231n 2025](https://cs231n.stanford.edu/2025/) 学习视觉与深度学习。课程资料放在本地 `cs231n-2025/`，按现有 `.gitignore` 配置不上传。

## 学习进展

| 阶段 | 做了什么 | 代码与记录 |
| --- | --- | --- |
| 01 · MLP 入门 | MNIST 两层感知机，GPU 训练，TensorBoard 与计算完成后的逐个弹窗 | [MLP.py](code/MLP.py) · [运行说明](code/README.md) |
| 02 · 学习率调度 | 在相同 MLP、初始化和数据顺序下，对比固定学习率、Warmup + Cosine、OneCycleLR | [对比代码](code/MLP_lr_compare.py) · [实验记录](experiments/lr_schedulers/README.md) |

## 本次新增：学习率调度对比

原有 [MLP.py](code/MLP.py) 保留为入门基线。新增脚本一次跑完三种策略，使用相同的 `784 → 128 → 10` 网络和 SGD，主要变化如下：

| 策略 | 学习率怎么变化 | 与固定组相比改变了什么 |
| --- | --- | --- |
| `fixed` | 始终为 0.1 | 在本次对比中重新运行的基线 |
| `warmup_cosine` | 前 10% 更新从 0.01 线性升到 0.1，随后余弦降到 0.0001 | 手写升温与退火公式，直观看每个 batch 的学习率 |
| `onecycle` | 前 30% 更新升温，随后下降；起点 0.01、最高 0.1、终点 0.0001 | 使用 PyTorch OneCycleLR，关闭动量调度以控制变量 |

两种调度都按 **batch** 改变学习率。三组的网络、随机种子、batch 顺序、训练轮数、预处理和评估口径相同；每轮评估完整训练集与测试集。具体配置、实测数值和局限见 [实验记录](experiments/lr_schedulers/README.md)。

### 本次实测（完整 MNIST，5 轮，seed=42）

| 策略 | 训练准确率 | 测试准确率 | 测试 loss | 相对基线 |
| --- | --- | --- | --- | --- |
| fixed | 98.04% | **97.24%** | 0.0947 | — |
| warmup_cosine | 97.28% | **96.80%** | 0.1119 | -0.44 个百分点 |
| onecycle | 97.29% | **96.75%** | 0.1111 | -0.49 个百分点 |

这次配置下固定学习率表现最好。新增调度不等于准确率一定提高；本次记录了短训练预算下的实际取舍。完整参数、逐轮结果和分析见 [实验记录](experiments/lr_schedulers/README.md)。

## 运行

在支持 CUDA 的 conda 环境 `ptl-dl` 中：

```powershell
conda activate ptl-dl
cd code
python MLP.py             # 单独学习原始 MLP
python MLP_lr_compare.py  # 一次完成三种学习率实验
```

两个脚本分别运行即可。对比脚本的参数集中在开头；三组计算全部结束后，先显示指标对比图，关闭后再显示学习率曲线，不保存图片。批量运行时将 `show_plots` 改为 `False`。

需要比较历史记录时：

```powershell
python -m tensorboard.main --logdir runs --port 6006
```

在 [TensorBoard](http://localhost:6006/) 勾选同一个 `lr_compare_<时间>` 下的三组记录。各组使用相同的标签：`Loss/train`、`Loss/test`、`Accuracy/train`、`Accuracy/test`、`Optimization/learning_rate`，以后加入 CNN 等模型也可沿用这些标签。

## Git 与 GitHub 的使用习惯

一次概念改进使用一个功能分支；先提交可运行的代码，再提交对应的实验结果与分析。原始数据、权重和 TensorBoard 日志保存在本地，GitHub 主要保留代码、说明和小体积的 CSV。

参见 [日常 Git 工作流](docs/GIT_WORKFLOW.md)。
