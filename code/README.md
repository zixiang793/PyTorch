# PyTorch 实操代码

| 文件 | 用途 |
| --- | --- |
| [MLP.py](MLP.py) | 原始 MNIST 两层感知机，固定学习率，适合从头阅读训练流程 |
| [MLP_lr_compare.py](MLP_lr_compare.py) | 相同条件下对比 fixed、Warmup + Cosine、OneCycleLR |

新增对比运行：`python MLP_lr_compare.py`。参数仍集中在文件开头，使用 CUDA；三组完成后依次显示指标对比和学习率曲线。TensorBoard 使用同一组标签，可勾选不同策略叠加观察。数值和权重输出到本地 `outputs/lr_compare_<时间>/`；完整实验说明与结果见 [实验记录](../experiments/lr_schedulers/README.md)。

## 原始 MNIST 两层感知机：GPU 训练与依次查看结果

代码全部在 `MLP.py` 中，按顺序展开。仍然没有自定义类、辅助函数或命令行参数；实验名称和训练参数直接在文件开头修改。

参考视频：[识别手写数字——为什么你不该死磕 PyTorch 语法](https://www.bilibili.com/video/BV1uPGj6fEQh)。网络为 `784 → 128 → 10`，使用 ReLU、交叉熵、SGD，学习率 0.1，batch size 128，默认训练 5 轮。

## 运行与查看结果

```powershell
conda activate ptl-dl
cd D:\a_study_data\Agent\bnu\DL\code
python MLP.py
```

使用完整 MNIST：60,000 张训练图片、10,000 张测试图片。首次运行自动下载，之后复用 `data/`。代码明确使用 CUDA GPU，模型、训练数据和评估数据都会移到 GPU；启动时会打印显卡名称，需要在支持 CUDA 的 `ptl-dl` 环境中运行。

**训练、评估和预测计算全部完成后**，才开始显示 Matplotlib 图窗，不保存图片文件。一次只显示一个，点击右上角叉号关闭后，再弹出下一个，顺序如下：

- **Loss and Accuracy**：训练集与测试集的 loss、准确率曲线。
- **Confusion Matrix**：测试集混淆矩阵，行是真实数字，列是预测数字。
- **Predictions**：固定 9 张测试图片的预测；T 为真实数字，P 为预测数字，绿色正确、红色错误。

训练过程中不弹出图窗，进度和指标显示在终端。查看图窗时可以拖动和缩放；关闭第三个图窗后程序退出。图表使用英文标题以避免中文字体兼容问题，代码注释使用中文。

终端找不到 conda 时，可直接使用环境解释器：

```powershell
& 'D:\a_study_data\Agent\bnu\Miniconda3\envs\ptl-dl\python.exe' 'D:\a_study_data\Agent\bnu\DL\code\MLP.py'
```

## TensorBoard：之后比较不同实验

每次运行都在 `runs/` 下创建独立日志目录，名称包含模型、隐藏层大小、学习率、batch size、轮数与时间。例如：

```text
MLP_h128_lr0.1_bs128_epochs5_<时间>
```

日志记录训练/测试 loss、准确率、模型计算图、权重与梯度分布，以及本次参数。梯度分布来自本轮最后一个训练 batch。Matplotlib 的图像只在窗口显示，不写入图片文件或 TensorBoard 图像日志。

想比较实验时，再启动 TensorBoard：

```powershell
conda activate ptl-dl
cd D:\a_study_data\Agent\bnu\DL\code
python -m tensorboard.main --logdir runs --port 6006
```

打开 [TensorBoard](http://127.0.0.1:6006/)，勾选多次实验。在 Scalars 中可以对比不同实验的同一指标，在 Custom Scalars 中可以对比每次实验的 train/test 曲线。

以后学习 CNN 时，定义相应的网络，修改 `experiment_name`，也把日志写到此 `runs/` 目录，并继续使用 `Loss/train`、`Loss/test`、`Accuracy/train`、`Accuracy/test` 这四个指标名称。这样可以直接与 MLP 的记录叠加比较。仅修改实验名称不会把 MLP 网络变成 CNN。

进行对比时尽量保持相同的数据划分、预处理、随机种子和评估口径。每轮结束后，用同一版固定模型评估完整训练集和测试集；loss 按图片数加权平均，准确率为总正确数除以总图片数。

测试集不参与训练。正式调参时，应另外划分验证集，根据验证集选择参数，最后再评估测试集。

## 文件保存位置

- `data/MNIST/`：MNIST 数据，复用已有下载。
- `runs/<实验名称>/`：TensorBoard 日志，供之后对比。
- `outputs/<实验名称>/final.pth`：最后一轮的模型参数。

Matplotlib 不保存 PNG、JPG 等绘图文件。之前版本的实验记录仍保留在原位置，使用 `--logdir runs` 也能读取。
