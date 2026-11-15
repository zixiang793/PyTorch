"""MNIST 两层感知机：按顺序阅读、直接运行即可。

运行环境：conda activate ptl-dl
开始训练：python MLP.py
查看日志：python -m tensorboard.main --logdir runs --port 6006
"""

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")  # 计算全部结束后，用独立窗口依次展示结果。
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


# ==================== 1. 设置参数 ====================
# 想做对比实验，直接修改这里的数值。
experiment_name = "MLP"  # 实验名称；以后换成 CNN 等模型时，也修改这个名称。
epochs = 5              # 完整学习训练集的次数。
batch_size = 128        # 每次用多少张图片更新一次参数。
hidden_size = 128       # 隐藏层的神经元数量。
learning_rate = 0.1     # 每次沿梯度方向更新参数的步长。

torch.manual_seed(42)
torch.set_num_threads(4)  # 只限制 CPU 辅助计算的线程数，模型训练使用下面的 GPU。
device = torch.device("cuda")
print(f"使用 GPU：{torch.cuda.get_device_name(device)}")

# 路径以本文件的位置为起点，因此不受终端当前工作目录影响。
code_dir = Path(__file__).resolve().parent
run_name = (f"{experiment_name}_h{hidden_size}_lr{learning_rate}_bs{batch_size}_epochs{epochs}_"
            + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
log_dir = code_dir / "runs" / run_name
output_dir = code_dir / "outputs" / run_name
output_dir.mkdir(parents=True)


# ==================== 2. 准备数据 ====================
transform = transforms.Compose([
    transforms.ToTensor(),  # 像素从 0~255 变为 0~1；图像形状变为 [1, 28, 28]。
    transforms.Normalize((0.1307,), (0.3081,)),  # 标准化：(像素 - 均值) / 标准差。
])

# 第一次运行时自动下载，之后直接读取本地数据。
train_dataset = datasets.MNIST(code_dir / "data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(code_dir / "data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 这两个 loader 只用于每轮结束后的评估，固定顺序，且不更新参数。
train_eval_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# ==================== 3. 定义网络、损失函数和优化器 ====================
# “两层”指两个 Linear 层，即一个隐藏层和一个输出层。
# nn.Sequential 会让数据依次经过下面的各层，无需另外定义类和 forward 方法。
model = nn.Sequential(
    nn.Flatten(),                    # [B, 1, 28, 28] -> [B, 784]；B 是 batch 中的图片数。
    nn.Linear(784, hidden_size),      # [B, 784] -> [B, 128]
    nn.ReLU(),                       # 加入非线性，让网络能学习更复杂的规律。
    nn.Linear(hidden_size, 10),       # [B, 128] -> [B, 10]，对应数字 0~9 的分数。
).to(device)

# 输出是原始分数 logits，交叉熵内部会处理 log-softmax，所以模型末尾不加 Softmax。
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(model)


# ==================== 4. 准备 TensorBoard ====================
writer = SummaryWriter(log_dir=str(log_dir))
writer.add_graph(model, torch.zeros(1, 1, 28, 28, device=device))
writer.add_text("Parameters", f"Model: {experiment_name}, hidden size: {hidden_size}, "
                f"learning rate: {learning_rate}, batch size: {batch_size}, epochs: {epochs}, seed: 42")

# 在 Custom Scalars 页面，把训练集与测试集曲线画在同一个坐标轴上。
writer.add_custom_scalars({"Train vs Test": {
    "Loss": ["Multiline", ["Loss/train", "Loss/test"]],
    "Accuracy": ["Multiline", ["Accuracy/train", "Accuracy/test"]],
}})

# 留下 9 张测试图片，训练完成后展示它们的预测结果。
example_images, example_labels = next(iter(test_loader))
example_images = example_images[:9]
example_labels = example_labels[:9]


# 每轮结束后，将完整训练集和测试集的指标加入列表，用来画曲线。
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []


# ==================== 5. 开始训练 ====================
for epoch in range(1, epochs + 1):
    model.train()  # 切换到训练模式。

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)  # 真实标签是 0~9 的整数，形状为 [B]。

        optimizer.zero_grad()                  # 清空上一个 batch 留下的梯度。
        outputs = model(images)                # ① 前向传播：得到每张图片的 10 个类别分数。
        loss = loss_function(outputs, labels)  # ② 计算损失：预测分数与真实标签差多少。
        loss.backward()                        # ③ 反向传播：计算各参数的梯度。
        optimizer.step()                       # ④ 更新参数：沿梯度下降方向调整权重。

    # ==================== 6. 评估训练集和测试集 ====================
    # 每轮训练结束后，用同一个固定模型评估完整数据集，方便公平地比较两条曲线。
    # eval() 切换模型行为；no_grad() 关闭梯度记录，两者作用不同。
    model.eval()
    confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)
    print(f"Epoch {epoch}/{epochs}", end=" | ")

    with torch.no_grad():
        # 训练集和测试集的评估过程相同，依次计算即可。
        for dataset_name, loader in [("train", train_eval_loader), ("test", test_loader)]:
            total_loss = 0.0
            total_correct = 0

            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                predicted_labels = outputs.argmax(dim=1)  # 选取分数最高的类别。

                # loss.item() 是当前 batch 的平均值，乘图片数后才能累计总损失。
                # 最后一个 batch 可能不足 256 张，因此不能直接对各 batch 的均值再取平均。
                total_loss += loss.item() * images.size(0)
                total_correct += (predicted_labels == labels).sum().item()

                if dataset_name == "test" and epoch == epochs:
                    # 只统计最终模型的混淆矩阵：行是真实数字，列是预测数字。
                    for true_label, predicted_label in zip(labels.cpu(), predicted_labels.cpu()):
                        confusion_matrix[true_label.item(), predicted_label.item()] += 1

            average_loss = total_loss / len(loader.dataset)
            accuracy = total_correct / len(loader.dataset)
            writer.add_scalar(f"Loss/{dataset_name}", average_loss, epoch)
            writer.add_scalar(f"Accuracy/{dataset_name}", accuracy, epoch)
            if dataset_name == "train":
                train_losses.append(average_loss)
                train_accuracies.append(accuracy * 100)  # Matplotlib 用百分数显示准确率。
            else:
                test_losses.append(average_loss)
                test_accuracies.append(accuracy * 100)
            print(f"{dataset_name}: loss={average_loss:.4f}, accuracy={accuracy:.2%}", end=" | ")
    print(flush=True)

    # 这里的梯度来自本轮最后一个训练 batch，评估过程不会改变它。
    for name, parameter in model.named_parameters():
        writer.add_histogram(f"Weights/{name}", parameter.detach().cpu(), epoch)
        writer.add_histogram(f"Gradients/{name}", parameter.grad.detach().cpu(), epoch)

    writer.flush()  # 及时写入磁盘，训练时也能在 TensorBoard 中看到更新。


# ==================== 7. 完成计算，保存模型和日志 ====================
with torch.no_grad():
    example_predictions = model(example_images.to(device)).argmax(dim=1).cpu()

# 保存最后一轮的参数；测试集只用于观察，不参与参数更新或最佳模型选择。
torch.save(model.state_dict(), output_dir / "final.pth")
writer.close()
print(f"模型已保存：{output_dir / 'final.pth'}")
print(f"本次 TensorBoard 实验：{run_name}")
print("计算已全部结束。接下来依次展示三张图，关闭当前图窗后显示下一张。")


# ==================== 8. 依次查看结果，不保存图片 ====================
# block=True 会等待当前图窗关闭，再继续执行下一段绘图代码。
# 每看完一张再创建下一张，因此不会同时弹出多个窗口。
plt.ioff()

# 第一张：完整训练过程的 loss 和准确率曲线。
curve_figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4))
curve_figure.canvas.manager.set_window_title("1/3 - MNIST Loss and Accuracy")
epoch_numbers = range(1, epochs + 1)
loss_axis.plot(epoch_numbers, train_losses, "o-", label="Train")
loss_axis.plot(epoch_numbers, test_losses, "o-", label="Test")
loss_axis.set(title="Loss", ylabel="Cross entropy")
accuracy_axis.plot(epoch_numbers, train_accuracies, "o-", label="Train")
accuracy_axis.plot(epoch_numbers, test_accuracies, "o-", label="Test")
accuracy_axis.set(title="Accuracy", ylabel="Accuracy (%)")
for axis in (loss_axis, accuracy_axis):
    axis.set_xlabel("Epoch")
    axis.set_xticks(epoch_numbers)
    axis.legend()
    axis.grid(alpha=0.3)
curve_figure.tight_layout()
plt.show(block=True)

# 第二张：最终模型的混淆矩阵，对角线越集中，预测越准确。
matrix_figure, matrix_axis = plt.subplots(figsize=(6, 5))
matrix_figure.canvas.manager.set_window_title("2/3 - MNIST Confusion Matrix")
heatmap = matrix_axis.imshow(confusion_matrix.numpy(), cmap="Blues")
matrix_figure.colorbar(heatmap, ax=matrix_axis)
matrix_axis.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit",
                ylabel="True digit", title="Test confusion matrix")
matrix_figure.tight_layout()
plt.show(block=True)

# 第三张：9 张图片的最终预测，绿色正确、红色错误。
prediction_figure, prediction_axes = plt.subplots(3, 3, figsize=(6, 6))
prediction_figure.canvas.manager.set_window_title("3/3 - MNIST Predictions")
prediction_figure.suptitle("T = true label, P = predicted label")
for index, axis in enumerate(prediction_axes.flat):
    image = example_images[index, 0] * 0.3081 + 0.1307  # 还原标准化，恢复为 0~1 像素。
    true_label = example_labels[index].item()
    predicted_label = example_predictions[index].item()
    axis.imshow(image.numpy(), cmap="gray", vmin=0, vmax=1)
    axis.set_title(f"T: {true_label}, P: {predicted_label}",
                   color="green" if true_label == predicted_label else "red")
    axis.axis("off")
prediction_figure.tight_layout()
plt.show(block=True)
