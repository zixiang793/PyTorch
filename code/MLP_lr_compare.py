"""一次比较三种学习率策略。原来的入门脚本 MLP.py 保持独立。

conda activate ptl-dl
python MLP_lr_compare.py
python -m tensorboard.main --logdir runs --port 6006
"""

import csv
import json
import math
import platform
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


# ==================== 1. 三组共用的参数 ====================
epochs = 5
batch_size = 128
hidden_size = 128
seed = 42
max_lr = 0.1            # 固定组的学习率，也是两个调度组的最高学习率。
start_lr = 0.01         # 两个调度组从相同的学习率开始。
min_lr = 0.0001         # 两个调度组在最后一次更新时降到这个学习率。
warmup_ratio = 0.1     # Warmup + Cosine：前 10% 的更新线性升温。
onecycle_pct = 0.3     # OneCycleLR：前 30% 的更新升温，随后下降。
show_plots = True      # 三组全部算完后依次看图；批量运行时可以改为 False。
strategies = ["fixed", "warmup_cosine", "onecycle"]

device = torch.device("cuda")
torch.set_num_threads(4)  # 只控制 CPU 辅助计算，模型和 batch 放在 GPU 上。
code_dir = Path(__file__).resolve().parent
comparison_id = datetime.now().strftime("lr_compare_%Y%m%d_%H%M%S_%f")
output_dir = code_dir / "outputs" / comparison_id
output_dir.mkdir(parents=True)
print(f"GPU: {torch.cuda.get_device_name(device)} | Comparison: {comparison_id}", flush=True)


# ==================== 2. 准备共同的数据 ====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_dataset = datasets.MNIST(code_dir / "data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(code_dir / "data", train=False, download=True, transform=transform)
train_eval_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
total_steps = epochs * steps_per_epoch
warmup_steps = int(total_steps * warmup_ratio)

config = dict(epochs=epochs, batch_size=batch_size, hidden_size=hidden_size, seed=seed,
              max_lr=max_lr, start_lr=start_lr, min_lr=min_lr, warmup_ratio=warmup_ratio,
              onecycle_pct=onecycle_pct, optimizer="SGD", momentum=0, weight_decay=0,
              cycle_momentum=False, strategies=strategies, total_steps=total_steps,
              warmup_steps=warmup_steps, train_samples=len(train_dataset),
              test_samples=len(test_dataset), normalization=dict(mean=0.1307, std=0.3081),
              comparison_id=comparison_id, torch_version=str(torch.__version__),
              torchvision_version=version("torchvision"), tensorboard_version=version("tensorboard"),
              python_version=platform.python_version(), gpu=torch.cuda.get_device_name(device))
(output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
histories = {}     # 保存三组的曲线，最后一起画图。
epoch_rows = []    # 保存每轮的数值，便于做表格和记录到 Git。
lr_rows = []       # 保存每一次更新真正使用的学习率。
summary_rows = []


# ==================== 3. 依次训练三组，不在训练时弹窗 ====================
for strategy in strategies:
    # 每组重新设种子、建立模型，保证初始权重相同。
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10),
    ).to(device)

    # 专门的随机数生成器：三组每一轮的 batch 顺序也相同。
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=shuffle_generator)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0)
    scheduler = None
    if strategy == "onecycle":
        scheduler = OneCycleLR(
            optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
            pct_start=onecycle_pct, anneal_strategy="cos",
            div_factor=max_lr / start_lr, final_div_factor=start_lr / min_lr,
            cycle_momentum=False,  # 只比较学习率，不同时改变动量。
        )

    writer = SummaryWriter(str(code_dir / "runs" / comparison_id / strategy))
    writer.add_text("Parameters", "```json\n" + json.dumps(config, indent=2) + "\n```")
    writer.add_graph(model, torch.zeros(1, 1, 28, 28, device=device))
    writer.add_custom_scalars({"Train vs Test": {
        "Loss": ["Multiline", ["Loss/train", "Loss/test"]],
        "Accuracy": ["Multiline", ["Accuracy/train", "Accuracy/test"]],
    }})
    history = dict(train_loss=[], test_loss=[], train_accuracy=[], test_accuracy=[], lr=[])
    global_step = 0
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if strategy == "warmup_cosine":
                # 直接写出公式：先线性从 start_lr 升到 max_lr，再余弦下降。
                # step 从 0 开始；最后一次更新是 total_steps - 1。
                if global_step < warmup_steps:
                    lr = start_lr + (max_lr - start_lr) * global_step / warmup_steps
                else:
                    progress = (global_step - warmup_steps) / (total_steps - warmup_steps - 1)
                    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
                optimizer.param_groups[0]["lr"] = lr

            # 在更新前记录，画出的就是这个 batch 真正使用的学习率。
            used_lr = optimizer.param_groups[0]["lr"]
            history["lr"].append(used_lr)
            lr_rows.append(dict(strategy=strategy, step=global_step, learning_rate=used_lr))
            writer.add_scalar("Optimization/learning_rate", used_lr, global_step)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            # OneCycleLR 每个 batch 更新一次，顺序在 optimizer.step() 后面。
            if scheduler is not None:
                scheduler.step()
            global_step += 1

        # ==================== 4. 每轮评估完整训练集与测试集 ====================
        model.eval()
        row = dict(strategy=strategy, epoch=epoch)
        with torch.no_grad():
            for dataset_name, loader in [("train", train_eval_loader), ("test", test_loader)]:
                total_loss, total_correct = 0.0, 0
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    total_loss += loss_function(outputs, labels).item() * labels.size(0)
                    total_correct += (outputs.argmax(1) == labels).sum().item()
                average_loss = total_loss / len(loader.dataset)
                accuracy = total_correct / len(loader.dataset)
                history[f"{dataset_name}_loss"].append(average_loss)
                history[f"{dataset_name}_accuracy"].append(accuracy)
                row[f"{dataset_name}_loss"] = average_loss
                row[f"{dataset_name}_accuracy"] = accuracy
                writer.add_scalar(f"Loss/{dataset_name}", average_loss, epoch)
                writer.add_scalar(f"Accuracy/{dataset_name}", accuracy, epoch)
        epoch_rows.append(row)
        for name, parameter in model.named_parameters():
            writer.add_histogram(f"Weights/{name}", parameter.detach().cpu(), epoch)
            writer.add_histogram(f"Gradients/{name}", parameter.grad.detach().cpu(), epoch)
        writer.flush()
        print(f"{strategy:15s} | Epoch {epoch}/{epochs} | "
              f"train loss {row['train_loss']:.4f}, acc {row['train_accuracy']:.2%} | "
              f"test loss {row['test_loss']:.4f}, acc {row['test_accuracy']:.2%}", flush=True)

    # 固定轮数后保存最终模型，不根据测试集挑选最佳模型。
    torch.save(model.state_dict(), output_dir / f"{strategy}.pth")
    writer.close()
    histories[strategy] = history
    summary_rows.append(dict(strategy=strategy, epochs=epochs,
                             train_loss=row["train_loss"], test_loss=row["test_loss"],
                             train_accuracy=row["train_accuracy"], test_accuracy=row["test_accuracy"],
                             first_lr=history["lr"][0], last_lr=history["lr"][-1],
                             updates=global_step, seconds=time.perf_counter() - start_time))


# ==================== 5. 保存数值，方便复现和 GitHub 展示 ====================
# 原始运行产物放在被 Git 忽略的 outputs/，选定的一次实验可整理到 experiments/。
for filename, rows in [("summary.csv", summary_rows), ("epoch_metrics.csv", epoch_rows),
                       ("learning_rates.csv", lr_rows)]:
    with (output_dir / filename).open("w", newline="", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
print(f"三组计算已完成，模型、参数和 CSV 位于：{output_dir}", flush=True)


# ==================== 6. 计算结束后依次看图，不保存图片 ====================
if show_plots:
    plt.ioff()
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    figure.canvas.manager.set_window_title("1/2 - MLP Learning Rate Comparison")
    for axis, metric in zip(axes.flat, ["train_loss", "test_loss", "train_accuracy", "test_accuracy"]):
        for strategy in strategies:
            axis.plot(range(1, epochs + 1), histories[strategy][metric], "o-", label=strategy)
        axis.set(title=metric, xlabel="Epoch")
        axis.set_xticks(range(1, epochs + 1))
        axis.legend()
        axis.grid(alpha=0.3)
    figure.tight_layout()
    plt.show(block=True)  # 关闭曲线窗口后，才创建学习率窗口。

    figure, axis = plt.subplots(figsize=(10, 4))
    figure.canvas.manager.set_window_title("2/2 - Learning Rate Used at Each Update")
    for strategy in strategies:
        axis.plot(histories[strategy]["lr"], label=strategy)
    axis.set(title="Learning rate schedule", xlabel="Update (starts at 0)", ylabel="Learning rate")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    plt.show(block=True)
