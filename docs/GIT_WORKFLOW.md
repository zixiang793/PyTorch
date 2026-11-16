# 学习项目的 Git 工作流

这里以本次学习率对比为例。`code/` 记录实现，`experiments/` 记录实测，Git 历史记录每一步为什么发生。

## 开始一项改进

在仓库根目录运行，先查看工作区，提交或妥善保留自己已有的修改，再开分支：

```powershell
git status
git switch main
git pull --ff-only
git switch -c feat/my-next-experiment
```

## 提交代码与结果

改完后先运行实验，阅读差异。明确指定要提交的文件，避免把下载数据、日志或模型权重误加入：

```powershell
git diff
git add code/MLP_lr_compare.py
git diff --cached
git commit -m "feat: compare MLP learning rate schedules"

# 接着整理 experiments/ 中的配置、CSV 和结论，再单独提交结果
git add README.md code/README.md experiments/lr_schedulers docs/GIT_WORKFLOW.md
git diff --cached
git commit -m "docs: record controlled learning rate experiment"
git push -u origin feat/my-next-experiment
```

示例文件列表对应本次实验；下次应换成自己实际修改的文件。实验结论以对应配置和 CSV 为依据，不只记录“效果变好了”。运行产物默认留在被忽略的 `code/outputs/` 与 `code/runs/`。

## 合并与回看

可以在 GitHub 从功能分支创建 PR，检查差异后合并。个人项目也可以本地快进合并，保留两次提交的历史：

```powershell
git switch main
git merge --ff-only feat/my-next-experiment
git push origin main
git log --oneline --graph -8
```

若快进失败，先查看两边的新提交再处理，不用强制推送覆盖历史。每次完成后用 `git status` 确认工作区，用 GitHub 的提交记录检查代码与结果是否都已经上传。

仓库通过 SSH 连接：`git@github.com:zixiang793/PyTorch.git`。
