# FedAvg 基线：MNIST 少数类实验

本目录先实现**不加差分隐私**的 FedAvg，用于后续与 DP-FedAvg、Global-Adapt 思路比较。一个客户端是一份固定的本地数据；程序在一台电脑上顺序模拟 10 个客户端，不需要 10 台机器。测试集始终保持 MNIST 原样。

## 环境和运行

需要 Python、PyTorch、NumPy 和 Matplotlib（见 `requirements.txt`）。程序第一次运行时会下载 MNIST 原始压缩文件到 `data/`；不依赖 torchvision。

本机有 NVIDIA RTX 5060 Laptop GPU。项目内的 `.venv/` 安装了 CUDA 版 PyTorch，运行时默认自动选择 GPU；使用 `run.cmd` 可确保进入这个独立环境，不会误用全局的 CPU 版 PyTorch。启动时会打印 `Device: cuda (...)`，本次运行的 `config.json` 也会记录 `selected_device`。如需强制只用 CPU，可加 `--device cpu`。

如果以后清理了 `.venv/`，可在本机重新建立环境：

```powershell
python -m venv .venv --system-site-packages
.\.venv\Scripts\python.exe -m pip install numpy matplotlib
.\.venv\Scripts\python.exe -m pip install "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130
```

CUDA 版 PyTorch 安装包约 2 GB；这里的安装命令针对当前 Windows、Python 3.13 和 NVIDIA GPU 环境。其他电脑应参考 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选择对应版本。

在本目录的 PowerShell 中执行：

```powershell
.\run.cmd --rounds 1 --output-dir runs/first_round
```

这会做 **10 个客户端全部参与、每个客户端完整训练 1 个本地 epoch** 的一轮 FedAvg。默认模型是较轻量的 MLP，方便在 CPU 上检查流程；它的绝对准确率不能直接与论文使用 CNN 得到的表格数值比较。

正式基线可先运行 20 轮观察，再扩展到 200 轮：

```powershell
.\run.cmd --rounds 20 --output-dir runs/iid_20
.\run.cmd --rounds 200 --output-dir runs/iid_200
```

若想使用与论文相同的两层卷积结构，在命令末尾加 `--model paper-cnn`。CPU 训练会明显更慢。

## 客户端和数据

- 训练集只对数字 8 做下采样，使它约占训练样本的 1%；其余数字不删减。
- 默认 `--partition iid`：各数字尽量平均分到 10 个客户端。每个客户端有约 5400 个样本，其中约 55 个数字 8。
- 第二阶段可用 `--partition concentrated8 --eight-clients 2`，在保持数字 8 总数不变的情况下，把它集中到 2 个客户端。
- 默认每轮 10 个客户端全部参与。`--clients` 和 `--clients-per-round` 可分别修改总数与参与数。
- 每个客户端从同一全局模型出发，本地使用 SGD（学习率 0.01）、批量大小 32、训练 1 个 epoch；服务器按客户端样本数加权平均模型。

例如分布不均的对照实验：

```powershell
.\run.cmd --rounds 20 --partition concentrated8 --eight-clients 2 --output-dir runs/concentrated8_20
```

## 输出怎么看

- `clients.csv`：每个客户端的数据量和 0–9 各数字的数量，先检查它以确认分组正确。
- `metrics.csv`：第 0 轮训练前及每轮后的测试集总体准确率/Loss、数字 2 准确率/Loss、数字 8 准确率/Loss。
- `accuracy_2_vs_8.png`：每次训练完成后自动生成的折线图。横轴是联邦轮数，纵轴是测试集准确率；蓝线为数字 2，红线为数字 8。第 0 轮是尚未训练的随机模型，判断训练效果时主要看后续轮次。
- `config.json`：本次运行的设置和数字 8 实际比例。
- `global_model.pt`：最后一轮的全局模型参数。

这份基线**没有 DP 保证**。将来加入 DP 时，应先确定保护单位是单个样本还是整个客户端；两者的裁剪、加噪和隐私计算方式不同。若保护整个客户端，10/10 客户端每轮全部参加也没有客户端抽样带来的隐私放大效应。

参考：[FedAvg 原论文](https://proceedings.mlr.press/v54/mcmahan17a.html)；[所研究的 DP 公平性论文](https://arxiv.org/html/2206.07737)。
