"""A small, reproducible FedAvg baseline for the paper's imbalanced MNIST task."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import struct
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


class Images(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.from_numpy(images.copy())
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=True))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index].unsqueeze(0).float().div(255), self.labels[index]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.layers(x)


class PaperCNN(nn.Module):
    """Two tanh convolution layers (32, 16 channels; 3x3 kernels), as in the paper."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.Tanh(),
            nn.Conv2d(32, 16, 3), nn.Tanh(),
            nn.Flatten(), nn.Linear(16 * 24 * 24, 10),
        )

    def forward(self, x):
        return self.layers(x)


def make_model(name: str) -> nn.Module:
    return MLP() if name == "mlp" else PaperCNN()


def download_if_missing(folder: Path, filename: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / filename
    if target.exists():
        return target
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        print(f"Downloading {filename} ...", flush=True)
        with urllib.request.urlopen(MNIST_BASE + filename, timeout=60) as source, temporary.open("wb") as dest:
            while chunk := source.read(1024 * 1024):
                dest.write(chunk)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_idx(path: Path, magic: int) -> np.ndarray:
    with gzip.open(path, "rb") as source:
        header = source.read(16 if magic == 2051 else 8)
        if len(header) < 8:
            raise ValueError(f"Incomplete MNIST file: {path}")
        if struct.unpack(">I", header[:4])[0] != magic:
            raise ValueError(f"Invalid MNIST IDX header: {path}")
        if magic == 2051:
            _, count, height, width = struct.unpack(">IIII", header)
            shape = (count, height, width)
        else:
            _, count = struct.unpack(">II", header)
            shape = (count,)
        raw = source.read()
    data = np.frombuffer(raw, dtype=np.uint8)
    if data.size != math.prod(shape):
        raise ValueError(f"MNIST data size does not match header: {path}")
    return data.reshape(shape)


def load_mnist(folder: Path):
    files = {key: download_if_missing(folder, name) for key, name in MNIST_FILES.items()}
    train_x = read_idx(files["train_images"], 2051)
    train_y = read_idx(files["train_labels"], 2049)
    test_x = read_idx(files["test_images"], 2051)
    test_y = read_idx(files["test_labels"], 2049)
    if len(train_x) != len(train_y) or len(test_x) != len(test_y):
        raise ValueError("MNIST images and labels have different lengths")
    return train_x, train_y, test_x, test_y


def downsample_eight(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    other = np.flatnonzero(labels != 8)
    eights = np.flatnonzero(labels == 8)
    keep_eights = round(len(other) * fraction / (1 - fraction))
    if not 0 < keep_eights <= len(eights):
        raise ValueError("Requested digit-8 fraction cannot be made from this dataset")
    selected = np.concatenate((other, rng.choice(eights, keep_eights, replace=False)))
    rng.shuffle(selected)
    return selected


def partition_clients(
    labels: np.ndarray, selected: np.ndarray, clients: int, mode: str, eight_clients: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    partitions: list[list[np.ndarray]] = [[] for _ in range(clients)]
    for digit in range(10):
        digit_indices = selected[labels[selected] == digit].copy()
        rng.shuffle(digit_indices)
        receivers = range(eight_clients) if digit == 8 and mode == "concentrated8" else range(clients)
        for client, piece in zip(receivers, np.array_split(digit_indices, len(receivers))):
            partitions[client].append(piece)
    result = []
    for pieces in partitions:
        indices = np.concatenate(pieces)
        rng.shuffle(indices)
        result.append(indices)
    joined = np.concatenate(result)
    if len(joined) != len(selected) or len(np.unique(joined)) != len(selected):
        raise RuntimeError("Client partition lost or duplicated training samples")
    return result


def evaluate(model: nn.Module, dataset: Dataset, batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    count = np.zeros(10, dtype=np.int64)
    correct = np.zeros(10, dtype=np.int64)
    loss_sum = np.zeros(10, dtype=np.float64)
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            losses = nn.functional.cross_entropy(logits, labels, reduction="none")
            predictions = logits.argmax(dim=1)
            for digit in range(10):
                mask = labels == digit
                count[digit] += int(mask.sum())
                correct[digit] += int((predictions[mask] == labels[mask]).sum())
                loss_sum[digit] += float(losses[mask].sum())
    result = {
        "accuracy_all": 100 * correct.sum() / count.sum(),
        "loss_all": loss_sum.sum() / count.sum(),
    }
    for digit in (2, 8):
        result[f"accuracy_{digit}"] = 100 * correct[digit] / count[digit]
        result[f"loss_{digit}"] = loss_sum[digit] / count[digit]
    return {key: float(value) for key, value in result.items()}


def local_train(
    model_name: str,
    global_state: OrderedDict,
    dataset: Dataset,
    indices: np.ndarray,
    batch_size: int,
    learning_rate: float,
    local_epochs: int,
    max_local_batches: int | None,
    seed: int,
    device: torch.device,
) -> OrderedDict:
    model = make_model(model_name).to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=True,
        generator=generator, pin_memory=device.type == "cuda",
    )
    for _ in range(local_epochs):
        for batch_number, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            if max_local_batches is not None and batch_number + 1 >= max_local_batches:
                break
    return OrderedDict((name, value.detach().clone()) for name, value in model.state_dict().items())


def weighted_average(states: list[OrderedDict], sizes: list[int]) -> OrderedDict:
    if not states or len(states) != len(sizes) or any(size <= 0 for size in sizes):
        raise ValueError("FedAvg needs a nonempty state and a positive sample count per client")
    total = sum(sizes)
    return OrderedDict(
        (name, sum((state[name] * (size / total) for state, size in zip(states, sizes))))
        for name in states[0]
    )


def plot_class_accuracy(metrics_path: Path, image_path: Path) -> None:
    """Plot the test accuracy of true digit-2 and digit-8 examples by round."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    with metrics_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    rounds = [int(row["round"]) for row in rows]
    acc_2 = [float(row["accuracy_2"]) for row in rows]
    acc_8 = [float(row["accuracy_8"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(rounds, acc_2, color="#2563eb", linewidth=2.3, marker="o", markersize=4, label="Digit 2")
    ax.plot(rounds, acc_8, color="#dc2626", linewidth=2.3, marker="o", markersize=4, label="Digit 8")
    ax.set(title="FedAvg: test accuracy by digit", xlabel="Federated round", ylabel="Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0, right=max(1, rounds[-1]))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.text(0.5, 0.015, "Round 0 is the untrained model.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(image_path, dpi=160)
    plt.close(fig)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=10, help="Total simulated clients")
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--digit-eight-fraction", type=float, default=0.01)
    parser.add_argument("--partition", choices=["iid", "concentrated8"], default="iid")
    parser.add_argument("--eight-clients", type=int, default=2, help="Clients receiving digit 8 in concentrated8 mode")
    parser.add_argument("--model", choices=["mlp", "paper-cnn"], default="mlp")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-local-batches", type=int, default=None, help="Smoke test only; limits each local epoch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "runs" / "baseline")
    args = parser.parse_args()
    if args.clients < 1 or not 1 <= args.clients_per_round <= args.clients:
        parser.error("clients-per-round must be between 1 and clients")
    if args.rounds < 1 or args.local_epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("rounds, local-epochs, batch-size and learning-rate must be positive")
    if not 0 < args.digit_eight_fraction < 1:
        parser.error("digit-eight-fraction must be between 0 and 1")
    if args.partition == "concentrated8" and not 1 <= args.eight_clients <= args.clients:
        parser.error("eight-clients must be between 1 and clients")
    if args.max_local_batches is not None and args.max_local_batches < 1:
        parser.error("max-local-batches must be positive")
    return args


def main() -> None:
    args = arguments()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment; use the project's .venv Python")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    train_x, train_y, test_x, test_y = load_mnist(args.data_dir)
    selected = downsample_eight(train_y, args.digit_eight_fraction, args.seed)
    partitions = partition_clients(train_y, selected, args.clients, args.partition, args.eight_clients, args.seed + 1)
    train_dataset = Images(train_x, train_y)
    test_dataset = Images(test_x, test_y)
    distribution = []
    for client_id, indices in enumerate(partitions):
        counts = np.bincount(train_y[indices], minlength=10)
        distribution.append({"client": client_id, "samples": len(indices), **{f"digit_{i}": int(v) for i, v in enumerate(counts)}})
    with (args.output_dir / "clients.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=distribution[0].keys())
        writer.writeheader()
        writer.writerows(distribution)

    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings["train_samples"] = len(selected)
    settings["train_digit_8"] = int((train_y[selected] == 8).sum())
    settings["actual_digit_eight_fraction"] = settings["train_digit_8"] / len(selected)
    settings["test_samples"] = len(test_y)
    settings["note"] = "No differential privacy is applied in this FedAvg baseline."
    settings["selected_device"] = str(device)
    if device.type == "cuda":
        settings["gpu_name"] = torch.cuda.get_device_name(0)
    (args.output_dir / "config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

    model = make_model(args.model).to(device)
    initial = evaluate(model, test_dataset, 512, device)
    metric_fields = ["round", "selected_clients", "accuracy_all", "loss_all", "accuracy_2", "accuracy_8", "loss_2", "loss_8"]
    rng = np.random.default_rng(args.seed + 2)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerow({"round": 0, "selected_clients": "", **initial})
        output.flush()
        print(f"Train: {len(selected)} samples; digit 8: {settings['train_digit_8']} ({settings['actual_digit_eight_fraction']:.2%})", flush=True)
        print(f"Round 0: overall={initial['accuracy_all']:.2f}%, digit2={initial['accuracy_2']:.2f}%, digit8={initial['accuracy_8']:.2f}%", flush=True)
        for round_number in range(1, args.rounds + 1):
            chosen = sorted(rng.choice(args.clients, args.clients_per_round, replace=False).tolist())
            starting_state = OrderedDict((name, value.detach().clone()) for name, value in model.state_dict().items())
            local_states = []
            sizes = []
            for client_id in chosen:
                local_states.append(local_train(
                    args.model, starting_state, train_dataset, partitions[client_id], args.batch_size,
                    args.learning_rate, args.local_epochs, args.max_local_batches,
                    args.seed + round_number * args.clients + client_id, device,
                ))
                sizes.append(len(partitions[client_id]))
            model.load_state_dict(weighted_average(local_states, sizes))
            scores = evaluate(model, test_dataset, 512, device)
            writer.writerow({"round": round_number, "selected_clients": ",".join(map(str, chosen)), **scores})
            output.flush()
            print(f"Round {round_number}: overall={scores['accuracy_all']:.2f}%, digit2={scores['accuracy_2']:.2f}%, digit8={scores['accuracy_8']:.2f}%", flush=True)
    torch.save(OrderedDict((name, value.detach().cpu()) for name, value in model.state_dict().items()),
               args.output_dir / "global_model.pt")
    plot_class_accuracy(args.output_dir / "metrics.csv", args.output_dir / "accuracy_2_vs_8.png")
    print(f"Saved results to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
