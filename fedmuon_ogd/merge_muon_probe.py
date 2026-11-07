"""One-shot check: does Muon change MNIST model merging?

Four clients train from the same initialization on a Dirichlet split.
Each optimizer is then merged by uniform averaging and by RegMean.
This is a small diagnostic, not a paper result. FedAvg is already an average;
this script asks whether Muon's local trajectory makes that average, or RegMean,
more accurate.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN_IMG = Path("/tmp/mnist/train-images-idx3-ubyte")
TRAIN_LBL = Path("/tmp/mnist/train-labels-idx1-ubyte")
TEST_IMG = Path("/workspace/MNIST Merge/work/data/MNIST/raw/t10k-images-idx3-ubyte")
TEST_LBL = Path("/workspace/MNIST Merge/work/data/MNIST/raw/t10k-labels-idx1-ubyte")
OUT = Path("/workspace/fedmuon_ogd/hypothesis_logs/merge_muon_probe.json")


def load_images(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    _, n, rows, cols = struct.unpack(">IIII", raw[:16])
    images = torch.frombuffer(bytearray(raw[16:]), dtype=torch.uint8).clone()
    return images.view(n, rows * cols).float().div_(255.0)


def load_labels(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    _, n = struct.unpack(">II", raw[:8])
    labels = torch.frombuffer(bytearray(raw[8:]), dtype=torch.uint8).clone()
    return labels.long().view(n)


def newton_schulz(grad: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.float()
    norm = torch.linalg.norm(x)
    if float(norm) < eps:
        return torch.zeros_like(x)
    x = x / norm
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.T
    return x


class SmallMLP(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))

    def linears(self) -> list[nn.Linear]:
        return [self.fc1, self.fc2]


def dirichlet_split(labels: torch.Tensor, num_clients: int, alpha: float, seed: int) -> list[list[int]]:
    generator = torch.Generator().manual_seed(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for label in range(10):
        label_indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        label_indices = label_indices[torch.randperm(label_indices.numel(), generator=generator)]
        proportions = torch.distributions.Dirichlet(torch.full((num_clients,), alpha)).sample()
        counts = torch.floor(proportions * label_indices.numel()).to(torch.long)
        remainder = int(label_indices.numel() - counts.sum().item())
        if remainder > 0:
            for client_id in torch.argsort(proportions, descending=True)[:remainder].tolist():
                counts[client_id] += 1
        offset = 0
        for client_id, count in enumerate(counts.tolist()):
            if count <= 0:
                continue
            client_indices[client_id].extend(label_indices[offset : offset + count].tolist())
            offset += count
    return client_indices


def batches(n: int, batch_size: int, seed: int, epoch: int):
    generator = torch.Generator().manual_seed(seed + epoch * 1009)
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1024) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            logits = model(x[start : start + batch_size])
            correct += int((logits.argmax(dim=1) == y[start : start + batch_size]).sum())
    return correct / y.size(0)


def train_adam(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float, batch_size: int, seed: int) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for idx in batches(x.size(0), batch_size, seed, epoch):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            opt.step()


def train_muon(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    momentum: float = 0.95,
) -> None:
    matrix_params = [p for p in model.parameters() if p.ndim == 2]
    vector_params = [p for p in model.parameters() if p.ndim != 2]
    opt = torch.optim.Adam(vector_params, lr=1e-3)
    buffers = {id(p): torch.zeros_like(p) for p in matrix_params}
    model.train()
    for epoch in range(epochs):
        for idx in batches(x.size(0), batch_size, seed, epoch):
            opt.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            with torch.no_grad():
                for p in matrix_params:
                    grad = p.grad
                    if grad is None:
                        continue
                    buf = buffers[id(p)]
                    buf.lerp_(grad, 1.0 - momentum)
                    update = grad.lerp(buf, momentum)
                    p.add_(newton_schulz(update), alpha=-lr)
            opt.step()


def average_models(models: list[SmallMLP]) -> SmallMLP:
    merged = SmallMLP()
    state = merged.state_dict()
    sources = [model.state_dict() for model in models]
    for key in state:
        state[key] = torch.stack([source[key].float() for source in sources]).mean(dim=0)
    merged.load_state_dict(state)
    return merged


def regmean_stats(model: SmallMLP, x: torch.Tensor, batch_size: int = 512) -> list[tuple[torch.Tensor, torch.Tensor]]:
    model.eval()
    layers = model.linears()
    hh = [torch.zeros(layer.in_features + 1, layer.in_features + 1) for layer in layers]
    zh = [torch.zeros(layer.out_features, layer.in_features + 1) for layer in layers]
    total = 0
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            features = x[start : start + batch_size]
            batch = features.size(0)
            total += batch
            for idx, layer in enumerate(layers):
                aug = torch.cat([features, torch.ones(batch, 1)], dim=1)
                pre = layer(features)
                hh[idx] += aug.T @ aug
                zh[idx] += pre.T @ aug
                features = F.relu(pre) if idx == 0 else pre
    return [(layer_hh / total, layer_zh / total) for layer_hh, layer_zh in zip(hh, zh)]


def regmean_merge(models: list[SmallMLP], stats: list[list[tuple[torch.Tensor, torch.Tensor]]], ridge: float) -> SmallMLP:
    merged = SmallMLP()
    weight = 1.0 / len(models)
    for layer_idx, layer in enumerate(merged.linears()):
        hh_sum = None
        zh_sum = None
        for client_stats in stats:
            hh, zh = client_stats[layer_idx]
            hh_sum = weight * hh if hh_sum is None else hh_sum + weight * hh
            zh_sum = weight * zh if zh_sum is None else zh_sum + weight * zh
        solved = zh_sum @ torch.linalg.inv(hh_sum + ridge * torch.eye(hh_sum.size(0)))
        layer.weight.data.copy_(solved[:, :-1])
        layer.bias.data.copy_(solved[:, -1])
    return merged


def mean_pairwise_distance(models: list[SmallMLP]) -> float:
    vectors = [torch.nn.utils.parameters_to_vector(model.parameters()).detach() for model in models]
    total = 0.0
    count = 0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            total += float(torch.linalg.norm(vectors[i] - vectors[j]))
            count += 1
    return total / count


def run_optimizer(name: str, init: dict, train_x, train_y, test_x, test_y, splits, epochs, batch_size, seed, lr) -> dict:
    clients = []
    for client_id, indices in enumerate(splits):
        model = SmallMLP()
        model.load_state_dict(init)
        xx, yy = train_x[indices], train_y[indices]
        if name == "adam":
            train_adam(model, xx, yy, epochs, lr, batch_size, seed + client_id)
        else:
            train_muon(model, xx, yy, epochs, lr, batch_size, seed + client_id)
        clients.append(model)
    client_acc = [accuracy(model, test_x, test_y) for model in clients]
    averaged = average_models(clients)
    stats = [regmean_stats(model, train_x[indices]) for model, indices in zip(clients, splits)]
    merged = regmean_merge(clients, stats, ridge=1e-3)
    centralized = SmallMLP()
    centralized.load_state_dict(init)
    if name == "adam":
        train_adam(centralized, train_x, train_y, epochs, lr, batch_size, seed)
    else:
        train_muon(centralized, train_x, train_y, epochs, lr, batch_size, seed)
    return {
        "optimizer": name,
        "lr": lr,
        "client_test_acc_mean": sum(client_acc) / len(client_acc),
        "client_test_acc": client_acc,
        "simple_average_test_acc": accuracy(averaged, test_x, test_y),
        "regmean_test_acc": accuracy(merged, test_x, test_y),
        "centralized_test_acc": accuracy(centralized, test_x, test_y),
        "mean_pairwise_parameter_distance": mean_pairwise_distance(clients),
    }


def main() -> None:
    train_x, train_y = load_images(TRAIN_IMG), load_labels(TRAIN_LBL)
    test_x, test_y = load_images(TEST_IMG), load_labels(TEST_LBL)
    epochs = 2
    batch_size = 128
    num_clients = 4
    hidden = 64
    settings = []
    for alpha in (0.1, 1.0):
        for seed in (0, 1):
            torch.manual_seed(seed)
            init_model = SmallMLP(hidden)
            init = {key: value.detach().clone() for key, value in init_model.state_dict().items()}
            splits = dirichlet_split(train_y, num_clients, alpha, seed)
            for name, lr in (("adam", 1e-3), ("muon", 0.02), ("muon", 0.05)):
                row = run_optimizer(name, init, train_x, train_y, test_x, test_y, splits, epochs, batch_size, seed, lr)
                row.update(
                    {
                        "alpha": alpha,
                        "seed": seed,
                        "epochs": epochs,
                        "num_clients": num_clients,
                        "hidden": hidden,
                        "client_sizes": [len(indices) for indices in splits],
                        "regmean_data": "each client's own training set",
                        "muon": "Nesterov momentum 0.95, 5 Newton-Schulz steps, Adam lr 0.001 on bias",
                    }
                )
                settings.append(row)
                print(
                    f"alpha={alpha} seed={seed} {name} lr={lr} "
                    f"client={row['client_test_acc_mean']:.4f} "
                    f"avg={row['simple_average_test_acc']:.4f} "
                    f"regmean={row['regmean_test_acc']:.4f} "
                    f"central={row['centralized_test_acc']:.4f} "
                    f"dist={row['mean_pairwise_parameter_distance']:.3f}",
                    flush=True,
                )
    payload = {
        "claim": "small diagnostic only; not a paper result",
        "question": "Does local Muon make one-shot uniform averaging or RegMean more accurate than local Adam?",
        "runs": settings,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
