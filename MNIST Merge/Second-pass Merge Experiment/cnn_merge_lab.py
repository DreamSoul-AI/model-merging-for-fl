#!/usr/bin/env python3
"""
CNN-based merge lab for CIFAR experiments.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(width * 4, width * 2),
            nn.ReLU(),
            nn.Linear(width * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        return self.classifier(feats)

    def linear_layers(self) -> List[nn.Linear]:
        return [module for module in self.classifier if isinstance(module, nn.Linear)]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return feats.view(feats.size(0), -1)


@dataclass
class CNNExperimentConfig:
    data_root: Path
    dataset_name: str = "cifar10"
    device: str = "cpu"
    seed: int = 7
    num_clients: int = 10
    train_batch_size: int = 128
    eval_batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0
    width: int = 64
    split_mode: str = "iid"
    dirichlet_alpha: float = 0.5
    calibration_size: int = 256
    regmean_lambda: float = 1e-3
    client_epochs: Tuple[int, ...] = (5, 5, 5, 5, 5, 5, 5, 5, 5, 5)
    centralized_epochs: int = 5
    feature_skew_mode: str = "none"


@dataclass
class ClientSpec:
    client_id: int
    indices: List[int]
    epochs: int


@dataclass
class ClientResult:
    spec: ClientSpec
    model: SmallCNN
    train_metrics: Dict[str, float]


class TransformSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int], transform=None) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, target = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def get_dataset_and_transforms(config: CNNExperimentConfig) -> Tuple[Dataset, Dataset, int, transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if config.dataset_name == "cifar100":
        dataset_cls = datasets.CIFAR100
        num_classes = 100
    else:
        dataset_cls = datasets.CIFAR10
        num_classes = 10
    raw_train = dataset_cls(root=config.data_root, train=True, download=True, transform=None)
    raw_test = dataset_cls(root=config.data_root, train=False, download=True, transform=None)
    return raw_train, raw_test, num_classes, train_transform, eval_transform


def dataset_targets(dataset: Dataset) -> torch.Tensor:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("Dataset does not expose targets.")
    return torch.as_tensor(targets, dtype=torch.long)


def iid_split(dataset: Dataset, num_clients: int, seed: int) -> List[List[int]]:
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(len(dataset), generator=generator).tolist()
    chunk_size = math.ceil(len(shuffled) / num_clients)
    return [shuffled[i * chunk_size : (i + 1) * chunk_size] for i in range(num_clients)]


def dirichlet_split(dataset: Dataset, num_clients: int, alpha: float, seed: int, num_classes: int) -> List[List[int]]:
    labels = dataset_targets(dataset)
    generator = torch.Generator().manual_seed(seed)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for label in range(num_classes):
        label_indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        label_indices = label_indices[torch.randperm(label_indices.numel(), generator=generator)]
        proportions = torch.distributions.Dirichlet(torch.full((num_clients,), alpha)).sample()
        counts = torch.floor(proportions * label_indices.numel()).to(torch.long)
        remainder = label_indices.numel() - counts.sum().item()
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


def make_client_specs(config: CNNExperimentConfig, train_dataset: Dataset, num_classes: int) -> List[ClientSpec]:
    if config.split_mode == "iid":
        splits = iid_split(train_dataset, config.num_clients, config.seed)
    elif config.split_mode == "dirichlet":
        splits = dirichlet_split(train_dataset, config.num_clients, config.dirichlet_alpha, config.seed, num_classes)
    else:
        raise ValueError(f"Unsupported split mode: {config.split_mode}")
    return [ClientSpec(client_id=i, indices=list(indices), epochs=config.client_epochs[i]) for i, indices in enumerate(splits)]


def feature_skew_transform(client_id: int, mode: str):
    if mode == "none":
        return None
    transforms_list = [transforms.ToTensor()]
    if mode == "rotation":
        angles = [-25, -10, 0, 10, 25]
        angle = angles[client_id % len(angles)]
        transforms_list.insert(0, transforms.RandomRotation((angle, angle)))
    elif mode == "color":
        factors = [0.6, 0.8, 1.0, 1.2, 1.4]
        factor = factors[client_id % len(factors)]
        transforms_list.insert(0, transforms.ColorJitter(brightness=(factor, factor)))
    return transforms.Compose(transforms_list)


def make_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    transform=None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    subset = TransformSubset(dataset, indices, transform=transform)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def clone_model(model: SmallCNN, num_classes: int, width: int) -> SmallCNN:
    cloned = SmallCNN(num_classes=num_classes, width=width)
    cloned.load_state_dict(copy.deepcopy(model.state_dict()))
    return cloned


def train_model(model: SmallCNN, loader: DataLoader, epochs: int, device: torch.device, lr: float, weight_decay: float) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)
    model.train()
    for _ in range(epochs):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate_model(model: SmallCNN, loader: DataLoader, device: torch.device, num_classes: int) -> Dict[str, float]:
    model.to(device)
    model.eval()
    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    per_class_total = torch.zeros(num_classes, dtype=torch.long)
    per_class_correct = torch.zeros(num_classes, dtype=torch.long)
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        predictions = logits.argmax(dim=1)
        total_examples += targets.size(0)
        total_loss += loss.item()
        total_correct += (predictions == targets).sum().item()
        for label in range(num_classes):
            mask = targets == label
            if mask.any():
                per_class_total[label] += mask.sum().item()
                per_class_correct[label] += (predictions[mask] == label).sum().item()
    per_class_acc = {
        f"class_{label}_acc": float(per_class_correct[label].item()) / max(1, int(per_class_total[label].item()))
        for label in range(num_classes)
    }
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "macro_accuracy": sum(per_class_acc.values()) / num_classes,
        "worst_class_accuracy": min(per_class_acc.values()),
        **per_class_acc,
    }


def summarize_class_shares(dataset: Dataset, indices: Sequence[int], num_classes: int) -> Dict[str, float]:
    labels = dataset_targets(dataset)[list(indices)]
    counts = torch.bincount(labels, minlength=num_classes).float()
    shares = counts / counts.sum().clamp_min(1.0)
    return {f"class_{i}_share": float(shares[i].item()) for i in range(num_classes)}


def train_clients(
    config: CNNExperimentConfig,
    train_dataset: Dataset,
    num_classes: int,
    train_transform,
    eval_transform,
) -> Tuple[List[ClientResult], SmallCNN]:
    specs = make_client_specs(config, train_dataset, num_classes)
    base_model = SmallCNN(num_classes=num_classes, width=config.width)
    device = torch.device(config.device)
    results: List[ClientResult] = []
    for spec in specs:
        model = clone_model(base_model, num_classes, config.width)
        client_transform = train_transform
        if config.feature_skew_mode != "none":
            skew = feature_skew_transform(spec.client_id, config.feature_skew_mode)
            if skew is not None:
                client_transform = transforms.Compose(list(skew.transforms) + list(train_transform.transforms[1:]))
        train_loader = make_loader(
            train_dataset,
            spec.indices,
            config.train_batch_size,
            True,
            config.seed + spec.client_id,
            transform=client_transform,
        )
        train_model(model, train_loader, spec.epochs, device, config.lr, config.weight_decay)
        eval_loader = make_loader(
            train_dataset,
            spec.indices,
            config.eval_batch_size,
            False,
            config.seed + spec.client_id,
            transform=eval_transform,
        )
        metrics = evaluate_model(model, eval_loader, device, num_classes)
        metrics.update(summarize_class_shares(train_dataset, spec.indices, num_classes))
        results.append(ClientResult(spec=spec, model=model, train_metrics=metrics))
    return results, base_model


def average_state_dicts(state_dicts: Sequence[Dict[str, torch.Tensor]], weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    normalized = torch.tensor(weights, dtype=torch.float32)
    normalized = normalized / normalized.sum()
    merged: Dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        merged[key] = sum(weight * state[key] for weight, state in zip(normalized.tolist(), state_dicts))
    return merged


def simple_average(models: Sequence[SmallCNN], num_classes: int, width: int) -> SmallCNN:
    merged = clone_model(models[0], num_classes, width)
    merged.load_state_dict(average_state_dicts([m.state_dict() for m in models], [1.0] * len(models)))
    return merged


def weighted_average(models: Sequence[SmallCNN], weights: Sequence[float], num_classes: int, width: int) -> SmallCNN:
    merged = clone_model(models[0], num_classes, width)
    merged.load_state_dict(average_state_dicts([m.state_dict() for m in models], weights))
    return merged


@torch.no_grad()
def collect_regmean_stats(model: SmallCNN, loader: DataLoader, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    model.to(device)
    model.eval()
    layers = model.linear_layers()
    hh = [torch.zeros(layer.in_features + 1, layer.in_features + 1, device=device) for layer in layers]
    zh = [torch.zeros(layer.out_features, layer.in_features + 1, device=device) for layer in layers]
    total = 0
    for inputs, _ in loader:
        inputs = inputs.to(device)
        features = model.extract_features(inputs)
        current = features
        linear_index = 0
        for module in model.classifier:
            if isinstance(module, nn.Linear):
                augmented = torch.cat([current, torch.ones(current.size(0), 1, device=current.device)], dim=1)
                z = module(current)
                hh[linear_index] += augmented.t() @ augmented
                zh[linear_index] += z.t() @ augmented
                current = z
                linear_index += 1
            elif isinstance(module, nn.ReLU):
                current = module(current)
        total += inputs.size(0)
    return [(layer_hh / total, layer_zh / total) for layer_hh, layer_zh in zip(hh, zh)]


def regmean_merge(
    models: Sequence[SmallCNN],
    stats_per_client: Sequence[List[Tuple[torch.Tensor, torch.Tensor]]],
    weights: Sequence[float],
    ridge_lambda: float,
    num_classes: int,
    width: int,
) -> SmallCNN:
    merged = clone_model(models[0], num_classes, width)
    normalized = torch.tensor(weights, dtype=torch.float32)
    normalized = normalized / normalized.sum()
    merged_layers = merged.linear_layers()
    for layer_idx, layer in enumerate(merged_layers):
        hh_sum = None
        zh_sum = None
        for client_idx, stats in enumerate(stats_per_client):
            hh, zh = stats[layer_idx]
            weighted_hh = normalized[client_idx].item() * hh
            weighted_zh = normalized[client_idx].item() * zh
            hh_sum = weighted_hh if hh_sum is None else hh_sum + weighted_hh
            zh_sum = weighted_zh if zh_sum is None else zh_sum + weighted_zh
        solved = zh_sum @ torch.linalg.inv(hh_sum + ridge_lambda * torch.eye(hh_sum.size(0), device=hh_sum.device))
        layer.weight.data.copy_(solved[:, :-1].to(layer.weight.data.device))
        layer.bias.data.copy_(solved[:, -1].to(layer.bias.data.device))
    return merged


def make_calibration_loader(
    dataset: Dataset,
    indices: Sequence[int],
    calibration_size: int,
    batch_size: int,
    seed: int,
    transform,
) -> DataLoader:
    chosen = list(indices)
    if len(chosen) > calibration_size:
        generator = torch.Generator().manual_seed(seed)
        positions = torch.randperm(len(chosen), generator=generator)[:calibration_size].tolist()
        chosen = [chosen[pos] for pos in positions]
    return make_loader(dataset, chosen, batch_size, False, seed, transform=transform)


def evaluate_merged_model(
    model: SmallCNN,
    client_results: Sequence[ClientResult],
    train_dataset: Dataset,
    test_dataset: Dataset,
    eval_transform,
    config: CNNExperimentConfig,
    num_classes: int,
) -> Dict[str, object]:
    device = torch.device(config.device)
    test_loader = make_loader(
        test_dataset,
        list(range(len(test_dataset))),
        config.eval_batch_size,
        False,
        config.seed,
        transform=eval_transform,
    )
    global_metrics = evaluate_model(model, test_loader, device, num_classes)
    per_client_local = {}
    for client in client_results:
        loader = make_loader(
            train_dataset,
            client.spec.indices,
            config.eval_batch_size,
            False,
            config.seed + client.spec.client_id,
            transform=eval_transform,
        )
        per_client_local[f"client_{client.spec.client_id}_local_acc"] = evaluate_model(model, loader, device, num_classes)["accuracy"]
    return {"global": global_metrics, "per_client_local_accuracy": per_client_local}


def config_to_jsonable(config: CNNExperimentConfig) -> Dict[str, object]:
    payload = dict(config.__dict__)
    payload["data_root"] = str(payload["data_root"])
    payload["client_epochs"] = list(payload["client_epochs"])
    return payload


def run_cnn_experiment(config: CNNExperimentConfig) -> Dict[str, object]:
    set_seed(config.seed)
    train_dataset, test_dataset, num_classes, train_transform, eval_transform = get_dataset_and_transforms(config)
    client_results, base_model = train_clients(config, train_dataset, num_classes, train_transform, eval_transform)
    client_models = [client.model for client in client_results]
    sample_weights = [len(client.spec.indices) for client in client_results]
    epoch_weights = [client.spec.epochs for client in client_results]
    stats_per_client = [
        collect_regmean_stats(
            client.model,
            make_calibration_loader(
                train_dataset,
                client.spec.indices,
                config.calibration_size,
                config.eval_batch_size,
                config.seed + client.spec.client_id,
                eval_transform,
            ),
            torch.device(config.device),
        )
        for client in client_results
    ]

    centralized = clone_model(base_model, num_classes, config.width)
    centralized_loader = make_loader(
        train_dataset,
        list(range(len(train_dataset))),
        config.train_batch_size,
        True,
        config.seed,
        transform=train_transform,
    )
    train_model(centralized, centralized_loader, config.centralized_epochs, torch.device(config.device), config.lr, config.weight_decay)

    merges = {
        "simple_avg": evaluate_merged_model(simple_average(client_models, num_classes, config.width), client_results, train_dataset, test_dataset, eval_transform, config, num_classes),
        "sample_weighted_avg": evaluate_merged_model(weighted_average(client_models, sample_weights, num_classes, config.width), client_results, train_dataset, test_dataset, eval_transform, config, num_classes),
        "epoch_weighted_avg": evaluate_merged_model(weighted_average(client_models, epoch_weights, num_classes, config.width), client_results, train_dataset, test_dataset, eval_transform, config, num_classes),
        "regmean_equal": evaluate_merged_model(regmean_merge(client_models, stats_per_client, [1.0] * len(client_models), config.regmean_lambda, num_classes, config.width), client_results, train_dataset, test_dataset, eval_transform, config, num_classes),
        "regmean_epoch_weighted": evaluate_merged_model(regmean_merge(client_models, stats_per_client, epoch_weights, config.regmean_lambda, num_classes, config.width), client_results, train_dataset, test_dataset, eval_transform, config, num_classes),
        "centralized": evaluate_merged_model(centralized, client_results, train_dataset, test_dataset, eval_transform, config, num_classes),
    }

    test_loader = make_loader(test_dataset, list(range(len(test_dataset))), config.eval_batch_size, False, config.seed, transform=eval_transform)
    best_single = max(
        [(client.spec.client_id, evaluate_model(client.model, test_loader, torch.device(config.device), num_classes)) for client in client_results],
        key=lambda item: item[1]["accuracy"],
    )
    merges["single_best"] = {"client_id": best_single[0], "metrics": best_single[1]}

    return {
        "config": config_to_jsonable(config),
        "clients": [
            {
                "client_id": client.spec.client_id,
                "epochs": client.spec.epochs,
                "sample_count": len(client.spec.indices),
                "train_metrics": client.train_metrics,
            }
            for client in client_results
        ],
        "merges": merges,
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
