#!/usr/bin/env python3
"""
Core utilities for the second-pass MNIST merge lab.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SmallMLP(nn.Module):
    def __init__(self, hidden_sizes: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        dims = [28 * 28, *hidden_sizes, 10]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.view(x.size(0), -1))

    def linear_layers(self) -> List[nn.Linear]:
        return [layer for layer in self.network if isinstance(layer, nn.Linear)]

    def extract_penultimate_features(self, x: torch.Tensor) -> torch.Tensor:
        features = x.view(x.size(0), -1)
        linear_seen = 0
        total_linear = len(self.linear_layers())
        for module in self.network:
            if isinstance(module, nn.Linear):
                linear_seen += 1
                if linear_seen == total_linear:
                    return features
                features = module(features)
            elif isinstance(module, nn.ReLU):
                features = module(features)
        raise RuntimeError("Failed to extract penultimate features.")

    def collect_hidden_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = x.view(x.size(0), -1)
        hidden_states: List[torch.Tensor] = []
        for module in self.network:
            if isinstance(module, nn.Linear):
                features = module(features)
                hidden_states.append(features)
            elif isinstance(module, nn.ReLU):
                features = module(features)
        return hidden_states


@dataclass
class ClientSpec:
    client_id: int
    indices: List[int]
    epochs: int


@dataclass
class ClientResult:
    spec: ClientSpec
    model: SmallMLP
    train_metrics: Dict[str, float]


@dataclass
class ExperimentConfig:
    data_root: Path
    device: str = "cpu"
    seed: int = 7
    num_clients: int = 10
    train_batch_size: int = 128
    eval_batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden_sizes: Tuple[int, int] = (128, 128)
    split_mode: str = "iid"
    dirichlet_alpha: float = 0.5
    pathological_classes_per_client: int = 2
    calibration_size: int = 256
    regmean_lambda: float = 1e-3
    client_epochs: Tuple[int, ...] = (5, 5, 5, 5, 5, 5, 5, 5, 5, 5)
    centralized_epochs: int = 5
    post_merge_steps: int = 50
    post_merge_lr: float = 5e-4


def config_to_jsonable(config: ExperimentConfig) -> Dict[str, object]:
    payload = dict(config.__dict__)
    payload["data_root"] = str(payload["data_root"])
    payload["hidden_sizes"] = list(payload["hidden_sizes"])
    payload["client_epochs"] = list(payload["client_epochs"])
    return payload


def build_datasets(data_root: Path) -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


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


def dirichlet_split(dataset: Dataset, num_clients: int, alpha: float, seed: int) -> List[List[int]]:
    labels = dataset_targets(dataset)
    generator = torch.Generator().manual_seed(seed)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for label in range(10):
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


def pathological_split(dataset: Dataset, num_clients: int, classes_per_client: int, seed: int) -> List[List[int]]:
    labels = dataset_targets(dataset)
    generator = torch.Generator().manual_seed(seed)
    class_to_indices: Dict[int, List[int]] = {}
    for label in range(10):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)].tolist()
        class_to_indices[label] = shuffled

    splits: List[List[int]] = [[] for _ in range(num_clients)]
    shards_per_class = max(1, math.ceil((num_clients * classes_per_client) / 10))
    class_shards: Dict[int, List[List[int]]] = {}
    for label, indices in class_to_indices.items():
        shard_size = max(1, math.ceil(len(indices) / shards_per_class))
        class_shards[label] = [indices[i * shard_size : (i + 1) * shard_size] for i in range(shards_per_class)]

    class_offsets = {label: 0 for label in range(10)}
    for client_id in range(num_clients):
        chosen_classes = [int((client_id + offset * num_clients) % 10) for offset in range(classes_per_client)]
        for label in chosen_classes:
            offset = class_offsets[label] % len(class_shards[label])
            splits[client_id].extend(class_shards[label][offset])
            class_offsets[label] += 1
    return splits


def make_client_specs(config: ExperimentConfig, train_dataset: Dataset) -> List[ClientSpec]:
    if config.split_mode == "iid":
        splits = iid_split(train_dataset, config.num_clients, config.seed)
    elif config.split_mode == "dirichlet":
        splits = dirichlet_split(train_dataset, config.num_clients, config.dirichlet_alpha, config.seed)
    elif config.split_mode == "pathological":
        splits = pathological_split(
            train_dataset,
            config.num_clients,
            config.pathological_classes_per_client,
            config.seed,
        )
    else:
        raise ValueError(f"Unsupported split_mode: {config.split_mode}")
    return [
        ClientSpec(client_id=i, indices=list(indices), epochs=config.client_epochs[i])
        for i, indices in enumerate(splits)
    ]


def make_loader(dataset: Dataset, indices: Sequence[int], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=shuffle, generator=generator)


def clone_model(base_model: SmallMLP) -> SmallMLP:
    hidden_sizes = [layer.out_features for layer in base_model.linear_layers()[:-1]]
    model = SmallMLP(hidden_sizes=hidden_sizes)
    model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    return model


def train_model(
    model: SmallMLP,
    loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
) -> None:
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
def evaluate_model(model: SmallMLP, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    per_class_total = torch.zeros(10, dtype=torch.long)
    per_class_correct = torch.zeros(10, dtype=torch.long)

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        predictions = logits.argmax(dim=1)

        total_examples += targets.size(0)
        total_loss += loss.item()
        total_correct += (predictions == targets).sum().item()

        for label in range(10):
            mask = targets == label
            if mask.any():
                per_class_total[label] += mask.sum().item()
                per_class_correct[label] += (predictions[mask] == label).sum().item()

    per_class = {
        f"class_{label}_acc": float(per_class_correct[label].item()) / max(1, int(per_class_total[label].item()))
        for label in range(10)
    }
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "macro_accuracy": sum(per_class.values()) / 10.0,
        "worst_class_accuracy": min(per_class.values()),
        **per_class,
    }


def summarize_class_shares(dataset: Dataset, indices: Sequence[int]) -> Dict[str, float]:
    labels = dataset_targets(dataset)[list(indices)]
    counts = torch.bincount(labels, minlength=10).float()
    shares = counts / counts.sum().clamp_min(1.0)
    return {f"class_{i}_share": float(shares[i].item()) for i in range(10)}


def train_clients(config: ExperimentConfig, train_dataset: Dataset, base_model: SmallMLP) -> List[ClientResult]:
    device = torch.device(config.device)
    specs = make_client_specs(config, train_dataset)
    results: List[ClientResult] = []
    for spec in specs:
        loader = make_loader(train_dataset, spec.indices, config.train_batch_size, True, config.seed + spec.client_id)
        model = clone_model(base_model)
        train_model(model, loader, spec.epochs, device, config.lr, config.weight_decay)
        metrics = evaluate_model(model, loader, device)
        metrics.update(summarize_class_shares(train_dataset, spec.indices))
        results.append(ClientResult(spec=spec, model=model, train_metrics=metrics))
    return results


def train_centralized(config: ExperimentConfig, train_dataset: Dataset, base_model: SmallMLP) -> SmallMLP:
    device = torch.device(config.device)
    model = clone_model(base_model)
    loader = make_loader(train_dataset, list(range(len(train_dataset))), config.train_batch_size, True, config.seed)
    train_model(model, loader, config.centralized_epochs, device, config.lr, config.weight_decay)
    return model


def average_state_dicts(state_dicts: Sequence[Dict[str, torch.Tensor]], weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    normalized = torch.tensor(weights, dtype=torch.float32)
    normalized = normalized / normalized.sum()
    merged: Dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        merged[key] = sum(weight * state[key] for weight, state in zip(normalized.tolist(), state_dicts))
    return merged


def simple_average(models: Sequence[SmallMLP]) -> SmallMLP:
    merged = clone_model(models[0])
    merged.load_state_dict(average_state_dicts([model.state_dict() for model in models], [1.0] * len(models)))
    return merged


def weighted_average(models: Sequence[SmallMLP], weights: Sequence[float]) -> SmallMLP:
    merged = clone_model(models[0])
    merged.load_state_dict(average_state_dicts([model.state_dict() for model in models], weights))
    return merged


def linear_forward_with_stats(model: SmallMLP, inputs: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    features = inputs.view(inputs.size(0), -1)
    augmented_inputs: List[torch.Tensor] = []
    pre_activations: List[torch.Tensor] = []
    for module in model.network:
        if isinstance(module, nn.Linear):
            augmented_inputs.append(torch.cat([features, torch.ones(features.size(0), 1, device=features.device)], dim=1))
            z = module(features)
            pre_activations.append(z)
            features = z
        elif isinstance(module, nn.ReLU):
            features = module(features)
        else:
            raise TypeError(f"Unsupported layer: {type(module)!r}")
    return augmented_inputs, pre_activations


@torch.no_grad()
def collect_regmean_stats(model: SmallMLP, loader: DataLoader, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    model.to(device)
    model.eval()
    layers = model.linear_layers()
    hh = [torch.zeros(layer.in_features + 1, layer.in_features + 1, device=device) for layer in layers]
    zh = [torch.zeros(layer.out_features, layer.in_features + 1, device=device) for layer in layers]
    total = 0

    for inputs, _ in loader:
        inputs = inputs.to(device)
        aug_inputs, pre_activations = linear_forward_with_stats(model, inputs)
        batch_size = inputs.size(0)
        total += batch_size
        for idx, (h, z) in enumerate(zip(aug_inputs, pre_activations)):
            hh[idx] += h.t() @ h
            zh[idx] += z.t() @ h

    return [(layer_hh / total, layer_zh / total) for layer_hh, layer_zh in zip(hh, zh)]


def regmean_merge(
    models: Sequence[SmallMLP],
    stats_per_client: Sequence[List[Tuple[torch.Tensor, torch.Tensor]]],
    weights: Sequence[float],
    ridge_lambda: float,
) -> SmallMLP:
    merged = clone_model(models[0])
    normalized = torch.tensor(weights, dtype=torch.float32)
    normalized = normalized / normalized.sum()
    for layer_idx, layer in enumerate(merged.linear_layers()):
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


def regmean_merge_selected_layers(
    models: Sequence[SmallMLP],
    stats_per_client: Sequence[List[Tuple[torch.Tensor, torch.Tensor]]],
    weights: Sequence[float],
    ridge_lambda: float,
    selected_layer_indices: Sequence[int],
) -> SmallMLP:
    merged = weighted_average(models, weights)
    normalized = torch.tensor(weights, dtype=torch.float32)
    normalized = normalized / normalized.sum()
    selected = set(selected_layer_indices)
    for layer_idx, layer in enumerate(merged.linear_layers()):
        if layer_idx not in selected:
            continue
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
) -> DataLoader:
    chosen = list(indices)
    if len(chosen) > calibration_size:
        generator = torch.Generator().manual_seed(seed)
        positions = torch.randperm(len(chosen), generator=generator)[:calibration_size].tolist()
        chosen = [chosen[pos] for pos in positions]
    return make_loader(dataset, chosen, batch_size, False, seed)


def make_class_balanced_subset(dataset: Dataset, per_class: int, seed: int) -> List[int]:
    labels = dataset_targets(dataset)
    generator = torch.Generator().manual_seed(seed)
    selected: List[int] = []
    for label in range(10):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)].tolist()
        selected.extend(shuffled[:per_class])
    return selected


def fit_linear_probe(
    model: SmallMLP,
    dataset: Dataset,
    indices: Sequence[int],
    config: ExperimentConfig,
    steps: int,
    lr: float,
) -> SmallMLP:
    tuned = clone_model(model)
    device = torch.device(config.device)
    tuned.to(device)
    for parameter in tuned.parameters():
        parameter.requires_grad = False
    last_layer = tuned.linear_layers()[-1]
    last_layer.weight.requires_grad = True
    last_layer.bias.requires_grad = True
    optimizer = torch.optim.Adam([last_layer.weight, last_layer.bias], lr=lr)
    loader = make_loader(dataset, indices, config.train_batch_size, True, config.seed + 999)
    tuned.train()
    step_count = 0
    while step_count < steps:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = tuned(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            step_count += 1
            if step_count >= steps:
                break
    return tuned


@torch.no_grad()
def collect_penultimate_features(
    model: SmallMLP,
    dataset: Dataset,
    indices: Sequence[int],
    config: ExperimentConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(config.device)
    loader = make_loader(dataset, indices, config.eval_batch_size, False, config.seed + 555)
    model = clone_model(model)
    model.to(device)
    model.eval()
    features_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        feats = model.extract_penultimate_features(inputs).cpu()
        features_list.append(feats)
        labels_list.append(targets.cpu())
    return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)


def fit_ridge_probe(
    model: SmallMLP,
    dataset: Dataset,
    indices: Sequence[int],
    config: ExperimentConfig,
    ridge_lambda: float = 1.0,
) -> SmallMLP:
    tuned = clone_model(model)
    features, labels = collect_penultimate_features(model, dataset, indices, config)
    num_classes = 10
    one_hot = torch.zeros(features.size(0), num_classes, dtype=features.dtype)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    augmented = torch.cat([features, torch.ones(features.size(0), 1)], dim=1)
    identity = torch.eye(augmented.size(1), dtype=augmented.dtype)
    solved = torch.linalg.solve(augmented.t() @ augmented + ridge_lambda * identity, augmented.t() @ one_hot)
    weight = solved[:-1, :].t().contiguous()
    bias = solved[-1, :].contiguous()
    last_layer = tuned.linear_layers()[-1]
    last_layer.weight.data.copy_(weight.to(last_layer.weight.data.device))
    last_layer.bias.data.copy_(bias.to(last_layer.bias.data.device))
    return tuned


def fit_bias_only_calibration(
    model: SmallMLP,
    dataset: Dataset,
    indices: Sequence[int],
    config: ExperimentConfig,
    steps: int = 60,
    lr: float = 5e-3,
) -> SmallMLP:
    tuned = clone_model(model)
    device = torch.device(config.device)
    tuned.to(device)
    for parameter in tuned.parameters():
        parameter.requires_grad = False
    last_layer = tuned.linear_layers()[-1]
    last_layer.bias.requires_grad = True
    optimizer = torch.optim.Adam([last_layer.bias], lr=lr)
    loader = make_loader(dataset, indices, config.train_batch_size, True, config.seed + 777)
    tuned.train()
    step_count = 0
    while step_count < steps:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = tuned(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            step_count += 1
            if step_count >= steps:
                break
    return tuned


def fit_logit_affine_calibration(
    model: SmallMLP,
    dataset: Dataset,
    indices: Sequence[int],
    config: ExperimentConfig,
    steps: int = 80,
    lr: float = 1e-2,
    per_class_scale: bool = False,
    learn_bias: bool = True,
) -> SmallMLP:
    device = torch.device(config.device)
    frozen = clone_model(model)
    frozen.to(device)
    frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad = False

    num_classes = frozen.linear_layers()[-1].out_features
    scale_shape = (num_classes,) if per_class_scale else (1,)
    log_scale = torch.zeros(scale_shape, device=device, requires_grad=True)
    bias_delta = torch.zeros(num_classes, device=device, requires_grad=learn_bias)
    params = [log_scale]
    if learn_bias:
        params.append(bias_delta)
    optimizer = torch.optim.Adam(params, lr=lr)
    loader = make_loader(dataset, indices, config.train_batch_size, True, config.seed + 1212)

    step_count = 0
    while step_count < steps:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = frozen(inputs)
            scale = torch.exp(log_scale).view(1, -1 if per_class_scale else 1)
            calibrated_logits = logits * scale
            if learn_bias:
                calibrated_logits = calibrated_logits + bias_delta.view(1, -1)
            loss = F.cross_entropy(calibrated_logits, targets)
            loss.backward()
            optimizer.step()
            step_count += 1
            if step_count >= steps:
                break

    tuned = clone_model(model)
    last_layer = tuned.linear_layers()[-1]
    learned_scale = torch.exp(log_scale.detach())
    if not per_class_scale:
        learned_scale = learned_scale.repeat(num_classes)
    learned_bias = bias_delta.detach() if learn_bias else torch.zeros(num_classes, device=device)
    scale_cpu = learned_scale.to(last_layer.weight.data.device)
    bias_cpu = learned_bias.to(last_layer.bias.data.device)
    last_layer.weight.data.mul_(scale_cpu.unsqueeze(1))
    last_layer.bias.data.mul_(scale_cpu)
    last_layer.bias.data.add_(bias_cpu)
    return tuned


def fit_temperature_calibration(
    model: SmallMLP,
    dataset: Dataset,
    indices: Sequence[int],
    config: ExperimentConfig,
    steps: int = 80,
    lr: float = 1e-2,
) -> SmallMLP:
    return fit_logit_affine_calibration(
        model,
        dataset,
        indices,
        config,
        steps=steps,
        lr=lr,
        per_class_scale=False,
        learn_bias=False,
    )


def fit_vector_scaling_calibration(
    model: SmallMLP,
    dataset: Dataset,
    indices: Sequence[int],
    config: ExperimentConfig,
    steps: int = 100,
    lr: float = 1e-2,
) -> SmallMLP:
    return fit_logit_affine_calibration(
        model,
        dataset,
        indices,
        config,
        steps=steps,
        lr=lr,
        per_class_scale=True,
        learn_bias=True,
    )


def post_merge_finetune(
    model: SmallMLP,
    dataset: Dataset,
    indices: Sequence[int],
    config: ExperimentConfig,
    steps: Optional[int] = None,
    lr: Optional[float] = None,
    layers_to_train: Optional[Sequence[int]] = None,
) -> SmallMLP:
    tuned = clone_model(model)
    device = torch.device(config.device)
    tuned.to(device)
    if layers_to_train is not None:
        selected = set(layers_to_train)
        for idx, layer in enumerate(tuned.linear_layers()):
            trainable = idx in selected
            layer.weight.requires_grad = trainable
            layer.bias.requires_grad = trainable
    optimizer = torch.optim.Adam(
        [parameter for parameter in tuned.parameters() if parameter.requires_grad],
        lr=lr if lr is not None else config.post_merge_lr,
    )
    loader = make_loader(dataset, indices, config.train_batch_size, True, config.seed + 2048)
    tuned.train()
    max_steps = steps if steps is not None else config.post_merge_steps
    step_count = 0
    while step_count < max_steps:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = tuned(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            step_count += 1
            if step_count >= max_steps:
                break
    return tuned


def compute_client_validation_losses(
    client_results: Sequence[ClientResult],
    train_dataset: Dataset,
    config: ExperimentConfig,
) -> List[float]:
    device = torch.device(config.device)
    losses: List[float] = []
    for client in client_results:
        loader = make_calibration_loader(
            train_dataset,
            client.spec.indices,
            config.calibration_size,
            config.eval_batch_size,
            config.seed + client.spec.client_id,
        )
        metrics = evaluate_model(client.model, loader, device)
        losses.append(metrics["loss"])
    return losses


def validation_loss_weights(losses: Sequence[float], temperature: float = 5.0) -> List[float]:
    scores = torch.tensor([-temperature * loss for loss in losses], dtype=torch.float32)
    weights = torch.softmax(scores, dim=0)
    return weights.tolist()


def evaluate_merged_model(
    model: SmallMLP,
    client_results: Sequence[ClientResult],
    train_dataset: Dataset,
    test_loader: DataLoader,
    config: ExperimentConfig,
) -> Dict[str, object]:
    device = torch.device(config.device)
    global_metrics = evaluate_model(model, test_loader, device)
    per_client_local: Dict[str, float] = {}
    for client in client_results:
        loader = make_loader(
            train_dataset,
            client.spec.indices,
            config.eval_batch_size,
            False,
            config.seed + client.spec.client_id,
        )
        per_client_local[f"client_{client.spec.client_id}_local_acc"] = evaluate_model(model, loader, device)["accuracy"]
    return {"global": global_metrics, "per_client_local_accuracy": per_client_local}


@torch.no_grad()
def collect_logits(
    model: SmallMLP,
    dataset: Dataset,
    indices: Sequence[int],
    config: ExperimentConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(config.device)
    loader = make_loader(dataset, indices, config.eval_batch_size, False, config.seed + 313)
    model = clone_model(model)
    model.to(device)
    model.eval()
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        logits = model(inputs).cpu()
        logits_list.append(logits)
        labels_list.append(targets.cpu())
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


@torch.no_grad()
def collect_hidden_state_batches(
    model: SmallMLP,
    dataset: Dataset,
    indices: Sequence[int],
    config: ExperimentConfig,
) -> List[torch.Tensor]:
    device = torch.device(config.device)
    loader = make_loader(dataset, indices, config.eval_batch_size, False, config.seed + 919)
    model = clone_model(model)
    model.to(device)
    model.eval()
    aggregated: Optional[List[torch.Tensor]] = None
    for inputs, _ in loader:
        inputs = inputs.to(device)
        states = [state.cpu() for state in model.collect_hidden_states(inputs)]
        if aggregated is None:
            aggregated = [[] for _ in states]  # type: ignore[assignment]
        for idx, state in enumerate(states):
            aggregated[idx].append(state)  # type: ignore[index]
    assert aggregated is not None
    return [torch.cat(chunks, dim=0) for chunks in aggregated]  # type: ignore[arg-type]


def symmetric_kl_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    log_pa = F.log_softmax(logits_a, dim=1)
    log_pb = F.log_softmax(logits_b, dim=1)
    pa = log_pa.exp()
    pb = log_pb.exp()
    kl_ab = F.kl_div(log_pa, pb, reduction="batchmean", log_target=False)
    kl_ba = F.kl_div(log_pb, pa, reduction="batchmean", log_target=False)
    return float(0.5 * (kl_ab + kl_ba))


def prediction_agreement_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    pred_a = logits_a.argmax(dim=1)
    pred_b = logits_b.argmax(dim=1)
    return float((pred_a == pred_b).float().mean().item())


def mean_feature_cosine(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    norm_a = F.normalize(features_a, dim=1)
    norm_b = F.normalize(features_b, dim=1)
    return float((norm_a * norm_b).sum(dim=1).mean().item())


def linear_cka(features_a: torch.Tensor, features_b: torch.Tensor, eps: float = 1e-8) -> float:
    centered_a = features_a - features_a.mean(dim=0, keepdim=True)
    centered_b = features_b - features_b.mean(dim=0, keepdim=True)
    cross = centered_a.t() @ centered_b
    self_a = centered_a.t() @ centered_a
    self_b = centered_b.t() @ centered_b
    numerator = torch.sum(cross * cross)
    denominator = torch.sqrt(torch.sum(self_a * self_a) * torch.sum(self_b * self_b) + eps)
    return float((numerator / denominator.clamp_min(eps)).item())


def class_center_cosine(features_a: torch.Tensor, features_b: torch.Tensor, labels: torch.Tensor) -> float:
    cosines: List[torch.Tensor] = []
    for label in labels.unique(sorted=True):
        mask = labels == label
        if mask.sum().item() == 0:
            continue
        center_a = features_a[mask].mean(dim=0)
        center_b = features_b[mask].mean(dim=0)
        cosine = F.cosine_similarity(center_a.unsqueeze(0), center_b.unsqueeze(0), dim=1)
        cosines.append(cosine.squeeze(0))
    if not cosines:
        return 0.0
    return float(torch.stack(cosines).mean().item())


def run_single_experiment(config: ExperimentConfig) -> Dict[str, object]:
    set_seed(config.seed)
    train_dataset, test_dataset = build_datasets(config.data_root)
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)
    base_model = SmallMLP(hidden_sizes=config.hidden_sizes)
    client_results = train_clients(config, train_dataset, base_model)
    client_models = [item.model for item in client_results]

    sample_weights = [len(item.spec.indices) for item in client_results]
    epoch_weights = [item.spec.epochs for item in client_results]
    validation_losses = compute_client_validation_losses(client_results, train_dataset, config)
    val_loss_weights = validation_loss_weights(validation_losses)
    stats_per_client = [
        collect_regmean_stats(
            item.model,
            make_calibration_loader(
                train_dataset,
                item.spec.indices,
                config.calibration_size,
                config.eval_batch_size,
                config.seed + item.spec.client_id,
            ),
            torch.device(config.device),
        )
        for item in client_results
    ]

    merged_models = {
        "simple_avg": simple_average(client_models),
        "sample_weighted_avg": weighted_average(client_models, sample_weights),
        "epoch_weighted_avg": weighted_average(client_models, epoch_weights),
        "val_loss_weighted_avg": weighted_average(client_models, val_loss_weights),
        "regmean_equal": regmean_merge(client_models, stats_per_client, [1.0] * len(client_models), config.regmean_lambda),
        "regmean_epoch_weighted": regmean_merge(client_models, stats_per_client, epoch_weights, config.regmean_lambda),
        "regmean_val_loss_weighted": regmean_merge(client_models, stats_per_client, val_loss_weights, config.regmean_lambda),
        "centralized": train_centralized(config, train_dataset, base_model),
    }

    merges = {
        name: evaluate_merged_model(model, client_results, train_dataset, test_loader, config)
        for name, model in merged_models.items()
    }
    best_single = max(
        [
            (item.spec.client_id, evaluate_model(item.model, test_loader, torch.device(config.device)))
            for item in client_results
        ],
        key=lambda pair: pair[1]["accuracy"],
    )
    merges["single_best"] = {"client_id": best_single[0], "metrics": best_single[1]}

    return {
        "config": config_to_jsonable(config),
        "clients": [
            {
                "client_id": item.spec.client_id,
                "epochs": item.spec.epochs,
                "sample_count": len(item.spec.indices),
                "train_metrics": item.train_metrics,
            }
            for item in client_results
        ],
        "merges": merges,
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
