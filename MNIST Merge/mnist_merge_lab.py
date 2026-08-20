#!/usr/bin/env python3
"""
Minimal MNIST + MLP merge lab.

This script builds a controllable experiment platform for validating whether
data-guided merging (RegMean) can recover more ability than naive parameter
averaging when local models differ in training budget and data distribution.
"""

from __future__ import annotations

import argparse
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


class SmallMLP(nn.Module):
    def __init__(self, hidden_sizes: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        dims = [28 * 28, *hidden_sizes, 10]
        layers: List[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.network(x)

    def linear_layers(self) -> List[nn.Linear]:
        return [module for module in self.network if isinstance(module, nn.Linear)]


@dataclass
class ClientConfig:
    client_id: int
    epochs: int
    sample_count: int


@dataclass
class ClientArtifacts:
    config: ClientConfig
    indices: List[int]
    model: SmallMLP
    metrics: Dict[str, float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNIST merge lab")
    parser.add_argument("--data-root", type=Path, default=Path("./work/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/mnist_merge_lab"))
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["iid", "pair-label", "dirichlet"],
        default="iid",
    )
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--calibration-size", type=int, default=256)
    parser.add_argument("--regmean-lambda", type=float, default=1e-3)
    parser.add_argument(
        "--merge-methods",
        type=str,
        nargs="+",
        default=[
            "single_best",
            "simple_avg",
            "sample_weighted_avg",
            "epoch_weighted_avg",
            "regmean_equal",
            "regmean_epoch_weighted",
        ],
    )
    parser.add_argument(
        "--client-epochs",
        type=int,
        nargs="*",
        default=None,
        help="Optional per-client epochs. Defaults to all clients using 5 epochs.",
    )
    parser.add_argument("--centralized-epochs", type=int, default=5)
    parser.add_argument("--skip_centralized", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def build_datasets(data_root: Path) -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def make_client_epoch_list(num_clients: int, provided: Sequence[int] | None) -> List[int]:
    if provided is None or len(provided) == 0:
        return [5 for _ in range(num_clients)]
    if len(provided) != num_clients:
        raise ValueError(f"Expected {num_clients} epoch values, got {len(provided)}.")
    return list(provided)


def dataset_targets(dataset: Dataset) -> torch.Tensor:
    if hasattr(dataset, "targets"):
        return torch.as_tensor(getattr(dataset, "targets"), dtype=torch.long)
    raise ValueError("Dataset does not expose targets.")


def iid_split(dataset: Dataset, num_clients: int, seed: int) -> List[List[int]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(dataset), generator=generator).tolist()
    chunk_size = math.ceil(len(permutation) / num_clients)
    return [permutation[index * chunk_size : (index + 1) * chunk_size] for index in range(num_clients)]


def pair_label_split(dataset: Dataset, num_clients: int, seed: int) -> List[List[int]]:
    if num_clients != 10:
        raise ValueError("pair-label split expects exactly 10 clients.")
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
    labels = dataset_targets(dataset)
    generator = torch.Generator().manual_seed(seed)
    label_to_indices: Dict[int, List[int]] = {}
    for label in range(10):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)].tolist()
        label_to_indices[label] = shuffled

    usage_count = {label: 0 for label in range(10)}
    for left, right in pairs:
        usage_count[left] += 1
        usage_count[right] += 1

    label_chunks: Dict[int, List[List[int]]] = {}
    for label, indices in label_to_indices.items():
        chunk_size = math.ceil(len(indices) / usage_count[label])
        chunks = [indices[index * chunk_size : (index + 1) * chunk_size] for index in range(usage_count[label])]
        label_chunks[label] = chunks

    label_offsets = {label: 0 for label in range(10)}
    splits: List[List[int]] = []
    for left, right in pairs:
        left_chunk = label_chunks[left][label_offsets[left]]
        right_chunk = label_chunks[right][label_offsets[right]]
        label_offsets[left] += 1
        label_offsets[right] += 1
        splits.append(left_chunk + right_chunk)
    return splits


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
            order = torch.argsort(proportions, descending=True)
            counts[order[:remainder]] += 1
        offset = 0
        for client_id, count in enumerate(counts.tolist()):
            if count == 0:
                continue
            chunk = label_indices[offset : offset + count].tolist()
            client_indices[client_id].extend(chunk)
            offset += count
    return client_indices


def split_dataset(dataset: Dataset, args: argparse.Namespace) -> List[List[int]]:
    if args.split_mode == "iid":
        return iid_split(dataset, args.num_clients, args.seed)
    if args.split_mode == "pair-label":
        return pair_label_split(dataset, args.num_clients, args.seed)
    return dirichlet_split(dataset, args.num_clients, args.dirichlet_alpha, args.seed)


def make_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    subset = Subset(dataset, list(indices))
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_one_model(
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
    total_examples = 0
    total_loss = 0.0
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

    per_class_accuracy = {
        f"class_{label}_acc": (
            float(per_class_correct[label].item()) / max(1, int(per_class_total[label].item()))
        )
        for label in range(10)
    }
    macro_accuracy = sum(per_class_accuracy.values()) / 10.0
    worst_class_accuracy = min(per_class_accuracy.values())

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "macro_accuracy": macro_accuracy,
        "worst_class_accuracy": worst_class_accuracy,
        **per_class_accuracy,
    }


def clone_model(base_model: SmallMLP) -> SmallMLP:
    new_model = SmallMLP(hidden_sizes=[layer.out_features for layer in base_model.linear_layers()[:-1]])
    new_model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    return new_model


def average_state_dicts(state_dicts: Sequence[Dict[str, torch.Tensor]], weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    normalized = torch.tensor(weights, dtype=torch.float32)
    normalized = normalized / normalized.sum()
    merged: Dict[str, torch.Tensor] = {}
    for key in state_dicts[0].keys():
        merged[key] = sum(weight * state[key] for weight, state in zip(normalized.tolist(), state_dicts))
    return merged


def simple_average(models: Sequence[SmallMLP]) -> SmallMLP:
    state_dicts = [model.state_dict() for model in models]
    merged = clone_model(models[0])
    merged.load_state_dict(average_state_dicts(state_dicts, [1.0] * len(models)))
    return merged


def weighted_average(models: Sequence[SmallMLP], weights: Sequence[float]) -> SmallMLP:
    state_dicts = [model.state_dict() for model in models]
    merged = clone_model(models[0])
    merged.load_state_dict(average_state_dicts(state_dicts, weights))
    return merged


def linear_forward_with_activations(model: SmallMLP, inputs: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    features = inputs.view(inputs.size(0), -1)
    hidden_with_bias: List[torch.Tensor] = []
    pre_activations: List[torch.Tensor] = []

    for module in model.network:
        if isinstance(module, nn.Linear):
            augmented = torch.cat([features, torch.ones(features.size(0), 1, device=features.device)], dim=1)
            hidden_with_bias.append(augmented)
            z = module(features)
            pre_activations.append(z)
            features = z
        elif isinstance(module, nn.ReLU):
            features = module(features)
        else:
            raise TypeError(f"Unsupported module in network: {type(module)!r}")
    return hidden_with_bias, pre_activations


@torch.no_grad()
def collect_regmean_stats(
    model: SmallMLP,
    loader: DataLoader,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    model.to(device)
    model.eval()
    linear_layers = model.linear_layers()
    hh = [torch.zeros(layer.in_features + 1, layer.in_features + 1, device=device) for layer in linear_layers]
    zh = [torch.zeros(layer.out_features, layer.in_features + 1, device=device) for layer in linear_layers]
    total = 0

    for inputs, _ in loader:
        inputs = inputs.to(device)
        hidden_with_bias, pre_activations = linear_forward_with_activations(model, inputs)
        batch_size = inputs.size(0)
        total += batch_size
        for layer_index, (h_tilde, z) in enumerate(zip(hidden_with_bias, pre_activations)):
            hh[layer_index] += h_tilde.t() @ h_tilde
            zh[layer_index] += z.t() @ h_tilde

    stats: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for layer_hh, layer_zh in zip(hh, zh):
        stats.append((layer_hh / total, layer_zh / total))
    return stats


def regmean_merge(
    models: Sequence[SmallMLP],
    stats_per_client: Sequence[List[Tuple[torch.Tensor, torch.Tensor]]],
    client_weights: Sequence[float],
    ridge_lambda: float,
) -> SmallMLP:
    merged = clone_model(models[0])
    normalized = torch.tensor(client_weights, dtype=torch.float32)
    normalized = normalized / normalized.sum()
    merged_layers = merged.linear_layers()

    for layer_index, merged_layer in enumerate(merged_layers):
        hh_sum = None
        zh_sum = None
        for client_index, layer_stats in enumerate(stats_per_client):
            hh, zh = layer_stats[layer_index]
            weight = normalized[client_index].item()
            weighted_hh = weight * hh
            weighted_zh = weight * zh
            hh_sum = weighted_hh if hh_sum is None else hh_sum + weighted_hh
            zh_sum = weighted_zh if zh_sum is None else zh_sum + weighted_zh

        identity = torch.eye(hh_sum.size(0), device=hh_sum.device, dtype=hh_sum.dtype)
        solved = zh_sum @ torch.linalg.inv(hh_sum + ridge_lambda * identity)
        merged_layer.weight.data.copy_(solved[:, :-1].to(merged_layer.weight.data.device))
        merged_layer.bias.data.copy_(solved[:, -1].to(merged_layer.bias.data.device))
    return merged


def summarize_client_distribution(dataset: Dataset, indices: Sequence[int]) -> Dict[str, float]:
    labels = dataset_targets(dataset)[list(indices)]
    counts = torch.bincount(labels, minlength=10).float()
    proportions = counts / counts.sum().clamp_min(1.0)
    return {f"class_{label}_share": float(proportions[label].item()) for label in range(10)}


def train_clients(
    base_model: SmallMLP,
    train_dataset: Dataset,
    client_splits: Sequence[Sequence[int]],
    args: argparse.Namespace,
    device: torch.device,
) -> List[ClientArtifacts]:
    epoch_plan = make_client_epoch_list(args.num_clients, args.client_epochs)
    client_artifacts: List[ClientArtifacts] = []

    for client_id, indices in enumerate(client_splits):
        train_loader = make_loader(
            train_dataset,
            indices,
            batch_size=args.train_batch_size,
            shuffle=True,
            seed=args.seed + client_id,
        )
        model = clone_model(base_model)
        train_one_model(
            model=model,
            loader=train_loader,
            epochs=epoch_plan[client_id],
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        train_metrics = evaluate_model(model, train_loader, device)
        config = ClientConfig(client_id=client_id, epochs=epoch_plan[client_id], sample_count=len(indices))
        metrics = {
            **train_metrics,
            **summarize_client_distribution(train_dataset, indices),
        }
        client_artifacts.append(ClientArtifacts(config=config, indices=list(indices), model=model, metrics=metrics))
    return client_artifacts


def train_centralized(
    base_model: SmallMLP,
    train_dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> SmallMLP:
    indices = list(range(len(train_dataset)))
    loader = make_loader(
        train_dataset,
        indices,
        batch_size=args.train_batch_size,
        shuffle=True,
        seed=args.seed,
    )
    model = clone_model(base_model)
    train_one_model(
        model=model,
        loader=loader,
        epochs=args.centralized_epochs,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    return model


def make_calibration_loader(
    train_dataset: Dataset,
    indices: Sequence[int],
    calibration_size: int,
    batch_size: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    chosen = list(indices)
    if len(chosen) > calibration_size:
        perm = torch.randperm(len(chosen), generator=generator)[:calibration_size].tolist()
        chosen = [chosen[position] for position in perm]
    return make_loader(train_dataset, chosen, batch_size=batch_size, shuffle=False, seed=seed)


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = build_datasets(args.data_root)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    client_splits = split_dataset(train_dataset, args)

    base_model = SmallMLP(hidden_sizes=args.hidden_sizes)
    client_artifacts = train_clients(base_model, train_dataset, client_splits, args, device)
    client_models = [artifact.model for artifact in client_artifacts]
    client_eval_loaders = [
        make_loader(
            train_dataset,
            artifact.indices,
            batch_size=args.eval_batch_size,
            shuffle=False,
            seed=args.seed + artifact.config.client_id,
        )
        for artifact in client_artifacts
    ]

    results: Dict[str, object] = {
        "config": vars(args),
        "clients": [
            {
                "client_id": artifact.config.client_id,
                "epochs": artifact.config.epochs,
                "sample_count": artifact.config.sample_count,
                "train_metrics": artifact.metrics,
            }
            for artifact in client_artifacts
        ],
        "merges": {},
    }

    def evaluate_merged_model(model: SmallMLP) -> Dict[str, object]:
        metrics = evaluate_model(model, test_loader, device)
        per_client_accuracy = {
            f"client_{artifact.config.client_id}_local_acc": evaluate_model(model, loader, device)["accuracy"]
            for artifact, loader in zip(client_artifacts, client_eval_loaders)
        }
        return {
            "global": metrics,
            "per_client_local_accuracy": per_client_accuracy,
        }

    single_best = max(
        (
            (
                artifact.config.client_id,
                evaluate_model(artifact.model, test_loader, device),
            )
            for artifact in client_artifacts
        ),
        key=lambda item: item[1]["accuracy"],
    )
    if "single_best" in args.merge_methods:
        results["merges"]["single_best"] = {
            "client_id": single_best[0],
            "metrics": single_best[1],
        }

    if not args.skip_centralized:
        centralized_model = train_centralized(base_model, train_dataset, args, device)
        results["merges"]["centralized"] = evaluate_merged_model(centralized_model)

    if "simple_avg" in args.merge_methods:
        merged = simple_average(client_models)
        results["merges"]["simple_avg"] = evaluate_merged_model(merged)

    sample_weights = [artifact.config.sample_count for artifact in client_artifacts]
    if "sample_weighted_avg" in args.merge_methods:
        merged = weighted_average(client_models, sample_weights)
        results["merges"]["sample_weighted_avg"] = evaluate_merged_model(merged)

    epoch_weights = [artifact.config.epochs for artifact in client_artifacts]
    if "epoch_weighted_avg" in args.merge_methods:
        merged = weighted_average(client_models, epoch_weights)
        results["merges"]["epoch_weighted_avg"] = evaluate_merged_model(merged)

    if "regmean_equal" in args.merge_methods or "regmean_epoch_weighted" in args.merge_methods:
        stats_per_client = []
        for artifact in client_artifacts:
            calibration_loader = make_calibration_loader(
                train_dataset=train_dataset,
                indices=artifact.indices,
                calibration_size=args.calibration_size,
                batch_size=args.eval_batch_size,
                seed=args.seed + artifact.config.client_id,
            )
            stats = collect_regmean_stats(artifact.model, calibration_loader, device)
            stats_per_client.append(stats)

        if "regmean_equal" in args.merge_methods:
            merged = regmean_merge(
                models=client_models,
                stats_per_client=stats_per_client,
                client_weights=[1.0] * len(client_artifacts),
                ridge_lambda=args.regmean_lambda,
            )
            results["merges"]["regmean_equal"] = evaluate_merged_model(merged)

        if "regmean_epoch_weighted" in args.merge_methods:
            merged = regmean_merge(
                models=client_models,
                stats_per_client=stats_per_client,
                client_weights=epoch_weights,
                ridge_lambda=args.regmean_lambda,
            )
            results["merges"]["regmean_epoch_weighted"] = evaluate_merged_model(merged)

    return results


def main() -> None:
    args = parse_args()
    results = run_experiment(args)

    output_path = args.output_dir / "results.json"
    serializable = json.loads(json.dumps(results, default=lambda value: str(value)))
    output_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
    print(json.dumps(serializable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
