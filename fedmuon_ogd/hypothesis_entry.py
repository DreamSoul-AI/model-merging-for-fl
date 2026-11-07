"""Falsification entry for hypothesis H-absent. The hypothesis is not a result.

Each round freezes the model and the global control. Every selected client trains
from that snapshot. The server then applies the partial-participation update once:
X <- ((N-S)/N) X + (1/N) sum_{i in S} X_i, and
C <- C + (1/N) sum_{i in S} (C_i_new - C_i_old).
N is the number of clients and S is the number of participants.

H-absent is a tool diagnostic, not a paper claim. A client subspace versus a
shared subspace repeats local-versus-global subspace effects that already
appear in the literature. Do not scale this entry. The subspaces it measures
are right singular vectors of gradients, not activation bases.

A run is marked executed_on_rtx_5080 only after a training step has run on a
CUDA parameter device whose name contains 5080. Seeing that GPU in the machine
is not enough. Pass --device explicitly.

The one command on the GPU machine is bash fedmuon_ogd/start_5080.sh.
It runs this self-check with --device cuda, then the CIFAR-100 smoke config.
It does not start the 10-seed preregistered config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from decimal import Decimal
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_discriminate as audit

ROOT = Path(__file__).resolve().parent
NS_A = Decimal("3.4445")
NS_B = Decimal("-4.775")
NS_C = Decimal("2.0315")


def ns_scalar(value: Decimal) -> Decimal:
    return NS_A * value + NS_B * value ** 3 + NS_C * value ** 5


def ns_polynomial(matrix: torch.Tensor) -> torch.Tensor:
    gram = matrix @ matrix.transpose(-2, -1)
    return 3.4445 * matrix + (gram * -4.775 + gram @ gram * 2.0315) @ matrix


def foundation_checks() -> dict:
    """Separate exact one-sided preservation from finite precision, two-sided constraints, and the spectral bound."""
    scalar = ns_scalar(Decimal("0.5"))
    if scalar != Decimal("1.188859375"):
        raise AssertionError(f"scalar Newton-Schulz value is {scalar}")
    generator = torch.Generator().manual_seed(0)
    basis = torch.linalg.qr(torch.randn(5, 2, generator=generator, dtype=torch.float64)).Q
    complement = torch.linalg.qr(basis, mode="complete").Q[:, 2:]
    raw = torch.randn(6, 3, generator=generator, dtype=torch.float64) @ complement.T
    normalized = raw / torch.linalg.norm(raw)
    mapped = ns_polynomial(normalized)
    float32_mapped = ns_polynomial(normalized.float())
    twosided = torch.tensor([[0.3, 0.0], [0.7, 0.4]], dtype=torch.float64)
    twosided = twosided / torch.linalg.norm(twosided)
    twosided_mapped = ns_polynomial(twosided)
    report = {
        "scalar_0_5": str(scalar),
        "one_sided_float64_residual": float((mapped @ basis).norm()),
        "one_sided_float64_spectral_norm": float(torch.linalg.matrix_norm(mapped, ord=2)),
        "one_sided_float32_residual": float((float32_mapped @ basis.float()).norm()),
        "one_sided_float32_spectral_norm": float(torch.linalg.matrix_norm(float32_mapped, ord=2)),
        "two_sided_entry_before": float(twosided[0, 1]),
        "two_sided_entry_after": float(twosided_mapped[0, 1]),
    }
    if report["one_sided_float64_residual"] > 1e-12:
        raise AssertionError(report)
    if report["one_sided_float64_spectral_norm"] <= 1:
        raise AssertionError(report)
    if abs(report["two_sided_entry_before"]) > 1e-15 or abs(report["two_sided_entry_after"]) < 1e-3:
        raise AssertionError(report)
    report["unnormalized_ns_obeys_unit_spectral_bound"] = False
    report["statements"] = [
        "In exact arithmetic, X |-> aX + b XX^T X + c (XX^T)^2 X preserves XB=0.",
        "Coefficients 3.4445, -4.775, 2.0315 send 0.5 to 1.188859375.",
        "The measured spectral norm of one unnormalized Newton-Schulz step is about 1.199. That does not satisfy ||D||_2 <= 1.",
        "A training step that is required to lie in the unit spectral ball is divided by max(sigma, 1). The unnormalized quintic is not that step.",
        "A two-sided entry constraint is not preserved by that map.",
        "A float32 residual is a finite-precision observation, not a structural failure of the one-sided identity.",
    ]
    return report


PRIMARY_METRIC = "mean_absent_client_old_task_accuracy"
REJECT_IF = (
    "The 95% bootstrap interval of the mean paired difference of "
    "mean_absent_client_old_task_accuracy across seeds covers 0 or lies at or below 0."
)


def executed_on_rtx_5080(parameter_device: str, cuda_training_steps: int, used_cuda_device_name: str | None) -> bool:
    """True only when training steps actually ran on a CUDA device named 5080."""
    if cuda_training_steps <= 0 or not str(parameter_device).startswith("cuda"):
        return False
    return bool(used_cuda_device_name and "5080" in used_cuda_device_name)


def cuda_device_name(parameter_device: str) -> str | None:
    if not str(parameter_device).startswith("cuda") or not torch.cuda.is_available():
        return None
    index = int(parameter_device.split(":")[1]) if ":" in parameter_device else 0
    return torch.cuda.get_device_name(index)


def environment_report(parameter_device: str, cuda_training_steps: int) -> dict:
    visible = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    used = cuda_device_name(parameter_device) if cuda_training_steps > 0 else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_cuda_device_name": visible,
        "parameter_device": parameter_device,
        "cuda_training_steps": int(cuda_training_steps),
        "used_cuda_device_name": used,
        "executed_on_rtx_5080": executed_on_rtx_5080(parameter_device, cuda_training_steps, used),
    }


def config_digest(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def tasks_from_config(config: dict) -> list[tuple[int, ...]]:
    width = int(config["classes_per_task"])
    count = int(config["num_classes"]) // width
    return [tuple(range(task * width, (task + 1) * width)) for task in range(count)]


def group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.norm1(self.conv1(inputs)))
        hidden = self.norm2(self.conv2(hidden))
        residual = inputs if self.skip is None else self.skip(inputs)
        return F.relu(hidden + residual)


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(group_count(64), 64)
        self.layer1 = self._make(64, 64, 2, 1)
        self.layer2 = self._make(64, 128, 2, 2)
        self.layer3 = self._make(128, 256, 2, 2)
        self.layer4 = self._make(256, 512, 2, 2)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.norm1(self.conv1(inputs)))
        hidden = self.layer4(self.layer3(self.layer2(self.layer1(hidden))))
        hidden = hidden.mean(dim=(2, 3))
        return self.fc(hidden)


class SmallMLP(nn.Module):
    def __init__(self, features: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(features, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(inputs)))


def make_model(profile: str, num_classes: int, features: int, seed: int) -> nn.Module:
    generator = torch.Generator().manual_seed(seed)
    if profile == "plumbing-mlp":
        model = SmallMLP(features, num_classes)
    elif profile == "cifar100-resnet18-groupnorm":
        model = ResNet18CIFAR(num_classes)
    else:
        raise ValueError(profile)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    return model


def matrix_view(tensor: torch.Tensor) -> torch.Tensor | None:
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 4:
        return tensor.reshape(tensor.shape[0], -1)
    return None


def restore_view(matrix: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return matrix.reshape(like.shape).to(dtype=like.dtype)


def spectral_norm(matrix: torch.Tensor) -> float:
    if matrix.ndim < 2:
        return float(matrix.norm())
    return float(torch.linalg.matrix_norm(matrix.detach().float(), ord=2))


def canonicalize_signs(vectors: torch.Tensor) -> torch.Tensor:
    signed = vectors.clone()
    for column in range(signed.shape[1]):
        pivot = int(torch.argmax(signed[:, column].abs()))
        if signed[pivot, column] < 0:
            signed[:, column] = -signed[:, column]
    return signed


def top_right_vectors(matrix: torch.Tensor, rank: int) -> torch.Tensor | None:
    """Right singular vectors of a gradient matrix. Relative cutoff is 1e-8 * sigma_max."""
    work = matrix.detach().to(torch.float64).cpu()
    if work.numel() == 0 or float(work.norm()) == 0.0:
        return None
    _left, values, right = torch.linalg.svd(work, full_matrices=False)
    if values.numel() == 0 or float(values[0]) == 0.0:
        return None
    kept = min(rank, int((values > values[0] * 1e-8).sum()))
    if kept <= 0:
        return None
    return canonicalize_signs(right[:kept, :].T.contiguous())


def top_invariant_subspace(pairs: list[tuple[torch.Tensor, float]], cap: int) -> torch.Tensor | None:
    """Top eigenvectors of sum_i w_i V_i V_i^T.

    Each V_i holds orthonormal columns. Every pair with a positive weight enters
    the sum, so a later client is not dropped because an earlier client already
    filled `cap` columns. The sum does not depend on pair order in exact arithmetic.
    """
    columns = []
    for factor, weight in pairs:
        if factor is None or weight <= 0 or factor.numel() == 0:
            continue
        columns.append(factor.detach().to(torch.float64).cpu() * (float(weight) ** 0.5))
    if not columns:
        return None
    stacked = torch.cat(columns, dim=1)
    width, count = stacked.shape
    if count <= width:
        gram = stacked.T @ stacked
        values, vectors = torch.linalg.eigh(gram)
        chosen = [int(index) for index in torch.argsort(values, descending=True).tolist() if float(values[index]) > 1e-8]
        chosen = chosen[:cap]
        if not chosen:
            return None
        mapped = stacked @ vectors[:, chosen]
    else:
        covariance = stacked @ stacked.T
        values, vectors = torch.linalg.eigh(covariance)
        chosen = [int(index) for index in torch.argsort(values, descending=True).tolist() if float(values[index]) > 1e-8]
        chosen = chosen[:cap]
        if not chosen:
            return None
        mapped = vectors[:, chosen]
    factor, _ = torch.linalg.qr(mapped, mode="reduced")
    return canonicalize_signs(factor[:, : mapped.shape[1]].contiguous())


def projector(vectors: torch.Tensor) -> torch.Tensor:
    return vectors @ vectors.T


def clone_named(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in state.items()}


def fedmuon_partial_aggregate(snapshot, local_states, snapshot_control, increments, n_total: int):
    """One synchronous FedMuon aggregate.

    X <- ((N-S)/N) X + (1/N) sum_{i in S} X_i
    C <- C + (1/N) sum_{i in S} (C_i_new - C_i_old)
    N is the number of clients. S is the number of participants. Non-participants
    keep the round-start model in the average and add no control increment.
    """
    participants = len(local_states)
    if participants != len(increments):
        raise RuntimeError("each participant needs one control increment")
    new_params = {}
    scale_keep = (n_total - participants) / n_total
    scale_client = 1.0 / n_total
    for name, value in snapshot.items():
        acc = value.detach() * scale_keep
        for local in local_states:
            acc = acc + local[name].detach() * scale_client
        new_params[name] = acc
    new_control = {}
    for name, value in snapshot_control.items():
        acc = value.detach().clone()
        for increment in increments:
            acc = acc + increment[name].detach() * scale_client
        new_control[name] = acc
    return new_params, new_control


def max_state_diff(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    names = set(left) | set(right)
    if set(left) != set(right):
        return float("inf")
    return max(float((left[name] - right[name]).abs().max()) for name in names)


def zero_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(parameter) for name, parameter in model.named_parameters()}


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def load_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def parameter_device(model: nn.Module) -> str:
    return str(next(model.parameters()).device)


def move_state(state: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.detach().to(device) for name, value in state.items()}


def cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def move_basis(basis: dict[str, torch.Tensor] | None, device: torch.device) -> dict[str, torch.Tensor] | None:
    if basis is None:
        return None
    return {name: value.detach().to(device=device, dtype=torch.float32) for name, value in basis.items()}


def cpu_basis(basis: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
    if basis is None:
        return None
    return {name: value.detach().cpu().clone() for name, value in basis.items()}


def prepare_inputs(inputs: torch.Tensor, profile: str) -> torch.Tensor:
    if profile == "cifar100-resnet18-groupnorm":
        return inputs.view(-1, 32, 32, 3).permute(0, 3, 1, 2).contiguous()
    return inputs


def client_step(model, momentum, control, global_control, batch_x, batch_y, basis, alpha, lr, lr_vector):
    device = next(model.parameters()).device
    if batch_x.device != device or batch_y.device != device:
        raise RuntimeError(f"batch is on {batch_x.device}/{batch_y.device}, model is on {device}")
    for name in momentum:
        if momentum[name].device != device or control[name].device != device or global_control[name].device != device:
            raise RuntimeError(f"optimizer state {name} is not on {device}")
    if basis is not None:
        for name, factor in basis.items():
            if factor.device != device:
                raise RuntimeError(f"protection basis {name} is on {factor.device}, model is on {device}")
    model.train()
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(batch_x), batch_y)
    loss.backward()
    residual = 0.0
    counted = 0
    norms = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            gradient = parameter.grad.detach()
            view = matrix_view(gradient)
            if view is None:
                momentum[name].mul_(1 - alpha).add_(gradient, alpha=alpha)
                corrected = momentum[name] - control[name] + global_control[name]
                parameter.add_(corrected, alpha=-lr_vector)
                continue
            momentum_view = matrix_view(momentum[name])
            control_view = matrix_view(control[name])
            global_view = matrix_view(global_control[name])
            momentum_view.mul_(1 - alpha).add_(view, alpha=alpha)
            corrected = momentum_view - control_view + global_view
            factor = None if basis is None or basis.get(name) is None else basis[name]
            direction = audit.onesided_lmo(corrected, factor)
            sigma = spectral_norm(direction)
            if sigma > 1:
                direction = direction / sigma
            sigma_after = spectral_norm(direction)
            if sigma_after > 1 + 1e-4:
                raise RuntimeError(f"{name} spectral norm {sigma_after} exceeds the unit ball after correction")
            if factor is not None:
                residual += float((direction.double() @ factor.double()).norm())
                counted += 1
            norms.append(sigma_after)
            parameter.add_(restore_view(direction, parameter), alpha=-lr)
    return residual / max(counted, 1), (sum(norms) / len(norms) if norms else 0.0)


def gradient_factors(model, features, labels, rank: int, profile: str, sample_size: int) -> tuple[dict[str, torch.Tensor], float]:
    """Right singular vectors of this client's gradient. This is a gradient basis, not an activation basis."""
    count = int(features.shape[0])
    if count < 2:
        return {}, 0.0
    taken = min(sample_size, count)
    device = next(model.parameters()).device
    model.train()
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(prepare_inputs(features[:taken], profile).to(device)), labels[:taken].to(device))
    loss.backward()
    factors = {}
    for name, parameter in model.named_parameters():
        view = matrix_view(parameter.grad.detach())
        if view is None:
            continue
        factor = top_right_vectors(view, rank)
        if factor is not None:
            factors[name] = factor
    model.zero_grad(set_to_none=True)
    return factors, float(taken)


def empty_contributions(n_clients: int) -> list[dict]:
    return [{} for _ in range(n_clients)]


def append_task_factors(contributions, clients, model, rank, profile, sample_size, task_id):
    for client_id, (features, labels) in enumerate(clients):
        factors, weight = gradient_factors(model, features, labels, rank, profile, sample_size)
        for name, factor in factors.items():
            contributions[client_id].setdefault(name, []).append(
                {"task": task_id, "weight": weight, "factor": factor.cpu()}
            )


def materialize_client_bases(contributions, cap: int) -> list[dict[str, torch.Tensor] | None]:
    bases = []
    for client in contributions:
        local = {}
        for name, items in client.items():
            vectors = top_invariant_subspace([(item["factor"], item["weight"]) for item in items], cap)
            if vectors is not None:
                local[name] = vectors
        bases.append(local or None)
    return bases


def materialize_shared_basis(contributions, cap: int) -> dict[str, torch.Tensor] | None:
    names = set()
    for client in contributions:
        names.update(client)
    shared = {}
    for name in sorted(names):
        pairs = [
            (item["factor"], item["weight"])
            for client in contributions
            for item in client.get(name, [])
        ]
        vectors = top_invariant_subspace(pairs, cap)
        if vectors is not None:
            shared[name] = vectors
    return shared or None


def random_matched_basis(shared: dict[str, torch.Tensor] | None, seed: int, task_id: int) -> dict[str, torch.Tensor] | None:
    """A control subspace with the same shapes as the shared gradient subspace. It is not a gradient basis."""
    if not shared:
        return None
    generator = torch.Generator().manual_seed(9000 + 13 * seed + task_id)
    matched = {}
    for name, vectors in shared.items():
        raw = torch.randn(vectors.shape[0], vectors.shape[1], generator=generator, dtype=torch.float64)
        factor, _ = torch.linalg.qr(raw, mode="reduced")
        matched[name] = canonicalize_signs(factor[:, : vectors.shape[1]].contiguous())
    return matched


def argmax_predictions(model, features, profile: str, batch_size: int) -> torch.Tensor:
    """Evaluate in batches so a CIFAR test split is not placed on the device at once."""
    device = next(model.parameters()).device
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), batch_size):
            batch = prepare_inputs(features[start : start + batch_size], profile).to(device)
            chunks.append(model(batch).argmax(dim=1).cpu())
    if not chunks:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(chunks)


def pool_client_test(test_splits, client_id: int, task_ids: list[int]):
    features = []
    labels = []
    for task_id in task_ids:
        client_x, client_y = test_splits[task_id][client_id]
        if int(client_y.numel()) == 0:
            continue
        features.append(client_x)
        labels.append(client_y)
    if not labels:
        return None, None
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def absent_old_task_metric(model, test_splits, chosen: list[int], n_clients: int, task_id: int, profile: str, batch_size: int) -> dict:
    """Unweighted mean of absent clients' argmax accuracy on their own pooled old-task test examples."""
    chosen_set = set(int(client_id) for client_id in chosen)
    absent = [client_id for client_id in range(n_clients) if client_id not in chosen_set]
    scored = []
    if task_id > 0:
        for client_id in absent:
            features, labels = pool_client_test(test_splits, client_id, list(range(task_id)))
            if labels is None:
                continue
            predicted = argmax_predictions(model, features, profile, batch_size)
            scored.append(
                {
                    "client": client_id,
                    "accuracy": float((predicted == labels).float().mean()),
                    "test_examples": int(labels.shape[0]),
                }
            )
    accuracies = [row["accuracy"] for row in scored]
    return {
        "absent_clients": absent,
        "scored_absent_clients": scored,
        PRIMARY_METRIC: None if not accuracies else sum(accuracies) / len(accuracies),
        "worst_absent_client_old_task_accuracy": None if not accuracies else min(accuracies),
    }


def checkpoint_path(out_dir: Path) -> Path:
    return out_dir / "checkpoint.pt"


def log_path(out_dir: Path) -> Path:
    return out_dir / "metrics.jsonl"


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def test_partition(test_x, test_y, tasks, n_clients, beta, seed):
    splits = []
    meta = []
    for task_id, classes in enumerate(tasks):
        clients, info = audit.dirichlet_partition(
            test_x, test_y, classes, n_clients, beta, seed=8100 + 100 * seed + task_id
        )
        expected = int(sum(int((test_y == label).sum()) for label in classes))
        got = int(sum(int(client[1].numel()) for client in clients))
        if got != expected:
            raise RuntimeError(f"test partition for task {task_id} has {got} examples, source has {expected}")
        splits.append(clients)
        meta.append(info)
    return splits, meta


def protocol_for(config, train_x, train_y, test_x, test_y, tasks, seed, protocol_dir: Path):
    protocol_dir.mkdir(parents=True, exist_ok=True)
    path = protocol_dir / f"seed{seed}.json"
    tensor_path = protocol_dir / f"seed{seed}.pt"
    if path.exists() and tensor_path.exists():
        meta = json.loads(path.read_text())
        stored = torch.load(tensor_path, weights_only=False)
        if "test_splits" not in stored:
            raise SystemExit(f"protocol {tensor_path} has no client test partition; remove it and rerun")
        return stored["splits"], stored["test_splits"], meta, stored["participation"], stored["batches"]
    splits, meta, participation, batches = audit.build_protocol(
        train_x, train_y, tasks, int(config["clients"]), int(config["participants_per_round"]),
        int(config["rounds_per_task"]), int(config["local_steps"]), int(config["batch_size"]),
        float(config["dirichlet_beta"]), seed,
    )
    test_splits, test_meta = test_partition(
        test_x, test_y, tasks, int(config["clients"]), float(config["dirichlet_beta"]), seed
    )
    serializable = {
        "seed": seed,
        "class_counts": [row["class_counts"] for row in meta],
        "client_index_hashes": [row["client_index_hashes"] for row in meta],
        "test_class_counts": [row["class_counts"] for row in test_meta],
        "test_client_index_hashes": [row["client_index_hashes"] for row in test_meta],
        "participation": participation,
        "empty_clients": [[i for i, client in enumerate(task) if client[0].size(0) < 2] for task in splits],
        "empty_test_clients": [[i for i, client in enumerate(task) if client[1].numel() == 0] for task in test_splits],
    }
    path.write_text(json.dumps(serializable))
    torch.save(
        {"splits": splits, "test_splits": test_splits, "participation": participation, "batches": batches},
        tensor_path,
    )
    return splits, test_splits, serializable, participation, batches


def clone_contributions(contributions):
    copied = []
    for client in contributions:
        client_copy = {}
        for name, items in client.items():
            client_copy[name] = [
                {"task": item["task"], "weight": item["weight"], "factor": item["factor"].detach().cpu().clone()}
                for item in items
            ]
        copied.append(client_copy)
    return copied


def basis_diff(left, right) -> float:
    if left is None and right is None:
        return 0.0
    if left is None or right is None or set(left) != set(right):
        return float("inf")
    return max_state_diff(left, right)


def state_bundle_diff(left: dict, right: dict) -> dict:
    momentum = max(max_state_diff(a, b) for a, b in zip(left["momentum"], right["momentum"]))
    control = max(max_state_diff(a, b) for a, b in zip(left["control"], right["control"]))
    client_bases = max(basis_diff(a, b) for a, b in zip(left["client_bases"], right["client_bases"]))
    return {
        "model": max_state_diff(left["model"], right["model"]),
        "momentum": momentum,
        "control": control,
        "global_control": max_state_diff(left["global_control"], right["global_control"]),
        "client_bases": client_bases,
        "shared_basis": basis_diff(left["shared_basis"], right["shared_basis"]),
        "gradient_shared": basis_diff(left["gradient_shared"], right["gradient_shared"]),
    }


def rewrite_log(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(record) + "\n" for record in records)
    path.write_text(text)


def run_arm(config, arm, seed, train_x, train_y, test_x, test_y, out_dir: Path, protocol_dir: Path, resume: bool, run_kind: str, device: torch.device, halt_after_saves: int | None = None, trace: list | None = None):
    """Checkpoint phases are train_round, task_boundary, and done.

    The boundary that appends this task's gradient factors and resets optimizer
    state is its own phase. A checkpoint at the end of the last round does not
    advance to the next task until that boundary has been saved.
    """
    tasks = tasks_from_config(config)
    splits, test_splits, _meta, participation, batches = protocol_for(
        config, train_x, train_y, test_x, test_y, tasks, seed, protocol_dir
    )
    features = train_x.shape[1] if train_x.ndim == 2 else 0
    model = make_model(config["profile"], int(config["num_classes"]), features, seed).to(device)
    n_clients = int(config["clients"])
    momentum = [zero_state(model) for _ in range(n_clients)]
    control = [zero_state(model) for _ in range(n_clients)]
    global_control = zero_state(model)
    client_bases = [None for _ in range(n_clients)]
    shared_basis = None
    gradient_shared = None
    contributions = empty_contributions(n_clients)
    cursor = {"phase": "train_round", "task_id": 0, "round_id": 0}
    records: list[dict] = []
    cuda_training_steps = 0
    digest = config_digest(config)
    existing = checkpoint_path(out_dir)
    if existing.exists() and not resume:
        raise SystemExit(f"checkpoint exists, pass --resume to continue: {existing}")
    if resume and existing.exists():
        saved = torch.load(existing, weights_only=False)
        if saved["config_sha256"] != digest or saved["arm"] != arm or saved["seed"] != seed:
            raise SystemExit("checkpoint does not match this config, arm, and seed")
        load_state(model, saved["model"])
        momentum = [move_state(item, device) for item in saved["momentum"]]
        control = [move_state(item, device) for item in saved["control"]]
        global_control = move_state(saved["global_control"], device)
        client_bases = [move_basis(item, device) for item in saved["client_bases"]]
        shared_basis = move_basis(saved["shared_basis"], device)
        gradient_shared = cpu_basis(saved["gradient_shared"])
        contributions = saved["contributions"]
        cursor = {"phase": saved["phase"], "task_id": saved["task_id"], "round_id": saved["round_id"]}
        records = list(saved["records"])
        cuda_training_steps = int(saved.get("cuda_training_steps", 0))
        rewrite_log(log_path(out_dir), records)
    elif resume:
        raise SystemExit(f"no checkpoint to resume: {existing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha = float(config["momentum_alpha"])
    lr = float(config["lr"])
    lr_vector = float(config["lr_vector"])
    rank = int(config["rank"])
    cap = int(config["basis_rank_cap"])
    sample_size = int(config["basis_sample"])
    eval_batch = int(config["eval_batch_size"])
    saves = 0

    def capture() -> dict:
        env = environment_report(parameter_device(model), cuda_training_steps)
        payload = {
            "config_sha256": digest,
            "arm": arm,
            "seed": seed,
            "run_kind": run_kind,
            "hypothesis_status": "unverified",
            "is_new_method": False,
            "phase": cursor["phase"],
            "task_id": cursor["task_id"],
            "round_id": cursor["round_id"],
            "model": cpu_state(clone_state(model)),
            "momentum": [cpu_state(item) for item in momentum],
            "control": [cpu_state(item) for item in control],
            "global_control": cpu_state(global_control),
            "client_bases": [cpu_basis(item) for item in client_bases],
            "shared_basis": cpu_basis(shared_basis),
            "gradient_shared": cpu_basis(gradient_shared),
            "contributions": clone_contributions(contributions),
            "records": list(records),
            "cuda_training_steps": cuda_training_steps,
            "parameter_device": parameter_device(model),
            "executed_on_rtx_5080": env["executed_on_rtx_5080"],
        }
        save_checkpoint(existing, payload)
        rewrite_log(log_path(out_dir), records)
        (out_dir / "environment.json").write_text(json.dumps(env, indent=2))
        if trace is not None:
            trace.append(payload)
        return payload

    while cursor["phase"] != "done":
        task_id = cursor["task_id"]
        if cursor["phase"] == "train_round":
            round_id = cursor["round_id"]
            chosen = [int(client_id) for client_id in participation[task_id][round_id]]
            local_residuals = []
            spectral_norms = []
            if chosen:
                snapshot = clone_state(model)
                control_snapshot = clone_named(global_control)
                local_states = []
                increments = []
                for client_id in chosen:
                    features_i, labels_i = splits[task_id][client_id]
                    load_state(model, snapshot)
                    old_control = clone_named(control[client_id])
                    basis = shared_basis if arm in ("shared", "random_matched_rank") else client_bases[client_id]
                    step_ids = batches[task_id][client_id][round_id * int(config["local_steps"]) : (round_id + 1) * int(config["local_steps"])]
                    residual_sum = 0.0
                    norm_sum = 0.0
                    taken = 0
                    for local_index in step_ids:
                        if len(local_index) < 2:
                            continue
                        batch_x = prepare_inputs(features_i[local_index], config["profile"]).to(device)
                        batch_y = labels_i[local_index].to(device)
                        residual, norm = client_step(
                            model, momentum[client_id], control[client_id], control_snapshot,
                            batch_x, batch_y, basis, alpha, lr, lr_vector,
                        )
                        if device.type == "cuda":
                            cuda_training_steps += 1
                        residual_sum += residual
                        norm_sum += norm
                        taken += 1
                    new_control = clone_named(momentum[client_id])
                    increments.append({name: new_control[name] - old_control[name] for name in new_control})
                    control[client_id] = new_control
                    local_states.append(clone_state(model))
                    local_residuals.append(residual_sum / max(taken, 1))
                    spectral_norms.append(norm_sum / max(taken, 1))
                aggregated, global_control = fedmuon_partial_aggregate(
                    snapshot, local_states, control_snapshot, increments, n_clients
                )
                load_state(model, aggregated)
            env = environment_report(parameter_device(model), cuda_training_steps)
            metric = absent_old_task_metric(model, test_splits, chosen, n_clients, task_id, config["profile"], eval_batch)
            record = {
                "task": task_id,
                "round": round_id,
                "participants": chosen,
                "empty_round": not chosen,
                "run_kind": run_kind,
                "hypothesis_status": "unverified",
                "is_new_method": False,
                "parameter_device": parameter_device(model),
                "eval_batch_size": eval_batch,
                "cuda_training_steps": cuda_training_steps,
                "executed_on_rtx_5080": env["executed_on_rtx_5080"],
                "primary_metric": PRIMARY_METRIC,
                "mean_local_residual": None if not local_residuals else sum(local_residuals) / len(local_residuals),
                "mean_step_spectral_norm": None if not spectral_norms else sum(spectral_norms) / len(spectral_norms),
                **metric,
            }
            records.append(record)
            if round_id + 1 == int(config["rounds_per_task"]):
                cursor = {"phase": "task_boundary", "task_id": task_id, "round_id": round_id}
            else:
                cursor = {"phase": "train_round", "task_id": task_id, "round_id": round_id + 1}
        elif cursor["phase"] == "task_boundary":
            append_task_factors(contributions, splits[task_id], model, rank, config["profile"], sample_size, task_id)
            client_bases_cpu = materialize_client_bases(contributions, cap)
            gradient_shared = materialize_shared_basis(contributions, cap)
            if arm == "shared":
                shared_basis = move_basis(gradient_shared, device)
            elif arm == "random_matched_rank":
                shared_basis = move_basis(random_matched_basis(gradient_shared, seed, task_id), device)
            else:
                shared_basis = None
            client_bases = [move_basis(item, device) for item in client_bases_cpu]
            if config.get("reset_state_at_task_boundary", True):
                momentum = [zero_state(model) for _ in range(n_clients)]
                control = [zero_state(model) for _ in range(n_clients)]
                global_control = zero_state(model)
            if task_id + 1 == len(tasks):
                cursor = {"phase": "done", "task_id": task_id + 1, "round_id": 0}
            else:
                cursor = {"phase": "train_round", "task_id": task_id + 1, "round_id": 0}
        else:
            raise RuntimeError(f"unknown phase {cursor['phase']}")
        capture()
        saves += 1
        if halt_after_saves is not None and saves >= halt_after_saves:
            return {
                "arm": arm,
                "seed": seed,
                "run_kind": run_kind,
                "log": str(log_path(out_dir)),
                "stopped_early": True,
                "phase": cursor["phase"],
                "task_id": cursor["task_id"],
                "round_id": cursor["round_id"],
            }
    return {"arm": arm, "seed": seed, "run_kind": run_kind, "log": str(log_path(out_dir)), "stopped_early": False}


def synthetic_problem(seed: int):
    generator = torch.Generator().manual_seed(1000 + seed)
    train_x = torch.rand(80, 8, generator=generator)
    train_y = torch.arange(80) % 4
    test_x = torch.rand(40, 8, generator=generator)
    test_y = torch.arange(40) % 4
    return train_x, train_y, test_x, test_y


def plumbing_config() -> dict:
    return {
        "hypothesis_id": "H-absent",
        "status": "plumbing_not_a_result",
        "profile": "plumbing-mlp",
        "num_classes": 4,
        "classes_per_task": 2,
        "clients": 4,
        "participants_per_round": 2,
        "rounds_per_task": 2,
        "local_steps": 1,
        "batch_size": 8,
        "dirichlet_beta": 0.5,
        "rank": 2,
        "basis_rank_cap": 4,
        "basis_sample": 8,
        "eval_batch_size": 4,
        "lr": 0.05,
        "lr_vector": 0.05,
        "momentum_alpha": 0.1,
        "seeds": [0],
        "arms": ["client_specific", "shared", "random_matched_rank"],
        "lmo": "float64_compact_polar_of_GN",
        "reset_state_at_task_boundary": True,
        "scientific": False,
    }


def load_cifar100_pickle(root: Path):
    """Official CIFAR-100 python pickle layout, used when torchvision cannot import."""
    import pickle
    base = root / "cifar-100-python"
    train_file = base / "train"
    test_file = base / "test"
    if not train_file.exists() or not test_file.exists():
        raise SystemExit(
            f"CIFAR-100 was not found at {root}. Set CIFAR100_ROOT to a directory containing cifar-100-python/. Nothing was invented in its place."
        )
    with train_file.open("rb") as handle:
        train = pickle.load(handle, encoding="bytes")
    with test_file.open("rb") as handle:
        test = pickle.load(handle, encoding="bytes")
    def hwc(raw):
        # Official batches store R, G, B planes. The training path views a flat row as HWC.
        return torch.from_numpy(raw.copy()).view(-1, 3, 32, 32).permute(0, 2, 3, 1).contiguous().view(-1, 3072).float().div_(255.0)
    train_x = hwc(train[b"data"])
    train_y = torch.tensor(train[b"fine_labels"], dtype=torch.long)
    test_x = hwc(test[b"data"])
    test_y = torch.tensor(test[b"fine_labels"], dtype=torch.long)
    if train_x.shape != (50000, 3072) or test_x.shape != (10000, 3072):
        raise RuntimeError((train_x.shape, test_x.shape))
    return train_x, train_y, test_x, test_y


def load_cifar100(root: Path):
    try:
        from torchvision.datasets import CIFAR100
        train = CIFAR100(root=str(root), train=True, download=False)
        test = CIFAR100(root=str(root), train=False, download=False)
    except Exception:
        return load_cifar100_pickle(root)
    train_x = torch.from_numpy(train.data).float().div_(255.0).reshape(len(train.data), -1)
    train_y = torch.tensor(train.targets, dtype=torch.long)
    test_x = torch.from_numpy(test.data).float().div_(255.0).reshape(len(test.data), -1)
    test_y = torch.tensor(test.targets, dtype=torch.long)
    return train_x, train_y, test_x, test_y


def order_invariance_check() -> dict:
    """Fixed batches: client order does not change the synchronous aggregate."""
    generator = torch.Generator().manual_seed(7)
    source = SmallMLP(8, 4)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    n_total = 4
    momentum = []
    control = []
    for client_id in range(n_total):
        client_momentum = zero_state(source)
        client_control = zero_state(source)
        for name in client_momentum:
            client_momentum[name].add_(0.01 * (client_id + 1))
            client_control[name].add_(0.005 * (client_id + 1))
        momentum.append(client_momentum)
        control.append(client_control)
    global_control = zero_state(source)
    for name in global_control:
        global_control[name].add_(0.02)
    batches = [
        (torch.rand(8, 8, generator=generator), torch.randint(0, 4, (8,), generator=generator))
        for _ in range(2)
    ]
    snapshot = clone_state(source)
    control_snapshot = clone_named(global_control)

    def isolated(client_id: int):
        local = SmallMLP(8, 4)
        load_state(local, snapshot)
        client_momentum = clone_named(momentum[client_id])
        client_control = clone_named(control[client_id])
        old = clone_named(client_control)
        client_step(
            local, client_momentum, client_control, control_snapshot,
            batches[client_id][0], batches[client_id][1], None, 0.1, 0.05, 0.05,
        )
        increment = {name: client_momentum[name] - old[name] for name in old}
        return clone_state(local), increment, client_momentum

    first = isolated(0)
    second = isolated(1)
    reference = {}
    for name, value in snapshot.items():
        reference[name] = value * ((n_total - 2) / n_total) + first[0][name] / n_total + second[0][name] / n_total
    reference_control = {}
    for name, value in control_snapshot.items():
        reference_control[name] = value + first[1][name] / n_total + second[1][name] / n_total
    participant_only = {}
    for name in snapshot:
        participant_only[name] = first[0][name] / 2 + second[0][name] / 2

    def run_order(order: list[int]):
        local = SmallMLP(8, 4)
        load_state(local, snapshot)
        client_momentum = [clone_named(item) for item in momentum]
        client_control = [clone_named(item) for item in control]
        local_states = []
        increments = []
        for client_id in order:
            load_state(local, snapshot)
            old = clone_named(client_control[client_id])
            client_step(
                local, client_momentum[client_id], client_control[client_id], control_snapshot,
                batches[client_id][0], batches[client_id][1], None, 0.1, 0.05, 0.05,
            )
            new_control = clone_named(client_momentum[client_id])
            increments.append({name: new_control[name] - old[name] for name in old})
            client_control[client_id] = new_control
            local_states.append(clone_state(local))
        aggregated, updated_control = fedmuon_partial_aggregate(
            snapshot, local_states, control_snapshot, increments, n_total
        )
        return aggregated, updated_control, client_momentum

    forward = run_order([0, 1])
    reverse = run_order([1, 0])
    report = {
        "n_total": n_total,
        "participants": 2,
        "order_parameter_diff": max_state_diff(forward[0], reverse[0]),
        "order_global_control_diff": max_state_diff(forward[1], reverse[1]),
        "order_momentum_diff": max(
            max_state_diff(left, right) for left, right in zip(forward[2], reverse[2])
        ),
        "reference_parameter_diff": max_state_diff(forward[0], reference),
        "reference_control_diff": max_state_diff(forward[1], reference_control),
        "participant_only_average_diff": max_state_diff(forward[0], participant_only),
    }
    tolerance = 1e-5
    if (
        report["order_parameter_diff"] > tolerance
        or report["order_global_control_diff"] > tolerance
        or report["order_momentum_diff"] > tolerance
        or report["reference_parameter_diff"] > tolerance
        or report["reference_control_diff"] > tolerance
        or report["participant_only_average_diff"] < 1e-3
    ):
        raise AssertionError(report)
    return report


def subspace_order_check() -> dict:
    eye = torch.eye(4, dtype=torch.float64)
    factors = [eye[:, index : index + 1] for index in range(3)]
    weights = [1.0, 2.0, 4.0]
    forward = top_invariant_subspace(list(zip(factors, weights)), 2)
    reverse = top_invariant_subspace(list(zip(reversed(factors), reversed(weights))), 2)
    projector_diff = float((projector(forward) - projector(reverse)).abs().max())
    heavy = float((projector(forward) @ eye[:, 2]).norm())
    light = float((projector(forward) @ eye[:, 0]).norm())
    broken = torch.cat(factors, dim=1)[:, :2]
    broken_heavy = float((projector(broken) @ eye[:, 2]).norm())
    accumulated = top_invariant_subspace([(eye[:, :1], 1.0), (eye[:, 1:2], 1.0)], 2)
    overwritten = top_invariant_subspace([(eye[:, 1:2], 1.0)], 2)
    report = {
        "client_order_projector_diff": projector_diff,
        "largest_weight_direction_overlap": heavy,
        "smallest_weight_direction_overlap": light,
        "qr_prefix_overlap_with_largest_weight": broken_heavy,
        "accumulated_task0_overlap": float((projector(accumulated) @ eye[:, 0]).norm()),
        "overwrite_task0_overlap": float((projector(overwritten) @ eye[:, 0]).norm()),
    }
    if (
        projector_diff > 1e-8
        or heavy < 0.99
        or light > 1e-6
        or broken_heavy > 1e-6
        or report["accumulated_task0_overlap"] < 0.99
        or report["overwrite_task0_overlap"] > 1e-6
    ):
        raise AssertionError(report)
    return report


def eval_batch_check() -> dict:
    model = SmallMLP(8, 4)
    seen = []
    original = model.forward

    def wrapped(inputs):
        seen.append((int(inputs.shape[0]), str(inputs.device)))
        return original(inputs)

    model.forward = wrapped
    features = torch.rand(20, 8)
    predicted = argmax_predictions(model, features, "plumbing-mlp", 4)
    if predicted.shape != (20,) or seen != [(4, "cpu")] * 5:
        raise AssertionError(seen)
    return {"chunks": seen, "predictions": int(predicted.shape[0])}


def flag_check() -> dict:
    cases = {
        "cpu_with_5080_visible": executed_on_rtx_5080("cpu", 0, "NVIDIA GeForce RTX 5080"),
        "cuda_device_without_a_step": executed_on_rtx_5080("cuda:0", 0, "NVIDIA GeForce RTX 5080"),
        "cuda_steps_on_5080": executed_on_rtx_5080("cuda:0", 3, "NVIDIA GeForce RTX 5080"),
        "cuda_steps_on_other_gpu": executed_on_rtx_5080("cuda:0", 3, "NVIDIA GeForce RTX 4090"),
    }
    if cases["cpu_with_5080_visible"] or cases["cuda_device_without_a_step"] or cases["cuda_steps_on_other_gpu"] or not cases["cuda_steps_on_5080"]:
        raise AssertionError(cases)
    return cases


def placement_probe(device: torch.device) -> dict:
    model = SmallMLP(8, 4).to(device)
    momentum = zero_state(model)
    control = zero_state(model)
    global_control = zero_state(model)
    basis = {"fc1.weight": torch.eye(8, 2, device=device)}
    batch_x = torch.rand(4, 8, device=device)
    batch_y = torch.randint(0, 4, (4,), device=device)
    client_step(model, momentum, control, global_control, batch_x, batch_y, basis, 0.1, 0.05, 0.05)
    if parameter_device(model) != str(device):
        raise AssertionError(parameter_device(model))
    if momentum["fc1.weight"].device != device or basis["fc1.weight"].device != device:
        raise AssertionError("state left the requested device")
    return {"parameter_device": parameter_device(model), "cuda_training_steps": 1 if device.type == "cuda" else 0}


def checkpoint_cases(config, train_x, train_y, test_x, test_y, work: Path) -> dict:
    full_trace = []
    run_arm(
        config, "shared", 0, train_x, train_y, test_x, test_y,
        work / "full", work / "protocol", False, "self_check", torch.device("cpu"), trace=full_trace,
    )
    labels = {1: "within_task", 2: "end_of_task_before_boundary", 3: "after_task_boundary"}
    expected_cursor = {1: ("train_round", 0, 1), 2: ("task_boundary", 0, 1), 3: ("train_round", 1, 0)}
    report = {}
    for halt, name in labels.items():
        cursor = (full_trace[halt - 1]["phase"], full_trace[halt - 1]["task_id"], full_trace[halt - 1]["round_id"])
        if cursor != expected_cursor[halt]:
            raise AssertionError((name, cursor))
        run_arm(
            config, "shared", 0, train_x, train_y, test_x, test_y,
            work / name, work / "protocol", False, "self_check", torch.device("cpu"), halt_after_saves=halt,
        )
        stopped = torch.load(work / name / "checkpoint.pt", weights_only=False)
        stopped_diff = state_bundle_diff(stopped, full_trace[halt - 1])
        # A crash could have appended a duplicate line after the checkpoint. Resume must rewrite the log.
        with log_path(work / name).open("a") as handle:
            handle.write(json.dumps({"task": 99, "round": 99, "duplicate": True}) + "\n")
        run_arm(
            config, "shared", 0, train_x, train_y, test_x, test_y,
            work / name, work / "protocol", True, "self_check", torch.device("cpu"),
        )
        finished = torch.load(work / name / "checkpoint.pt", weights_only=False)
        final_diff = state_bundle_diff(finished, full_trace[-1])
        lines = [json.loads(line) for line in log_path(work / name).read_text().splitlines() if line]
        keys = [(row["task"], row["round"]) for row in lines]
        full_lines = [json.loads(line) for line in log_path(work / "full").read_text().splitlines() if line]
        if keys != [(row["task"], row["round"]) for row in full_lines] or len(keys) != len(set(keys)):
            raise AssertionError({"name": name, "keys": keys})
        if any(value > 1e-6 for value in stopped_diff.values()) or any(value > 1e-6 for value in final_diff.values()):
            raise AssertionError({"name": name, "stopped": stopped_diff, "final": final_diff})
        scored = [row for row in lines if row["task"] > 0 and row["scored_absent_clients"]]
        if not scored:
            raise AssertionError({"name": name, "metric": "no absent client with its own old-task test examples"})
        for row in lines:
            scored_ids = {item["client"] for item in row["scored_absent_clients"]}
            if scored_ids & set(row["participants"]) or not scored_ids <= set(row["absent_clients"]):
                raise AssertionError({"name": name, "row": row["task"], "round": row["round"]})
        report[name] = {"stopped": stopped_diff, "final": final_diff, "log_rows": len(lines), "phase_at_halt": stopped["phase"]}
    return report


def self_check(destination: Path, device: torch.device) -> dict:
    checks = foundation_checks()
    placement = placement_probe(device)
    env = environment_report(placement["parameter_device"], placement["cuda_training_steps"])
    model = make_model("cifar100-resnet18-groupnorm", 100, 0, 0)
    images = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([1, 2])
    logits = model(images)
    if tuple(logits.shape) != (2, 100):
        raise AssertionError(logits.shape)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    weight = model.conv1.weight.grad
    direction = audit.onesided_lmo(weight.reshape(weight.shape[0], -1), None)
    if direction.shape != (64, 27):
        raise AssertionError(direction.shape)
    registered = load_config(ROOT / "configs" / "h_absent.json")
    if registered["primary_metric"] != PRIMARY_METRIC or registered["reject_if"] != REJECT_IF:
        raise AssertionError("config metric does not match the code")
    if registered.get("is_new_method", True) or "activation subspace" in json.dumps(registered):
        raise AssertionError("config still calls the basis an activation subspace or a new method")
    if "void" not in registered.get("voided_evidence", "").lower() or "0.065" in registered["statement"]:
        raise AssertionError("voided MLP evidence must be named as void and must not appear in the hypothesis statement")
    config = plumbing_config()
    work = destination / "plumbing"
    if work.exists():
        for child in sorted(work.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    train_x, train_y, test_x, test_y = synthetic_problem(0)
    order_report = order_invariance_check()
    subspace_report = subspace_order_check()
    batch_report = eval_batch_check()
    flags = flag_check()
    resume_report = checkpoint_cases(config, train_x, train_y, test_x, test_y, work)
    protocol = json.loads((work / "protocol" / "seed0.json").read_text())
    if protocol["client_index_hashes"] == protocol["test_client_index_hashes"]:
        raise AssertionError("train and test client partitions are not independent")
    covered = sum(sum(counts.values()) for counts in protocol["test_class_counts"][0])
    if covered != int((test_y < 2).sum()):
        raise AssertionError((covered, int((test_y < 2).sum())))
    report = {
        "scientific": False,
        "is_new_method": False,
        "hypothesis_status": "unverified",
        "executed_on_rtx_5080": env["executed_on_rtx_5080"],
        "environment": env,
        "placement": placement,
        "flag_cases": flags,
        "foundation": checks,
        "order_invariance": order_report,
        "shared_subspace": subspace_report,
        "eval_batches": batch_report,
        "resume": resume_report,
        "resnet_forward_shape": [2, 100],
        "conv_polar_shape": list(direction.shape),
        "primary_metric": PRIMARY_METRIC,
    }
    trained_on_cuda = placement["cuda_training_steps"] > 0 and str(placement["parameter_device"]).startswith("cuda")
    if report["executed_on_rtx_5080"] and not trained_on_cuda:
        raise AssertionError("executed_on_rtx_5080 was set without a CUDA training step")
    if device.type != "cuda" and report["executed_on_rtx_5080"]:
        raise AssertionError(report)
    if trained_on_cuda and not report["executed_on_rtx_5080"]:
        used = env.get("used_cuda_device_name") or ""
        if "5080" in used:
            raise AssertionError("a 5080 training step was not recorded")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "self_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "self_check": "passed",
        "scientific": False,
        "is_new_method": False,
        "executed_on_rtx_5080": report["executed_on_rtx_5080"],
        "parameter_device": placement["parameter_device"],
        "log": str(destination / "self_check.json"),
    }), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "h_absent.json")
    parser.add_argument("--arm", choices=["client_specific", "shared", "random_matched_rank"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "hypothesis_logs")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"{args.device} was requested but CUDA is not available")
    if args.self_check:
        self_check(args.out, device)
        return
    if args.arm is None or args.seed is None:
        raise SystemExit("a run needs --arm, --seed, and --device; use --self-check for the plumbing test")
    config = load_config(args.config)
    if args.seed not in config["seeds"] or args.arm not in config["arms"]:
        raise SystemExit("arm or seed is outside the fixed config")
    if config.get("is_new_method", True):
        raise SystemExit("this entry is a diagnostic hypothesis, not a new method")
    import os
    root = Path(os.environ.get(config["data_root_env"], ""))
    train_x, train_y, test_x, test_y = load_cifar100(root)
    digest = config_digest(config)
    out_dir = args.out / digest[:12] / args.arm / f"seed{args.seed}"
    run_arm(
        config, args.arm, args.seed, train_x, train_y, test_x, test_y,
        out_dir, args.out / digest[:12] / "protocol", args.resume, config.get("run_kind", "preregistered"), device,
    )




if __name__ == "__main__":
    main()
