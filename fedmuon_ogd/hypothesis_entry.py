"""Falsification entry for hypothesis H-absent. The hypothesis is not a result.

Each round freezes the model and the global control. Every selected client trains
from that snapshot. The server then applies the partial-participation update once:
X <- ((N-S)/N) X + (1/N) sum_{i in S} X_i, and
C <- C + (1/N) sum_{i in S} (C_i_new - C_i_old).
N is the number of clients and S is the number of participants.

H-absent says that, under partial participation, old-class argmax accuracy of a
client who did not participate is lower when each client protects its own
subspace than when all clients share one subspace of the same rank. The arms
are measurement conditions. They are not a new method.

Excluded by abstract, with the full-text boundary still unread:
SSF (arXiv:2604.25467), subspace Muon residual correction (arXiv:2610.19844),
and Muon without a protection constraint versus dedicated continual learning
(arXiv:2609.24678).

This process records the device it actually sees. It does not claim a run on
an RTX 5080 unless the device name contains 5080.

    python3 fedmuon_ogd/hypothesis_entry.py --self-check
    python3 fedmuon_ogd/hypothesis_entry.py --config fedmuon_ogd/configs/h_absent.json --arm shared --seed 0
    python3 fedmuon_ogd/hypothesis_entry.py --config fedmuon_ogd/configs/h_absent.json --arm shared --seed 0 --resume
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


def environment_report() -> dict:
    cuda = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if cuda else "cpu"
    capability = list(torch.cuda.get_device_capability(0)) if cuda else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "device_name": name,
        "device_capability": capability,
        "executed_on_rtx_5080": cuda and "5080" in name,
        "requirements": {
            "paper_profile": "Python 3.10+, a CUDA build of PyTorch that supports the GPU it runs on, torchvision, and CIFAR-100 at the path in CIFAR100_ROOT.",
            "memory": "ResNet-18 with 20 client momentum and control copies is sized for a 16 GB GPU.",
            "this_session": "No RTX 5080 connection is available here. A run on CPU or another GPU must keep executed_on_rtx_5080 false.",
        },
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


def top_right(matrix: torch.Tensor, rank: int) -> torch.Tensor | None:
    if matrix.numel() == 0 or float(matrix.norm()) < 1e-8:
        return None
    _left, _values, right = torch.linalg.svd(matrix.float(), full_matrices=False)
    kept = min(rank, right.shape[0])
    return right[:kept, :].T.contiguous()


def merge_bases(pieces: list[torch.Tensor], rank: int) -> torch.Tensor:
    factor, _ = torch.linalg.qr(torch.cat(pieces, dim=1), mode="reduced")
    return factor[:, : min(rank, factor.shape[1])].contiguous()


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
            parameter.copy_(state[name])


def prepare_inputs(inputs: torch.Tensor, profile: str) -> torch.Tensor:
    if profile == "cifar100-resnet18-groupnorm":
        return inputs.view(-1, 32, 32, 3).permute(0, 3, 1, 2).contiguous()
    return inputs


def client_step(model, momentum, control, global_control, batch_x, batch_y, basis, alpha, lr, lr_vector):
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


def average_gradient(model, inputs, labels) -> dict[str, torch.Tensor]:
    model.train()
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(inputs), labels)
    loss.backward()
    return {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}


def update_bases(model, clients, bases, rank, mode, seed, task_id, profile):
    collected: dict[str, list[torch.Tensor]] = {}
    updated = []
    for client_id, (features, labels) in enumerate(clients):
        if features.size(0) < 2:
            updated.append(bases[client_id])
            continue
        sample = min(32, int(features.size(0)))
        gradients = average_gradient(
            model, prepare_inputs(features[:sample], profile), labels[:sample]
        )
        local = {}
        for name, gradient in gradients.items():
            view = matrix_view(gradient)
            if view is None:
                continue
            local[name] = top_right(view, rank)
            if local[name] is not None:
                collected.setdefault(name, []).append(local[name])
        updated.append(local)
    shared = None
    if mode == "shared":
        shared = {name: merge_bases(pieces, rank) for name, pieces in collected.items() if pieces}
    elif mode == "random_matched_rank":
        generator = torch.Generator().manual_seed(9000 + 13 * seed + task_id)
        shared = {}
        for name, parameter in model.named_parameters():
            view = matrix_view(parameter)
            if view is None:
                continue
            raw = torch.randn(view.shape[1], rank, generator=generator)
            factor, _ = torch.linalg.qr(raw, mode="reduced")
            shared[name] = factor[:, : min(rank, factor.shape[1])].contiguous()
    return updated, shared


def evaluate(model, features, labels, classes, profile) -> float | None:
    mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    for label in classes:
        mask |= labels == label
    if int(mask.sum()) == 0:
        return None
    model.eval()
    with torch.no_grad():
        predicted = model(prepare_inputs(features[mask], profile)).argmax(dim=1)
    return float((predicted.cpu() == labels[mask]).float().mean())


def worst_client(model, test_x, test_y, counts, classes, profile) -> float | None:
    scores = []
    for client_counts in counts:
        held = [label for label in classes if client_counts.get(str(label), client_counts.get(label, 0)) > 0]
        if not held:
            continue
        score = evaluate(model, test_x, test_y, held, profile)
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    return min(scores)


def checkpoint_path(out_dir: Path) -> Path:
    return out_dir / "checkpoint.pt"


def log_path(out_dir: Path) -> Path:
    return out_dir / "metrics.jsonl"


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def protocol_for(config, train_x, train_y, tasks, seed, protocol_dir: Path):
    protocol_dir.mkdir(parents=True, exist_ok=True)
    path = protocol_dir / f"seed{seed}.json"
    tensor_path = protocol_dir / f"seed{seed}.pt"
    if path.exists() and tensor_path.exists():
        meta = json.loads(path.read_text())
        stored = torch.load(tensor_path, weights_only=False)
        return stored["splits"], meta, stored["participation"], stored["batches"]
    splits, meta, participation, batches = audit.build_protocol(
        train_x, train_y, tasks, int(config["clients"]), int(config["participants_per_round"]),
        int(config["rounds_per_task"]), int(config["local_steps"]), int(config["batch_size"]),
        float(config["dirichlet_beta"]), seed,
    )
    serializable = {
        "seed": seed,
        "class_counts": [row["class_counts"] for row in meta],
        "client_index_hashes": [row["client_index_hashes"] for row in meta],
        "participation": participation,
        "empty_clients": [[i for i, client in enumerate(task) if client[0].size(0) < 2] for task in splits],
    }
    path.write_text(json.dumps(serializable))
    torch.save({"splits": splits, "participation": participation, "batches": batches}, tensor_path)
    return splits, serializable, participation, batches


def run_arm(config, arm, seed, train_x, train_y, test_x, test_y, out_dir: Path, protocol_dir: Path, resume: bool, scientific: bool, stop_after_records: int | None = None):
    tasks = tasks_from_config(config)
    splits, meta, participation, batches = protocol_for(config, train_x, train_y, tasks, seed, protocol_dir)
    features = train_x.shape[1] if train_x.ndim == 2 else 0
    model = make_model(config["profile"], int(config["num_classes"]), features, seed)
    momentum = [zero_state(model) for _ in range(int(config["clients"]))]
    control = [zero_state(model) for _ in range(int(config["clients"]))]
    global_control = zero_state(model)
    bases = [None for _ in range(int(config["clients"]))]
    shared = None
    start_task = 0
    start_round = 0
    digest = config_digest(config)
    existing = checkpoint_path(out_dir)
    if existing.exists() and not resume:
        raise SystemExit(f"checkpoint exists, pass --resume to continue: {existing}")
    if resume and existing.exists():
        saved = torch.load(existing, weights_only=False)
        if saved["config_sha256"] != digest or saved["arm"] != arm or saved["seed"] != seed:
            raise SystemExit("checkpoint does not match this config, arm, and seed")
        load_state(model, saved["model"])
        momentum = saved["momentum"]
        control = saved["control"]
        global_control = saved["global_control"]
        bases = saved["bases"]
        shared = saved["shared"]
        start_task = saved["next_task"]
        start_round = saved["next_round"]
    elif resume:
        raise SystemExit(f"no checkpoint to resume: {existing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    records_written = len(log_path(out_dir).read_text().splitlines()) if log_path(out_dir).exists() else 0
    env = environment_report()
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))
    alpha = float(config["momentum_alpha"])
    lr = float(config["lr"])
    lr_vector = float(config["lr_vector"])
    rank = int(config["rank"])
    n_clients = int(config["clients"])
    for task_id in range(start_task, len(tasks)):
        round_start = start_round if task_id == start_task else 0
        for round_id in range(round_start, int(config["rounds_per_task"])):
            chosen = list(participation[task_id][round_id])
            local_residuals = []
            spectral_norms = []
            if not chosen:
                record = {"task": task_id, "round": round_id, "participants": [], "empty_round": True, "scientific": scientific}
                with log_path(out_dir).open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
            else:
                snapshot = clone_state(model)
                control_snapshot = clone_named(global_control)
                local_states = []
                increments = []
                for client_id in chosen:
                    features_i, labels_i = splits[task_id][client_id]
                    load_state(model, snapshot)
                    old_control = clone_named(control[client_id])
                    basis = shared if arm in ("shared", "random_matched_rank") else bases[client_id]
                    step_ids = batches[task_id][client_id][round_id * int(config["local_steps"]) : (round_id + 1) * int(config["local_steps"])]
                    residual_sum = 0.0
                    norm_sum = 0.0
                    taken = 0
                    for local_index in step_ids:
                        if len(local_index) < 2:
                            continue
                        batch_x = prepare_inputs(features_i[local_index], config["profile"])
                        residual, norm = client_step(
                            model, momentum[client_id], control[client_id], control_snapshot,
                            batch_x, labels_i[local_index], basis, alpha, lr, lr_vector,
                        )
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
                current = evaluate(model, test_x, test_y, tasks[task_id], config["profile"])
                old_scores = [
                    evaluate(model, test_x, test_y, task, config["profile"])
                    for task in tasks[:task_id]
                ]
                old_scores = [score for score in old_scores if score is not None]
                record = {
                    "task": task_id,
                    "round": round_id,
                    "participants": chosen,
                    "empty_round": False,
                    "scientific": scientific,
                    "executed_on_rtx_5080": env["executed_on_rtx_5080"],
                    "device_name": env["device_name"],
                    "mean_local_residual": sum(local_residuals) / len(local_residuals),
                    "mean_step_spectral_norm": sum(spectral_norms) / len(spectral_norms),
                    "current_task_argmax_acc": current,
                    "old_task_argmax_acc": None if not old_scores else sum(old_scores) / len(old_scores),
                    "worst_client_old_acc": None if task_id == 0 else worst_client(
                        model, test_x, test_y, meta["class_counts"][task_id - 1], tasks[task_id - 1], config["profile"]
                    ),
                }
                with log_path(out_dir).open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
            records_written += 1
            next_round = 0 if round_id + 1 == int(config["rounds_per_task"]) else round_id + 1
            next_task = task_id + 1 if next_round == 0 else task_id
            save_checkpoint(existing, {
                "config_sha256": digest,
                "arm": arm,
                "seed": seed,
                "scientific": scientific,
                "next_task": next_task,
                "next_round": next_round,
                "model": clone_state(model),
                "momentum": momentum,
                "control": control,
                "global_control": global_control,
                "bases": bases,
                "shared": shared,
            })
            if stop_after_records is not None and records_written >= stop_after_records:
                return {"arm": arm, "seed": seed, "scientific": scientific, "log": str(log_path(out_dir)), "stopped_early": True}
        bases, shared = update_bases(model, splits[task_id], bases, rank, arm, seed, task_id, config["profile"])
        if config.get("reset_state_at_task_boundary", True):
            momentum = [zero_state(model) for _ in range(n_clients)]
            control = [zero_state(model) for _ in range(n_clients)]
            global_control = zero_state(model)
    return {"arm": arm, "seed": seed, "scientific": scientific, "log": str(log_path(out_dir))}


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
        "lr": 0.05,
        "lr_vector": 0.05,
        "momentum_alpha": 0.1,
        "seeds": [0],
        "arms": ["client_specific", "shared", "random_matched_rank"],
        "lmo": "float64_compact_polar_of_GN",
        "reset_state_at_task_boundary": True,
        "scientific": False,
    }


def load_cifar100(root: Path):
    try:
        from torchvision.datasets import CIFAR100
    except ImportError as exc:
        raise SystemExit("the CIFAR-100 profile requires torchvision") from exc
    try:
        train = CIFAR100(root=str(root), train=True, download=False)
        test = CIFAR100(root=str(root), train=False, download=False)
    except Exception as exc:
        raise SystemExit(f"CIFAR-100 was not found at {root}. Set CIFAR100_ROOT on the machine that will run the hypothesis. Nothing was invented in its place.") from exc
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


def self_check(destination: Path) -> dict:
    checks = foundation_checks()
    env = environment_report()
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
    config = plumbing_config()
    work = destination / "plumbing"
    if work.exists():
        for child in work.rglob("*"):
            if child.is_file():
                child.unlink()
    train_x, train_y, test_x, test_y = synthetic_problem(0)
    order_report = order_invariance_check()
    run_arm(config, "client_specific", 0, train_x, train_y, test_x, test_y, work / "uninterrupted", work / "protocol", False, False)
    uninterrupted = torch.load(work / "uninterrupted" / "checkpoint.pt", weights_only=False)
    first = run_arm(config, "client_specific", 0, train_x, train_y, test_x, test_y, work / "partial", work / "protocol", False, False, stop_after_records=1)
    saved = torch.load(work / "partial" / "checkpoint.pt", weights_only=False)
    if saved["next_task"] != 0 or saved["next_round"] != 1:
        raise AssertionError((saved["next_task"], saved["next_round"]))
    continued = run_arm(config, "client_specific", 0, train_x, train_y, test_x, test_y, work / "partial", work / "protocol", True, False)
    final = torch.load(work / "partial" / "checkpoint.pt", weights_only=False)
    lines = (work / "partial" / "metrics.jsonl").read_text().strip().splitlines()
    if len(lines) < 2 or final["scientific"] or json.loads(lines[0])["scientific"]:
        raise AssertionError("plumbing log was marked as a scientific result")
    resume_diff = max(
        max_state_diff(uninterrupted["model"], final["model"]),
        max_state_diff(uninterrupted["global_control"], final["global_control"]),
        max(max_state_diff(left, right) for left, right in zip(uninterrupted["momentum"], final["momentum"])),
        max(max_state_diff(left, right) for left, right in zip(uninterrupted["control"], final["control"])),
    )
    report = {
        "scientific": False,
        "executed_on_rtx_5080": env["executed_on_rtx_5080"],
        "environment": env,
        "foundation": checks,
        "order_invariance": order_report,
        "resnet_forward_shape": [2, 100],
        "conv_polar_shape": list(direction.shape),
        "conv_polar_spectral_norm": spectral_norm(direction),
        "plumbing_rounds_logged": len(lines),
        "resume_state_diff": resume_diff,
        "resume_matches_uninterrupted": resume_diff <= 1e-6 and continued["arm"] == first["arm"],
        "hypothesis_status": "unverified",
    }
    if not report["resume_matches_uninterrupted"]:
        raise AssertionError(report)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "self_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"self_check": "passed", "scientific": False, "executed_on_rtx_5080": False, "log": str(destination / "self_check.json")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "h_absent.json")
    parser.add_argument("--arm", choices=["client_specific", "shared", "random_matched_rank"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "hypothesis_logs")
    args = parser.parse_args()
    if args.self_check:
        self_check(args.out)
        return
    if args.arm is None or args.seed is None:
        raise SystemExit("a paper run needs --arm and --seed; use --self-check for the plumbing test")
    config = load_config(args.config)
    if args.seed not in config["seeds"] or args.arm not in config["arms"]:
        raise SystemExit("arm or seed is outside the fixed config")
    import os
    root = Path(os.environ.get(config["data_root_env"], ""))
    train_x, train_y, test_x, test_y = load_cifar100(root)
    digest = config_digest(config)
    out_dir = args.out / digest[:12] / args.arm / f"seed{args.seed}"
    run_arm(
        config, args.arm, args.seed, train_x, train_y, test_x, test_y,
        out_dir, args.out / digest[:12] / "protocol", args.resume, True,
    )


if __name__ == "__main__":
    main()
