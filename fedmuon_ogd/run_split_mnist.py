"""Federated continual learning: FedMuon with and without an OGD constraint.

Split-MNIST, class-incremental, one global model. Each task is trained with
FedMuon (bias-corrected spectral LMO). Two ways of adding OGD are compared
against plain FedMuon:

* proj: remove the protected bilinear component, then apply Muon.
  G <- G - U (U^T G V) V^T, then the usual Newton-Schulz step.
* dual: Muon-OGD. Solve the spectral-norm ball step subject to U^T Delta V = 0
  by a few dual iterations, then take the matrix sign of the corrected matrix.

Protected bases are the leading singular vectors of the averaged client
gradient at the end of each task. No replay buffer is kept.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch
import torch.nn.functional as F

TRAIN_IMG = Path("/tmp/mnist/train-images-idx3-ubyte")
TRAIN_LBL = Path("/tmp/mnist/train-labels-idx1-ubyte")
TEST_IMG = Path("/workspace/MNIST Merge/work/data/MNIST/raw/t10k-images-idx3-ubyte")
TEST_LBL = Path("/workspace/MNIST Merge/work/data/MNIST/raw/t10k-labels-idx1-ubyte")

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
MATRIX_NAMES = ("fc1.weight", "fc2.weight")
BIAS_NAMES = ("fc1.bias", "fc2.bias")


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
    """Approximate the polar factor UV^T of a gradient matrix."""
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


def qr_basis(basis: torch.Tensor, k_max: int) -> torch.Tensor:
    q, r = torch.linalg.qr(basis, mode="reduced")
    energy = r.abs().diagonal()
    keep = energy > 1e-4
    q = q[:, keep]
    if q.size(1) > k_max:
        q = q[:, :k_max]
    return q


def merge_basis(old: torch.Tensor | None, new: torch.Tensor, k_max: int) -> torch.Tensor:
    stacked = new if old is None or old.numel() == 0 else torch.cat([old, new], dim=1)
    return qr_basis(stacked, k_max)


def project_out(grad: torch.Tensor, left: torch.Tensor | None, right: torch.Tensor | None) -> torch.Tensor:
    if left is None or right is None or left.numel() == 0 or right.numel() == 0:
        return grad
    return grad - left @ (left.T @ grad @ right) @ right.T


def dual_correct(
    grad: torch.Tensor,
    left: torch.Tensor | None,
    right: torch.Tensor | None,
    lambdas: torch.Tensor | None,
    eta_dual: float,
    inner: int,
    ns_steps: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Matrix sign of the spectral step subject to U^T Delta V = 0.

    The matrix sign is unchanged by positive rescaling, so the dual variable is
    fit to the normalized gradient. Each candidate step is accepted only when it
    decreases the nuclear-norm dual objective.
    """
    if left is None or right is None or left.numel() == 0 or right.numel() == 0:
        return newton_schulz(grad, ns_steps), None
    scale = torch.linalg.norm(grad).clamp_min(1e-8)
    normalized = grad / scale
    k_left, k_right = left.size(1), right.size(1)
    if lambdas is None or lambdas.shape != (k_left, k_right):
        lambdas = torch.zeros(k_left, k_right)
    signed = newton_schulz(normalized + left @ lambdas @ right.T, ns_steps)
    for _ in range(inner):
        alignment = left.T @ signed @ right
        if float(alignment.abs().mean()) < 1e-3:
            break
        current = normalized + left @ lambdas @ right.T
        base = torch.linalg.svdvals(current).sum()
        step = eta_dual
        accepted = False
        for _attempt in range(6):
            trial = lambdas - step * alignment
            candidate = normalized + left @ trial @ right.T
            value = torch.linalg.svdvals(candidate).sum()
            if float(value) < float(base) - 1e-6:
                lambdas = trial
                signed = newton_schulz(candidate, ns_steps)
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    return signed, lambdas


def init_model(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    params = {
        "fc1.weight": torch.empty(128, 784),
        "fc1.bias": torch.zeros(128),
        "fc2.weight": torch.empty(10, 128),
        "fc2.bias": torch.zeros(10),
    }
    torch.nn.init.kaiming_uniform_(params["fc1.weight"], a=5**0.5, generator=g)
    torch.nn.init.kaiming_uniform_(params["fc2.weight"], a=5**0.5, generator=g)
    for name in params:
        params[name] = params[name].contiguous()
    return params


def clone_params(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in params.items()}


def zeros_like_params(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(value) for name, value in params.items()}


def forward(params: dict[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
    hidden = F.linear(inputs, params["fc1.weight"], params["fc1.bias"])
    hidden = F.relu(hidden)
    return F.linear(hidden, params["fc2.weight"], params["fc2.bias"])


def batch_grad(params: dict[str, torch.Tensor], inputs: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    leaves = {name: value.detach().requires_grad_(True) for name, value in params.items()}
    loss = F.cross_entropy(forward(leaves, inputs), labels)
    grads = torch.autograd.grad(loss, list(leaves.values()))
    return {name: grad.detach() for name, grad in zip(leaves, grads)}


def iterate_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int, generator: torch.Generator):
    n = x.size(0)
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        if idx.numel() < 2:
            continue
        yield x[idx], y[idx]


def partition_task(
    images: torch.Tensor,
    labels: torch.Tensor,
    classes: tuple[int, int],
    n_clients: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    mask = (labels == classes[0]) | (labels == classes[1])
    x = images[mask]
    y = labels[mask]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(x.size(0), generator=g)
    x, y = x[order], y[order]
    shards = []
    for client in range(n_clients):
        shards.append((x[client::n_clients].contiguous(), y[client::n_clients].contiguous()))
    return shards


def evaluate_task(params: dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor, classes: tuple[int, int]) -> float:
    mask = (y == classes[0]) | (y == classes[1])
    inputs = x[mask]
    labels = y[mask]
    correct = 0
    total = inputs.size(0)
    with torch.no_grad():
        for start in range(0, total, 1024):
            logits = forward(params, inputs[start : start + 1024])
            correct += int((logits.argmax(dim=1) == labels[start : start + 1024]).sum())
    return correct / max(total, 1)


def average_client_gradient(
    params: dict[str, torch.Tensor],
    clients: list[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    total = zeros_like_params(params)
    count = 0
    for x, y in clients:
        for start in range(0, x.size(0), batch_size):
            xb, yb = x[start : start + batch_size], y[start : start + batch_size]
            if xb.size(0) < 2:
                continue
            grads = batch_grad(params, xb, yb)
            weight = xb.size(0)
            for name in total:
                total[name].add_(grads[name], alpha=weight)
            count += weight
    for name in total:
        total[name].div_(max(count, 1))
    return total


def update_subspace(
    subspace: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]],
    gradient: dict[str, torch.Tensor],
    rank_per_task: int,
    k_max: int,
) -> dict[str, tuple[torch.Tensor | None, torch.Tensor | None]]:
    updated = {}
    for name in MATRIX_NAMES:
        matrix = gradient[name]
        left_full, _, right_full = torch.linalg.svd(matrix, full_matrices=False)
        rank = min(rank_per_task, left_full.size(1), right_full.size(0))
        left_new = left_full[:, :rank]
        right_new = right_full[:rank, :].T
        cap = min(k_max, matrix.size(0), matrix.size(1))
        old_left, old_right = subspace.get(name, (None, None))
        updated[name] = (merge_basis(old_left, left_new, cap), merge_basis(old_right, right_new, cap))
    return updated


def local_train(
    params: dict[str, torch.Tensor],
    momentum: dict[str, torch.Tensor],
    control: dict[str, torch.Tensor],
    global_control: dict[str, torch.Tensor],
    data: tuple[torch.Tensor, torch.Tensor],
    subspace: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]],
    lambdas: dict[str, torch.Tensor | None],
    method: str,
    steps: int,
    batch_size: int,
    alpha: float,
    lr: float,
    lr_bias: float,
    eta_dual: float,
    inner: int,
    ns_steps: int,
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor | None], float]:
    x, y = data
    residual = 0.0
    residual_count = 0
    taken = 0
    # Cycle the local data if the client is smaller than steps * batch.
    stored = list(iterate_batches(x, y, batch_size, generator))
    if not stored:
        return params, momentum, lambdas, 0.0
    cursor = 0
    while taken < steps:
        xb, yb = stored[cursor % len(stored)]
        cursor += 1
        grads = batch_grad(params, xb, yb)
        for name in MATRIX_NAMES:
            momentum[name].mul_(1 - alpha).add_(grads[name], alpha=alpha)
            corrected = momentum[name] - control[name] + global_control[name]
            left, right = subspace.get(name, (None, None))
            if method == "fedmuon":
                direction = newton_schulz(corrected, ns_steps)
            elif method == "proj":
                direction = newton_schulz(project_out(corrected, left, right), ns_steps)
            elif method == "dual":
                direction, lambdas[name] = dual_correct(
                    corrected, left, right, lambdas.get(name), eta_dual, inner, ns_steps
                )
            else:
                raise ValueError(method)
            params[name].add_(direction, alpha=-lr)
            if left is not None and right is not None and left.numel() and right.numel():
                residual += float((left.T @ direction @ right).abs().mean())
                residual_count += 1
        for name in BIAS_NAMES:
            momentum[name].mul_(1 - alpha).add_(grads[name], alpha=alpha)
            corrected = momentum[name] - control[name] + global_control[name]
            params[name].add_(corrected, alpha=-lr_bias)
        taken += 1
    return params, momentum, lambdas, residual / max(residual_count, 1)


def aggregate(global_params, global_control, client_params, client_controls, old_controls):
    n = len(client_params)
    new_params = zeros_like_params(global_params)
    for params in client_params:
        for name in new_params:
            new_params[name].add_(params[name], alpha=1.0 / n)
    new_control = clone_params(global_control)
    for new_c, old_c in zip(client_controls, old_controls):
        for name in new_control:
            new_control[name].add_(new_c[name] - old_c[name], alpha=1.0 / n)
    return new_params, new_control


def run_method(args, method: str, seed: int, train, test_x, test_y, init) -> dict:
    g = torch.Generator().manual_seed(1000 + seed)
    params = clone_params(init)
    n_clients = len(train[0])
    momentum = [zeros_like_params(params) for _ in range(n_clients)]
    control = [zeros_like_params(params) for _ in range(n_clients)]
    global_control = zeros_like_params(params)
    lambdas = [{} for _ in range(n_clients)]
    subspace: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {}
    history = []
    for task_id, clients in enumerate(train):
        for _round in range(args.rounds):
            updated_params = []
            updated_control = []
            old_controls = []
            residuals = []
            for client_id, data in enumerate(clients):
                local = clone_params(params)
                old_controls.append(clone_params(control[client_id]))
                local, momentum[client_id], lambdas[client_id], residual = local_train(
                    local,
                    momentum[client_id],
                    control[client_id],
                    global_control,
                    data,
                    subspace,
                    lambdas[client_id],
                    method,
                    args.local_steps,
                    args.batch_size,
                    args.alpha,
                    args.lr,
                    args.lr_bias,
                    args.eta_dual,
                    args.inner,
                    args.ns_steps,
                    g,
                )
                control[client_id] = clone_params(momentum[client_id])
                updated_params.append(local)
                updated_control.append(control[client_id])
                residuals.append(residual)
            params, global_control = aggregate(params, global_control, updated_params, updated_control, old_controls)
        task_acc = [evaluate_task(params, test_x, test_y, classes) for classes in TASKS[: task_id + 1]]
        history.append(
            {
                "task": task_id,
                "per_task_acc": task_acc,
                "average_acc": sum(task_acc) / len(task_acc),
                "current_acc": task_acc[-1],
                "constraint_residual": sum(residuals) / len(residuals),
            }
        )
        print(
            f"{method} seed={seed} after task {task_id}: "
            f"AA={history[-1]['average_acc']:.3f} current={task_acc[-1]:.3f} "
            f"residual={history[-1]['constraint_residual']:.4f}",
            flush=True,
        )
        if method != "fedmuon":
            grad = average_client_gradient(params, clients, args.batch_size)
            subspace = update_subspace(subspace, grad, args.rank_per_task, args.k_max)
            for client_id in range(n_clients):
                lambdas[client_id] = {}
        for client_id in range(n_clients):
            momentum[client_id] = zeros_like_params(params)
            control[client_id] = zeros_like_params(params)
        global_control = zeros_like_params(params)
    final = history[-1]["per_task_acc"]
    after_trained = [history[i]["current_acc"] for i in range(len(history))]
    backward = [final[i] - after_trained[i] for i in range(len(final) - 1)]
    backward_transfer = sum(backward) / len(backward) if backward else 0.0
    return {
        "method": method,
        "seed": seed,
        "history": history,
        "final_average_acc": history[-1]["average_acc"],
        "final_per_task_acc": final,
        "backward_transfer": backward_transfer,
        "rank_per_task": args.rank_per_task,
        "k_max": args.k_max,
        "lr": args.lr,
        "lr_bias": args.lr_bias,
        "eta_dual": args.eta_dual,
    }


def pilot_lr(args, train, test_x, test_y, init) -> float:
    """Pick the smallest step that matches the best task-0 accuracy within 1 point."""
    scored = []
    clients = train[0]
    for lr in args.pilot_lrs:
        trial = argparse.Namespace(**vars(args))
        trial.lr = lr
        trial.lr_bias = lr
        trial.rounds = args.pilot_rounds
        result = run_method(trial, "fedmuon", seed=0, train=[clients], test_x=test_x, test_y=test_y, init=init)
        acc = result["final_per_task_acc"][0]
        print(f"pilot lr={lr}: task0 acc={acc:.3f}", flush=True)
        scored.append((acc, lr))
    best_acc = max(acc for acc, _ in scored)
    eligible = [lr for acc, lr in scored if acc >= best_acc - 0.01]
    return min(eligible)


def write_svg(results: list[dict], path: Path) -> None:
    methods = []
    for result in results:
        if result["method"] not in methods:
            methods.append(result["method"])
    n_tasks = max(len(result["history"]) for result in results)
    width, height = 760, 360
    left, right, top, bottom = 56, 20, 28, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {"fedmuon": "#b45309", "proj": "#1d4ed8", "dual": "#047857"}
    labels = {"fedmuon": "FedMuon", "proj": "FedMuon + project-then-Muon", "dual": "FedMuon + dual OGD"}

    def xy(task_index: int, acc: float) -> tuple[float, float]:
        x = left + plot_w * task_index / max(n_tasks - 1, 1)
        y = top + plot_h * (1 - acc)
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{left}" y="18" font-family="sans-serif" font-size="14">Average accuracy on tasks seen so far</text>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1):
        _, y = xy(0, tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#4b5563">{tick:.2f}</text>'
        )
    for task_index in range(n_tasks):
        x, _ = xy(task_index, 0)
        parts.append(
            f'<text x="{x:.1f}" y="{height-22}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4b5563">T{task_index+1}</text>'
        )
    for method in methods:
        rows = [r for r in results if r["method"] == method]
        series = []
        for task_index in range(n_tasks):
            vals = [r["history"][task_index]["average_acc"] for r in rows]
            series.append(sum(vals) / len(vals))
        points = " ".join(f"{xy(i, acc)[0]:.1f},{xy(i, acc)[1]:.1f}" for i, acc in enumerate(series))
        color = colors.get(method, "#111827")
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
        for i, acc in enumerate(series):
            x, y = xy(i, acc)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
    for i, method in enumerate(methods):
        color = colors.get(method, "#111827")
        y = 20 + i * 16
        parts.append(f'<rect x="{width-250}" y="{y-9}" width="10" height="10" fill="{color}"/>')
        parts.append(
            f'<text x="{width-234}" y="{y}" font-family="sans-serif" font-size="11">{labels.get(method, method)}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--local-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-bias", type=float, default=0.05)
    parser.add_argument("--eta-dual", type=float, default=1.0)
    parser.add_argument("--inner", type=int, default=8)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--rank-per-task", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--methods", nargs="+", default=["fedmuon", "proj", "dual"])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-rounds", type=int, default=8)
    parser.add_argument("--pilot-lrs", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--out", type=Path, default=Path("/workspace/fedmuon_ogd/outputs"))
    args = parser.parse_args()

    torch.set_num_threads(4)
    train_x, train_y = load_images(TRAIN_IMG), load_labels(TRAIN_LBL)
    test_x, test_y = load_images(TEST_IMG), load_labels(TEST_LBL)
    # One split shared by every method. The seed here only shuffles within a task.
    tasks = TASKS[: args.num_tasks]
    train = [
        partition_task(train_x, train_y, classes, args.clients, seed=100 + task_id)
        for task_id, classes in enumerate(tasks)
    ]
    init = init_model(0)
    if args.pilot:
        best = pilot_lr(args, train, test_x, test_y, init)
        print(f"selected lr={best}", flush=True)
        args.lr = best
        args.lr_bias = best

    results = []
    for method in args.methods:
        for seed in args.seeds:
            results.append(run_method(args, method, seed, train, test_x, test_y, init))

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "setting": {
            "dataset": "split-MNIST class-incremental",
            "tasks": [list(task) for task in TASKS],
            "clients": args.clients,
            "participation": "full",
            "partition": "IID equal shard within each task",
            "rounds_per_task": args.rounds,
            "local_steps": args.local_steps,
            "batch_size": args.batch_size,
            "alpha": args.alpha,
            "lr": args.lr,
            "lr_bias": args.lr_bias,
            "eta_dual": args.eta_dual,
            "inner": args.inner,
            "ns_steps": args.ns_steps,
            "rank_per_task": args.rank_per_task,
            "k_max": args.k_max,
            "model": "MLP 784-128-10",
        },
        "results": results,
    }
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    write_svg(results, args.out / "average_accuracy.svg")
    print("wrote", args.out / "results.json", flush=True)


if __name__ == "__main__":
    main()
