"""Audit the reported Split-MNIST numbers and test a federated claim.

The claim under test is not "Muon plus OGD". It is: when clients enforce
different one-sided constraints, each local update can be feasible while the
aggregated update is not. A single shared constraint is the excluded trivial
case, because feasibility is linear. A matched-rank random shared constraint
is the control for rank reduction.

Run from /workspace:

    python3 fedmuon_ogd/audit_discriminate.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_split_mnist as base

OUT = Path(__file__).resolve().parent / "audit_logs"
RESULTS = Path(__file__).resolve().parent / "outputs_directions" / "results.json"


def as_list(tensor: torch.Tensor, digits: int = 8):
    return [[round(float(v), digits) for v in row] for row in tensor.detach().cpu()]


def rank_tolerance(values: torch.Tensor, rows: int, cols: int) -> float:
    """Numerical rank cutoff matched to dtype and matrix size: σ_max · ε · max(m, n)."""
    eps = float(torch.finfo(values.dtype).eps)
    sigma = float(values[0]) if values.numel() else 0.0
    return sigma * eps * max(rows, cols, 1)


def compact_polar(matrix: torch.Tensor) -> torch.Tensor:
    eps = float(torch.finfo(matrix.dtype).eps)
    if float(matrix.detach().norm()) <= eps * max(matrix.shape):
        return torch.zeros_like(matrix)
    left, values, right = torch.linalg.svd(matrix, full_matrices=False)
    keep = values > rank_tolerance(values, matrix.shape[0], matrix.shape[1])
    if int(keep.sum()) == 0:
        return torch.zeros_like(matrix)
    return left[:, keep] @ right[keep, :]


def legacy_float32_polar(matrix: torch.Tensor) -> torch.Tensor:
    """Voided operator: relative cutoff 1e-10, not matched to float32 or the matrix size."""
    left, values, right = torch.linalg.svd(matrix, full_matrices=False)
    keep = values > 1e-10 * values[0].clamp_min(1e-10)
    if int(keep.sum()) == 0:
        return torch.zeros_like(matrix)
    return left[:, keep] @ right[keep, :]


def orthogonal_complement(basis: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(basis, mode="complete")
    return q[:, basis.shape[1] :].contiguous()


def onesided_lmo(grad: torch.Tensor, basis: torch.Tensor | None) -> torch.Tensor:
    """D = polar(G N) N^T in float64. N is an orthonormal basis of the complement of B."""
    work = grad.detach().to(torch.float64)
    if basis is None:
        return compact_polar(work).to(grad.dtype)
    columns = basis.detach().to(torch.float64)
    if columns.shape[1] >= columns.shape[0]:
        return torch.zeros_like(grad)
    complement = orthogonal_complement(columns)
    direction = compact_polar(work @ complement) @ complement.T
    return direction.to(grad.dtype)


def orthonormal(n: int, rank: int, generator: torch.Generator, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    raw = torch.randn(n, rank, generator=generator, dtype=dtype)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q[:, :rank]


def user_spectral_ball_regression() -> dict:
    """Two different orders. Do not attach both numbers to one matrix."""
    c = 1.0 / 3.0 ** 0.5
    s = (2.0 / 3.0) ** 0.5
    grad = torch.tensor([[c, s], [s, -c]], dtype=torch.float64)
    polar_g = compact_polar(grad)
    order_polar_then_t = polar_g.clone()
    order_polar_then_t[1, 1] = 0
    order_t_then_input = grad.clone()
    order_t_then_input[1, 1] = 0
    order_t_then_polar = compact_polar(order_t_then_input)
    norm_t_polar = float(torch.linalg.matrix_norm(order_polar_then_t, ord=2))
    norm_polar_t = float(torch.linalg.matrix_norm(order_t_then_polar, ord=2))
    d22_t_polar = float(order_polar_then_t[1, 1])
    d22_polar_t = float(order_t_then_polar[1, 1])
    same_matrix = float((order_polar_then_t - order_t_then_polar).abs().max()) == 0.0
    ok = (
        float((polar_g - grad).abs().max()) < 1e-12
        and abs(float(torch.linalg.matrix_norm(grad, ord=2)) - 1.0) < 1e-12
        and abs(d22_t_polar) < 1e-12
        and abs(norm_t_polar - 2.0 / 3.0 ** 0.5) < 1e-12
        and abs(norm_polar_t - 1.0) < 1e-12
        and abs(d22_polar_t + 1.0 / 3.0) < 1e-12
        and not same_matrix
    )
    payload = {
        "passed": ok,
        "G_equals_polar_G": True,
        "T_of_polar_G": as_list(order_polar_then_t, 12),
        "T_of_polar_G_spectral_norm": norm_t_polar,
        "T_of_polar_G_D22": d22_t_polar,
        "polar_of_T_G": as_list(order_t_then_polar, 12),
        "polar_of_T_G_spectral_norm": norm_polar_t,
        "polar_of_T_G_D22": d22_polar_t,
        "same_matrix": same_matrix,
        "statement": (
            "T(polar(G)) has spectral norm 2/sqrt(3) and D22 = 0: Frobenius projection can leave the spectral ball. "
            "polar(T(G)) has spectral norm 1 and D22 = -1/3: the polar factor can leave the constraint. "
            "These are two orders and two matrices."
        ),
    }
    print(
        f"CHECK T(polar(G)) shape=[2, 2] dtype=float64 spectral={norm_t_polar:.12f} D22={d22_t_polar:.12f}",
        flush=True,
    )
    print(
        f"CHECK polar(T(G)) shape=[2, 2] dtype=float64 spectral={norm_polar_t:.12f} D22={d22_polar_t:.12f}",
        flush=True,
    )
    if not ok:
        raise AssertionError(payload)
    return payload


def theory_audit() -> dict:
    grad = torch.tensor([[1.0, 0.85], [0.35, 0.45]])
    constraint = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    left = torch.tensor([[1.0], [0.0]])
    right = torch.tensor([[1.0], [0.0]])
    removed = grad - base.project_out(grad, left, right)
    projected_grad = base.project_out(grad, left, right)
    plain = base.newton_schulz(grad, steps=8)
    after_grad_proj = base.newton_schulz(projected_grad, steps=8)
    after_update_proj = base.project_out(plain, left, right)
    singular = {
        "G": as_list(torch.linalg.svdvals(grad).unsqueeze(0))[0],
        "projected_gradient": as_list(torch.linalg.svdvals(projected_grad).unsqueeze(0))[0],
        "plain_ns": as_list(torch.linalg.svdvals(plain).unsqueeze(0))[0],
        "project_gradient_then_ns": as_list(torch.linalg.svdvals(after_grad_proj).unsqueeze(0))[0],
        "ns_then_frobenius_projection": as_list(torch.linalg.svdvals(after_update_proj).unsqueeze(0))[0],
    }
    spectral = {key: values[0] for key, values in singular.items()}

    # General Frobenius projection need not preserve the spectral ball.
    generator = torch.Generator().manual_seed(11)
    ball_examples = []
    found = None
    for trial in range(200):
        basis = torch.randn(3, 3, 4, generator=generator)
        basis = basis.reshape(9, 4)
        q, _ = torch.linalg.qr(basis, mode="reduced")
        source = torch.randn(3, 3, generator=generator)
        source = source / torch.linalg.matrix_norm(source, ord=2)
        coeff = torch.stack([(source.reshape(-1) * q[:, j]).sum() for j in range(4)])
        projected = (q[:, :4] @ coeff).reshape(3, 3)
        spec = float(torch.linalg.matrix_norm(projected, ord=2))
        ball_examples.append(spec)
        if spec > 1.05 and found is None:
            found = {
                "trial": trial,
                "source_spectral_norm": 1.0,
                "projected_spectral_norm": spec,
                "projected_frobenius_norm": float(projected.norm()),
                "source": as_list(source, 6),
                "projection": as_list(projected, 6),
            }
            break
    onesided = onesided_checks()
    reference = reference_linear_constraints()
    return {
        "user_regression_frobenius_projection_exits_spectral_ball": user_spectral_ball_regression(),
        "twobytwo_hyperplane": {
            "scope": "single Frobenius constraint <A, D> = D_00 = 0; not DB=0",
            "G": as_list(grad, 6),
            "A": as_list(constraint, 6),
            "removed_component": as_list(removed, 6),
            "projected_gradient": as_list(projected_grad, 6),
            "plain_ns": as_list(plain, 6),
            "project_gradient_then_ns": as_list(after_grad_proj, 6),
            "ns_then_frobenius_projection": as_list(after_update_proj, 6),
            "singular_values": singular,
            "spectral_norms": spectral,
            "frobenius_projection_of_ns_stays_inside_spectral_ball": spectral["ns_then_frobenius_projection"] <= 1 + 1e-6,
            "residuals": {
                "plain_ns": float((left.T @ plain @ right).abs()),
                "project_gradient_then_ns": float((left.T @ after_grad_proj @ right).abs()),
                "ns_then_frobenius_projection": float((left.T @ after_update_proj @ right).abs()),
            },
        },
        "general_frobenius_projection_can_exit_spectral_ball": {
            "answer": True,
            "matrix": "T(polar(G))",
            "spectral_norm": 2.0 / 3.0 ** 0.5,
            "D22": 0.0,
            "different_order": "polar(T(G)) has spectral norm 1 and D22 = -1/3",
            "random_search_hit_above_1_05": found,
        },
        "max_random_projected_spectral_norm_before_first_hit": max(ball_examples),
        "onesided_lemma_checks": onesided,
        "reference_solver": reference,
    }


def emit_check(name: str, info: dict, passed: bool) -> None:
    values = ",".join(f"{value:.6e}" for value in info["singular_values"])
    print(
        f"CHECK {name} shape={info['shape']} dtype={info['dtype']} svd=[{values}] "
        f"BtB-I={info['BtB_minus_I']:.3e} NNt-P={info['NNt_minus_P']:.3e} "
        f"gap={info['objective_gap']:.3e} residual={info['residual']:.3e} pass={passed}",
        flush=True,
    )


def measure_reduced(grad: torch.Tensor, basis: torch.Tensor, eta: float = 1.0) -> dict:
    work = torch.float64
    matrix = grad.to(work)
    columns = basis.to(work)
    complement = orthogonal_complement(columns)
    projector = torch.eye(columns.shape[0], dtype=work) - columns @ columns.T
    reduced = matrix @ complement
    singular = torch.linalg.svdvals(reduced)
    direction = eta * (compact_polar(reduced) @ complement.T)
    nuclear = float(singular.sum())
    return {
        "shape": list(matrix.shape),
        "dtype": "float64",
        "singular_values": [float(value) for value in singular],
        "BtB_minus_I": float((columns.T @ columns - torch.eye(columns.shape[1], dtype=work)).abs().max()),
        "NNt_minus_P": float((complement @ complement.T - projector).abs().max()),
        "objective_gap": abs(float((matrix * direction).sum()) - eta * nuclear),
        "residual": float((direction @ columns).norm()),
        "spectral_norm": float(torch.linalg.matrix_norm(direction, ord=2)),
        "direction": direction,
        "reduced": reduced,
        "complement": complement,
        "columns": columns,
        "matrix": matrix,
        "projector": projector,
    }


def onesided_checks() -> dict:
    generator = torch.Generator().manual_seed(3)
    trials = []
    legacy = []
    direct = []
    failures = []
    for index in range(40):
        grad = torch.randn(7, 9, generator=generator, dtype=torch.float64)
        basis = orthonormal(9, 3, generator, dtype=torch.float64)
        measured = measure_reduced(grad, basis, eta=1.0)
        passed = (
            measured["BtB_minus_I"] < 1e-10
            and measured["NNt_minus_P"] < 1e-10
            and measured["objective_gap"] < 1e-8
            and measured["residual"] < 1e-8
            and measured["spectral_norm"] <= 1.0 + 1e-8
        )
        public = {key: value for key, value in measured.items() if key not in ("direction", "reduced", "complement", "columns", "matrix", "projector")}
        public["index"] = index
        trials.append(public)
        emit_check(f"reduced-polar[{index}]", public, passed)
        if not passed:
            failures.append(f"reduced-polar[{index}]")
        product = measured["matrix"] @ measured["projector"]
        direct_direction = compact_polar(product)
        direct_gap = abs(float((measured["matrix"] * direct_direction).sum()) - float(torch.linalg.svdvals(measured["reduced"]).sum()))
        direct_residual = float((direct_direction @ measured["columns"]).norm())
        direct_pass = direct_gap < 1e-8 and direct_residual < 1e-8
        direct_info = {
            "shape": list(product.shape),
            "dtype": "float64",
            "singular_values": [float(value) for value in torch.linalg.svdvals(product)],
            "BtB_minus_I": public["BtB_minus_I"],
            "NNt_minus_P": public["NNt_minus_P"],
            "objective_gap": direct_gap,
            "residual": direct_residual,
        }
        direct.append({key: direct_info[key] for key in ("objective_gap", "residual")})
        emit_check(f"float64-polar-GP[{index}]", direct_info, direct_pass)
        if not direct_pass:
            failures.append(f"float64-polar-GP[{index}]")
        old = legacy_float32_polar(product.to(torch.float32))
        old_info = {
            "shape": list(product.shape),
            "dtype": "float32",
            "singular_values": [float(value) for value in torch.linalg.svdvals(product.to(torch.float32))],
            "BtB_minus_I": float((basis.to(torch.float32).T @ basis.to(torch.float32) - torch.eye(3)).abs().max()),
            "NNt_minus_P": public["NNt_minus_P"],
            "objective_gap": abs(float((grad.to(torch.float32) * old).sum()) - float(torch.linalg.svdvals(product.to(torch.float32)).sum())),
            "residual": float((old @ basis.to(torch.float32)).norm()),
        }
        legacy.append(old_info)
        emit_check(f"void-float32-polar-GP[{index}]", old_info, old_info["residual"] < 1e-4 and old_info["objective_gap"] < 1e-4)
    reproduced = reproduce_voided_float32()
    completions = kernel_completions()
    for item in completions:
        emit_check(item["name"], item, item["passed"])
        if not item["passed"]:
            failures.append(item["name"])
    legacy_residual = max(item["residual"] for item in legacy)
    return {
        "passed": not failures,
        "failures": failures,
        "operator": "D = polar(GN) N^T in float64; rank cutoff is sigma_max * eps * max(m, n)",
        "trials": 40,
        "max_objective_gap": max(item["objective_gap"] for item in trials),
        "max_residual": max(item["residual"] for item in trials),
        "max_BtB_minus_I": max(item["BtB_minus_I"] for item in trials),
        "max_NNt_minus_P": max(item["NNt_minus_P"] for item in trials),
        "max_float64_polar_GP_gap": max(item["objective_gap"] for item in direct),
        "max_float64_polar_GP_residual": max(item["residual"] for item in direct),
        "voided_float32_polar_GP": {
            "relative_cutoff": 1e-10,
            "paired_float64_sample_max_residual": legacy_residual,
            "paired_float64_sample_max_objective_gap": max(item["objective_gap"] for item in legacy),
            "original_float32_draw": reproduced,
            "previous_diagnostic_void": True,
        },
        "kernel_completion_in_reduced_coordinates": [
            {key: value for key, value in item.items() if key != "passed"} | {"passed": item["passed"]}
            for item in completions
        ],
        "trial_records": trials,
        "momentum_note": (
            "The identity is one step. Momentum or a FedMuon correction only replaces G. "
            "In exact arithmetic the map X -> aX + b XX^T X + c (XX^T)^2 X preserves a one-sided constraint XB=0. "
            "Finite precision, two-sided constraints, and the spectral-norm bound are separate. "
            "Coefficients 3.4445, -4.775, 2.0315 send the scalar 0.5 to 1.188859375."
        ),
    }


def reproduce_voided_float32() -> dict:
    """The draw that kept a numerical null vector of P under the old cutoff."""
    generator = torch.Generator().manual_seed(3)
    worst = None
    records = []
    for index in range(40):
        grad = torch.randn(7, 9, generator=generator)
        basis = orthonormal(9, 3, generator)
        columns = basis.to(torch.float64)
        complement = orthogonal_complement(columns)
        projector = torch.eye(9, dtype=torch.float64) - columns @ columns.T
        product = grad @ (torch.eye(9) - basis @ basis.T)
        values = torch.linalg.svdvals(product)
        direction = legacy_float32_polar(product)
        info = {
            "index": index,
            "shape": [7, 9],
            "dtype": "float32",
            "singular_values": [float(value) for value in values],
            "BtB_minus_I": float((basis.T @ basis - torch.eye(3)).abs().max()),
            "NNt_minus_P": float((complement @ complement.T - projector).abs().max()),
            "objective_gap": abs(float((grad * direction).sum()) - float(values.sum())),
            "residual": float((direction @ basis).norm()),
        }
        emit_check(
            f"void-original-float32[{index}]",
            info,
            info["residual"] < 1e-4 and info["objective_gap"] < 1e-4,
        )
        records.append(info)
        if worst is None or info["residual"] > worst["residual"]:
            worst = info
    return {
        "worst_residual_index": worst["index"],
        "worst_residual": worst["residual"],
        "objective_gap_at_worst_residual": worst["objective_gap"],
        "max_objective_gap": max(item["objective_gap"] for item in records),
        "max_residual": max(item["residual"] for item in records),
        "singular_values_at_worst_residual": worst["singular_values"],
        "BtB_minus_I": worst["BtB_minus_I"],
        "NNt_minus_P": worst["NNt_minus_P"],
    }


def kernel_completions() -> list[dict]:
    generator = torch.Generator().manual_seed(11)
    basis = orthonormal(9, 3, generator, dtype=torch.float64)
    complement = orthogonal_complement(basis)
    coefficient = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    grad = coefficient @ complement[:, :5].T
    reduced = grad @ complement
    left, values, right = torch.linalg.svd(reduced, full_matrices=True)
    cutoff = rank_tolerance(values, reduced.shape[0], reduced.shape[1])
    rank = int((values > cutoff).sum())
    projector = torch.eye(9, dtype=torch.float64) - basis @ basis.T
    gram = float((basis.T @ basis - torch.eye(3, dtype=torch.float64)).abs().max())
    projector_error = float((complement @ complement.T - projector).abs().max())
    nuclear = float(values.sum())
    records = []
    if rank >= right.shape[0] or rank >= left.shape[1]:
        records.append(
            {
                "name": "kernel-completion",
                "passed": False,
                "shape": list(grad.shape),
                "dtype": "float64",
                "singular_values": [float(value) for value in values],
                "BtB_minus_I": gram,
                "NNt_minus_P": projector_error,
                "objective_gap": 1.0,
                "residual": 1.0,
            }
        )
        return records
    eta = 0.37
    for index in range(4):
        right_kernel = right[rank:, :].T
        left_kernel = left[:, rank:]
        right_vector = right_kernel @ torch.randn(right_kernel.shape[1], generator=generator, dtype=torch.float64)
        left_vector = left_kernel @ torch.randn(left_kernel.shape[1], generator=generator, dtype=torch.float64)
        right_vector = right_vector / right_vector.norm()
        left_vector = left_vector / left_vector.norm()
        factor = left[:, :rank] @ right[:rank, :] + left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
        direction = eta * (factor @ complement.T)
        info = {
            "name": f"kernel-completion[{index}]",
            "shape": list(grad.shape),
            "dtype": "float64",
            "singular_values": [float(value) for value in values],
            "BtB_minus_I": gram,
            "NNt_minus_P": projector_error,
            "objective_gap": abs(float((grad * direction).sum()) - eta * nuclear),
            "residual": float((direction @ basis).norm()),
            "spectral_norm": float(torch.linalg.matrix_norm(direction, ord=2)),
        }
        info["passed"] = (
            info["objective_gap"] < 1e-8
            and info["residual"] < 1e-8
            and abs(info["spectral_norm"] - eta) < 1e-8
            and gram < 1e-10
            and projector_error < 1e-10
        )
        records.append(info)
    return records


def old_dual_iteration(grad: torch.Tensor, constraints: list[torch.Tensor], steps: int = 8) -> torch.Tensor:
    lam = torch.zeros(len(constraints))
    normalized = grad / grad.norm().clamp_min(1e-8)
    current = normalized
    signed = compact_polar(current)
    for _ in range(steps):
        alignment = torch.tensor([(constraints[i] * signed).sum() for i in range(len(constraints))])
        current_value = torch.linalg.svdvals(current).sum()
        step = 1.0
        accepted = False
        for _attempt in range(6):
            trial = lam - step * alignment
            candidate = normalized + sum(trial[i] * constraints[i] for i in range(len(constraints)))
            value = torch.linalg.svdvals(candidate).sum()
            if float(value) < float(current_value) - 1e-6:
                lam = trial
                current = candidate
                signed = compact_polar(candidate)
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    return signed.detach()


def reference_linear_constraints() -> dict:
    generator = torch.Generator().manual_seed(19)
    records = []
    for case in range(5):
        grad = torch.randn(6, 5, generator=generator)
        constraints = []
        for _ in range(3):
            matrix = torch.randn(6, 5, generator=generator)
            constraints.append(matrix / matrix.norm())
        lam = torch.zeros(3, requires_grad=True)
        opt = torch.optim.LBFGS([lam], lr=0.8, max_iter=25, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            shifted = grad + sum(lam[i] * constraints[i] for i in range(3))
            values = torch.linalg.svdvals(shifted)
            loss = torch.sqrt(values * values + 1e-10).sum()
            loss.backward()
            return loss

        for _ in range(12):
            opt.step(closure)
        with torch.no_grad():
            shifted = grad + sum(lam[i] * constraints[i] for i in range(3))
            dual = float(torch.linalg.svdvals(shifted).sum())
            primal_direction = compact_polar(shifted)
            primal = float((grad * primal_direction).sum())
            residual = [float((constraints[i] * primal_direction).sum()) for i in range(3)]
            crude = old_dual_iteration(grad, constraints, steps=8)
            crude_primal = float((grad * crude).sum())
            crude_residual = [float((constraints[i] * crude).sum()) for i in range(3)]
        records.append(
            {
                "case": case,
                "dual_nuclear_norm": dual,
                "primal_objective": primal,
                "gap": dual - primal,
                "reference_residual_l2": sum(v * v for v in residual) ** 0.5,
                "eight_step_dual_objective": crude_primal,
                "eight_step_dual_residual_l2": sum(v * v for v in crude_residual) ** 0.5,
                "eight_step_objective_deficit": primal - crude_primal,
            }
        )
    return {
        "matrix_shape": [6, 5],
        "constraints": 3,
        "solver": "LBFGS on smoothed nuclear norm, mu=1e-10, exact SVD polar at the reported lambda",
        "cases": records,
        "max_reference_gap": max(abs(row["gap"]) for row in records),
        "max_reference_residual_l2": max(row["reference_residual_l2"] for row in records),
        "min_eight_step_residual_l2": min(row["eight_step_dual_residual_l2"] for row in records),
    }


def index_hash(index: torch.Tensor) -> str:
    raw = ",".join(str(int(v)) for v in index.detach().cpu().tolist())
    return hashlib.sha256(raw.encode()).hexdigest()


def dirichlet_partition(images, labels, classes, n_clients, beta, seed):
    """Official Dirichlet.sample inside fork_rng. Empty clients are kept explicitly."""
    buckets_x = [[] for _ in range(n_clients)]
    buckets_y = [[] for _ in range(n_clients)]
    buckets_i = [[] for _ in range(n_clients)]
    class_meta = []
    for offset, label in enumerate(classes):
        index = torch.where(labels == label)[0]
        with torch.random.fork_rng():
            torch.manual_seed(seed + 1009 * (offset + 1))
            props = torch.distributions.Dirichlet(torch.full((n_clients,), float(beta))).sample()
            order = torch.randperm(index.numel())
        index = index[order]
        counts = torch.floor(props * index.numel()).to(torch.long)
        short = int(index.numel() - int(counts.sum()))
        if short > 0:
            rank = torch.argsort(props, descending=True)
            for j in range(short):
                counts[int(rank[j % n_clients])] += 1
        if int(counts.sum()) != int(index.numel()):
            raise RuntimeError("Dirichlet allocation dropped samples")
        start = 0
        assigned = []
        for client, count in enumerate(counts.tolist()):
            chosen = index[start : start + count]
            start += count
            assigned.append(int(count))
            if count == 0:
                continue
            buckets_x[client].append(images[chosen])
            buckets_y[client].append(labels[chosen])
            buckets_i[client].append(chosen)
        class_meta.append(
            {
                "label": int(label),
                "counts": assigned,
                "index_hash": index_hash(index),
                "empty_clients": [i for i, count in enumerate(assigned) if count == 0],
            }
        )
    clients = []
    counts = []
    hashes = []
    for client in range(n_clients):
        if not buckets_x[client]:
            clients.append((torch.empty(0, images.size(1)), torch.empty(0, dtype=torch.long)))
            counts.append({int(label): 0 for label in classes})
            hashes.append(None)
            continue
        x = torch.cat(buckets_x[client], dim=0)
        y = torch.cat(buckets_y[client], dim=0)
        owned = torch.cat(buckets_i[client], dim=0)
        clients.append((x.contiguous(), y.contiguous()))
        counts.append({int(label): int((y == label).sum()) for label in classes})
        hashes.append(index_hash(owned))
    return clients, {"class_counts": counts, "client_index_hashes": hashes, "classes": class_meta}


def top_right(matrix: torch.Tensor, rank: int) -> torch.Tensor | None:
    if matrix.numel() == 0 or float(matrix.norm()) < 1e-10:
        return None
    _, _, right = torch.linalg.svd(matrix, full_matrices=False)
    rank = min(rank, right.size(0))
    return right[:rank, :].T.contiguous()


def merge_bases(bases: list[torch.Tensor], rank: int) -> torch.Tensor:
    stacked = torch.cat(bases, dim=1)
    q, _ = torch.linalg.qr(stacked, mode="reduced")
    return q[:, : min(rank, q.size(1))].contiguous()


def client_task_score(params, test_x, test_y, classes, held_counts) -> float:
    scores = []
    for label, count in held_counts.items():
        if count <= 0 or label not in classes:
            continue
        mask = test_y == label
        scores.append(base.evaluate_task(params, test_x, test_y, (label, label)) if False else None)
        inputs = test_x[mask]
        labels = test_y[mask]
        if inputs.numel() == 0:
            continue
        with torch.no_grad():
            pred = base.forward(params, inputs).argmax(dim=1)
        scores[-1] = float((pred == labels).float().mean())
    if not scores:
        return float("nan")
    return sum(v for v in scores if v is not None) / len([v for v in scores if v is not None])


def build_protocol(train_x, train_y, tasks, n_clients, participants, rounds, local_steps, batch_size, beta, seed):
    """One split, one participation sequence, and one batch sequence per seed."""
    splits = []
    meta = []
    for task_id, classes in enumerate(tasks):
        clients, info = dirichlet_partition(
            train_x, train_y, classes, n_clients, beta, seed=8000 + 100 * seed + task_id
        )
        splits.append(clients)
        meta.append(info)
    participation = []
    with torch.random.fork_rng():
        torch.manual_seed(5000 + seed)
        for task_id in range(len(tasks)):
            eligible = [i for i in range(n_clients) if splits[task_id][i][0].size(0) >= 2]
            task_rounds = []
            for _round in range(rounds):
                if not eligible:
                    task_rounds.append([])
                    continue
                perm = torch.randperm(len(eligible)).tolist()
                task_rounds.append([eligible[j] for j in perm[: min(participants, len(eligible))]])
            participation.append(task_rounds)
    batches = []
    with torch.random.fork_rng():
        torch.manual_seed(7000 + seed)
        for task_id in range(len(tasks)):
            per_client = []
            for client_id in range(n_clients):
                n_local = int(splits[task_id][client_id][0].size(0))
                seq = []
                for _step in range(rounds * local_steps):
                    if n_local < 2:
                        seq.append([])
                    else:
                        seq.append(torch.randperm(n_local)[: min(batch_size, n_local)].tolist())
                per_client.append(seq)
            batches.append(per_client)
    return splits, meta, participation, batches


def run_discriminate() -> dict:
    torch.set_num_threads(4)
    train_x, train_y = base.load_images(base.TRAIN_IMG), base.load_labels(base.TRAIN_LBL)
    test_x, test_y = base.load_images(base.TEST_IMG), base.load_labels(base.TEST_LBL)
    tasks = base.TASKS[:3]
    n_clients = 8
    participants = 4
    rounds = 8
    local_steps = 3
    batch_size = 64
    rank = 2
    beta = 0.5
    lr = 0.01
    init = base.init_model(0)
    modes = ("local", "shared", "random")
    records = []
    protocols = []
    synthetic = synthetic_aggregation()
    for seed in (0, 1):
        splits, meta, participation, batches = build_protocol(
            train_x, train_y, tasks, n_clients, participants, rounds, local_steps, batch_size, beta, seed
        )
        protocols.append(
            {
                "seed": seed,
                "class_counts": [row["class_counts"] for row in meta],
                "client_index_hashes": [row["client_index_hashes"] for row in meta],
                "class_allocation": [row["classes"] for row in meta],
                "participation": participation,
                "empty_clients": [
                    [i for i, client in enumerate(task) if client[0].size(0) < 2] for task in splits
                ],
            }
        )
        for mode in modes:
            records.append(
                train_one(
                    mode, seed, splits, meta, tasks, test_x, test_y, init,
                    n_clients, rounds, local_steps, rank, lr, participation, batches,
                )
            )
            last = records[-1]["after_last_task"]
            worst = last["worst_client_old_acc"]
            worst_text = "none" if worst is None else f"{worst:.3f}"
            print(
                f"{mode} seed={seed} AA={last['average_acc']:.3f} "
                f"current={last['current_acc']:.3f} old={last['old_acc']:.3f} "
                f"agg_residual={last['aggregate_residual_mean']:.4f} "
                f"local_residual={last['local_residual_mean']:.4f} "
                f"worst_client={worst_text}",
                flush=True,
            )
    return {
        "status": "rerun with polar(GN) N^T after the float32 polar(GP) diagnostic was voided",
        "role": "diagnostic only; local versus shared subspaces are not a paper contribution",
        "protocol": {
            "tasks": [list(task) for task in tasks],
            "clients": n_clients,
            "participants_per_round": participants,
            "partition": "torch.distributions.Dirichlet(beta=0.5).sample inside torch.random.fork_rng",
            "rounds": rounds,
            "local_steps": local_steps,
            "batch_size": batch_size,
            "rank": rank,
            "lr": lr,
            "constraint": "one-sided DB=0 via D=polar(GN)N^T in float64, then cast back; previous polar(GP) float32 run is void",
            "bias_correction": "FedMuon increment C <- C + (1/n) sum_participants (C_i_new - C_i_old)",
            "evaluation": "unmasked 10-way argmax on the class subset; not logit-masked",
            "shared_random_are_trivial": "one common B makes DB=0 linear, so the average remains feasible",
        },
        "synthetic_aggregation": synthetic,
        "protocols": protocols,
        "runs": records,
    }


def synthetic_aggregation() -> dict:
    generator = torch.Generator().manual_seed(5)
    height, width, rank, n_clients = 8, 6, 2, 4
    local = []
    for client in range(n_clients):
        grad = torch.randn(height, width, generator=generator)
        basis = orthonormal(width, rank, generator)
        update = onesided_lmo(grad, basis)
        local.append({"basis": basis, "update": update, "local_residual": float((update @ basis).norm())})
    average = sum(item["update"] for item in local) / n_clients
    per_client = [float((average @ item["basis"]).norm()) for item in local]
    common = local[0]["basis"]
    trivial = []
    for client in range(n_clients):
        grad = torch.randn(height, width, generator=generator)
        update = onesided_lmo(grad, common)
        trivial.append(update)
    trivial_average = sum(trivial) / n_clients
    return {
        "different_bases_local_residuals": [item["local_residual"] for item in local],
        "different_bases_aggregate_residuals": per_client,
        "shared_basis_aggregate_residual": float((trivial_average @ common).norm()),
    }


def train_one(mode, seed, splits, meta, tasks, test_x, test_y, init, n_clients, rounds, local_steps, rank, lr, participation, batches):
    params = base.clone_params(init)
    momentum = [base.zeros_like_params(params) for _ in range(n_clients)]
    control = [base.zeros_like_params(params) for _ in range(n_clients)]
    global_control = base.zeros_like_params(params)
    bases = [None for _ in range(n_clients)]
    shared = None
    history = []
    round_logs = []
    protected_names = base.MATRIX_NAMES
    for task_id, clients in enumerate(splits):
        classes_so_far = tasks[: task_id + 1]
        for round_id in range(rounds):
            chosen = list(participation[task_id][round_id])
            deltas = []
            local_residuals = []
            if not chosen:
                round_logs.append(
                    {
                        "task": task_id,
                        "round": round_id,
                        "participants": [],
                        "empty_round": True,
                        "mean_local_residual": 0.0,
                        "aggregate_residual_mean": 0.0,
                    }
                )
                continue
            snapshot = base.clone_params(params)
            control_snapshot = base.clone_params(global_control)
            local_states = []
            increments = []
            for client_id in chosen:
                local = base.clone_params(snapshot)
                old_control = base.clone_params(control[client_id])
                basis = shared if mode in ("shared", "random") else bases[client_id]
                step_batches = batches[task_id][client_id][round_id * local_steps : (round_id + 1) * local_steps]
                local, momentum[client_id], residual = local_steps_onesided(
                    local, momentum[client_id], control[client_id], control_snapshot,
                    clients[client_id], basis, step_batches, 0.1, lr, lr,
                )
                new_control = base.clone_params(momentum[client_id])
                increments.append({name: new_control[name] - old_control[name] for name in new_control})
                control[client_id] = new_control
                local_states.append(local)
                deltas.append((client_id, {name: local[name] - snapshot[name] for name in protected_names}))
                local_residuals.append(residual)
            participants = len(chosen)
            for name in params:
                params[name].copy_(snapshot[name] * ((n_clients - participants) / n_clients))
                for local in local_states:
                    params[name].add_(local[name], alpha=1.0 / n_clients)
            for name in global_control:
                global_control[name].copy_(control_snapshot[name])
                for increment in increments:
                    global_control[name].add_(increment[name], alpha=1.0 / n_clients)
            round_logs.append(
                {
                    "task": task_id,
                    "round": round_id,
                    "participants": chosen,
                    "empty_round": False,
                    "mean_local_residual": sum(local_residuals) / max(len(local_residuals), 1),
                    "aggregate_residual_mean": aggregate_residual(deltas, bases, shared, n_clients),
                }
            )
        task_acc = [base.evaluate_task(params, test_x, test_y, classes) for classes in classes_so_far]
        old = task_acc[:-1]
        client_scores = []
        if task_id > 0:
            for counts in meta[task_id - 1]["class_counts"]:
                score = score_classes(params, test_x, test_y, counts)
                if score is not None:
                    client_scores.append(score)
        history.append(
            {
                "task": task_id,
                "per_task_acc": task_acc,
                "average_acc": sum(task_acc) / len(task_acc),
                "current_acc": task_acc[-1],
                "old_acc": sum(old) / len(old) if old else task_acc[-1],
                "aggregate_residual_mean": mean_of(round_logs, task_id, "aggregate_residual_mean"),
                "local_residual_mean": mean_of(round_logs, task_id, "mean_local_residual"),
                "worst_client_old_acc": min(client_scores) if client_scores else None,
            }
        )
        bases, shared = update_bases(mode, params, clients, bases, rank, seed, task_id)
        for client_id in range(n_clients):
            momentum[client_id] = base.zeros_like_params(params)
            control[client_id] = base.zeros_like_params(params)
        global_control = base.zeros_like_params(params)
    return {"mode": mode, "seed": seed, "history": history, "after_last_task": history[-1], "rounds": round_logs}


def mean_of(rows, task_id, key):
    vals = [row[key] for row in rows if row["task"] == task_id]
    return sum(vals) / len(vals)


def score_classes(params, test_x, test_y, counts) -> float:
    scores = []
    for label, count in counts.items():
        if count <= 0:
            continue
        mask = test_y == label
        inputs = test_x[mask]
        labels = test_y[mask]
        with torch.no_grad():
            pred = base.forward(params, inputs).argmax(dim=1)
        scores.append(float((pred == labels).float().mean()))
    if not scores:
        return None
    return sum(scores) / len(scores)


def aggregate_residual(deltas, bases, shared, n_clients) -> float:
    if not deltas:
        return 0.0
    names = deltas[0][1].keys()
    total = 0.0
    count = 0
    for name in names:
        average = sum(delta[name] for _, delta in deltas) / n_clients
        targets = []
        if shared is not None and shared.get(name) is not None:
            targets.append(shared[name])
        else:
            for client_id, _ in deltas:
                if bases[client_id] is not None and bases[client_id].get(name) is not None:
                    targets.append(bases[client_id][name])
        for basis in targets:
            total += float((average @ basis).norm())
            count += 1
    return total / max(count, 1)


def update_bases(mode, params, clients, bases, rank, seed, task_id):
    new_bases = []
    collected = {name: [] for name in base.MATRIX_NAMES}
    for client_id, data in enumerate(clients):
        if data[0].size(0) < 2:
            new_bases.append(bases[client_id])
            continue
        grad = base.average_client_gradient(params, [data], 64)
        local = {}
        for name in base.MATRIX_NAMES:
            local[name] = top_right(grad[name], rank)
            if local[name] is not None:
                collected[name].append(local[name])
        new_bases.append(local)
    shared = None
    if mode == "shared":
        shared = {}
        for name, pieces in collected.items():
            shared[name] = merge_bases(pieces, rank) if pieces else None
    elif mode == "random":
        generator = torch.Generator().manual_seed(9000 + 13 * seed + task_id)
        shared = {}
        for name in base.MATRIX_NAMES:
            width = params[name].size(1)
            shared[name] = orthonormal(width, min(rank, width), generator)
    return new_bases, shared


def local_steps_onesided(params, momentum, control, global_control, data, basis, step_batches, alpha, lr, lr_bias):
    x, y = data
    if x.size(0) < 2 or not step_batches:
        return params, momentum, 0.0
    residual = 0.0
    count = 0
    for local_index in step_batches:
        if len(local_index) < 2:
            continue
        xb, yb = x[local_index], y[local_index]
        grads = base.batch_grad(params, xb, yb)
        for name in base.MATRIX_NAMES:
            momentum[name].mul_(1 - alpha).add_(grads[name], alpha=alpha)
            corrected = momentum[name] - control[name] + global_control[name]
            if basis is None or basis.get(name) is None:
                direction = onesided_lmo(corrected, None)
            else:
                bmat = basis[name]
                direction = onesided_lmo(corrected, bmat)
                residual += float((direction.double() @ bmat.double()).norm())
                count += 1
            params[name].add_(direction, alpha=-lr)
        for name in base.BIAS_NAMES:
            momentum[name].mul_(1 - alpha).add_(grads[name], alpha=alpha)
            corrected = momentum[name] - control[name] + global_control[name]
            params[name].add_(corrected, alpha=-lr_bias)
    return params, momentum, residual / max(count, 1)


def argmax_caveat() -> dict:
    before = torch.tensor([2.0, 0.5, -1.0])
    after = torch.tensor([2.0, 0.5, 3.0])
    return {
        "old_logits_unchanged": bool(torch.equal(before[:2], after[:2])),
        "argmax_before": int(before.argmax()),
        "argmax_after": int(after.argmax()),
        "statement": (
            "Layer protection and freezing old head rows do not guarantee class-incremental accuracy. "
            "Old logits can stay fixed while a new-class logit changes the argmax."
        ),
    }


def provenance() -> dict:
    payload = json.loads(RESULTS.read_text())
    rows = [row for row in payload["results"] if row["method"] == "classrow"]
    aa = sum(row["final_average_acc"] for row in rows) / len(rows)
    last = sum(row["history"][-1]["current_acc"] for row in rows) / len(rows)
    return {
        "commit": "79de8960defb029809b287fdb965ce90aa75766f",
        "results_file": str(RESULTS),
        "stdout_archived": False,
        "reproduction": (
            "python3 fedmuon_ogd/run_split_mnist.py --rounds 10 --local-steps 4 "
            "--lr 0.01 --lr-bias 0.01 --rank-per-task 2 --k-max 8 --soft-alpha 0.5 "
            "--seeds 0 1 --methods fedmuon proj feasible soft classrow "
            "--out fedmuon_ogd/outputs_directions"
        ),
        "seeds": [row["seed"] for row in rows],
        "per_seed_final_average_acc": [row["final_average_acc"] for row in rows],
        "per_seed_last_task_acc": [row["history"][-1]["current_acc"] for row in rows],
        "mean_final_average_acc": aa,
        "mean_last_task_acc": last,
        "reported_rounding": {"56.3%": round(aa * 100, 1), "60.6%": round(last * 100, 1)},
        "setting_in_same_file": payload["setting"],
        "evaluation": (
            "evaluate_task keeps test rows whose label is in the task pair, then takes "
            "argmax over all 10 logits. Seen-class subset, not masked logits."
        ),
        "freezing_old_logits_does_not_fix_argmax": argmax_caveat(),
        "compute": "4 CPU threads, torch 2.15.0+cpu, CUDA unavailable. No GPU was used.",
        "fedmuon_bias_correction": {
            "matches_paper_update": "M <- (1-alpha)M + alpha grad; step uses lmo(M - C_i + C); C_i <- M; full participation averages parameters and control increments by 1/n.",
            "deviations": [
                "Newton-Schulz coefficients are the quintic 3.4445, -4.7750, 2.0315, not FedMuon's 15/8, -5/4, 3/8.",
                "No per-layer dimension rescaling of the step size.",
                "Momentum and controls are reset at every task boundary.",
                "The 56.3% method additionally zeros old output rows after the matrix sign and zeros old biases. That is not FedMuon.",
                "Five Newton-Schulz steps, not an exact polar factor.",
                "Data are IID equal shards and all 8 clients participate. FedMuon's reported experiments use Dirichlet partitions and partial participation.",
            ],
        },
    }


def literature_screen() -> dict:
    return {
        "decision": "reject the current main line as a submission",
        "not_contributions": [
            "Muon plus an orthogonal constraint plus federated averaging.",
            "A shared protection subspace. FOT already extracts a global input subspace and projects the aggregated update. FedProTIP already projects local client updates with a broadcast global subspace.",
            "Budget-constrained subspace Muon. FedSubMuon already optimizes Muon coefficients inside a shared structured subspace and tracks bases.",
            "Spectral LMO with local steps and control variates. That is FedMuon. FedCALM already adds compression on top of that line.",
        ],
        "tool_lemmas_only": [
            "Nuclear-norm dual of a spectral ball with linear constraints.",
            "Compact polar of GP is optimal for the one-sided constraint DB=0.",
        ],
        "fedgpm": {
            "pdf_retrieved": False,
            "same_line_as_fedprotip": False,
            "note": (
                "No FedGPM PDF was readable. FedProTIP cites centralized GPM (Saha, Garg, Roy, 2021), not a paper titled FedGPM. "
                "A separate ICIC 2024 paper by Lin et al. is titled as a federated gradient-projection aggregation method for medical diagnosis. "
                "Another short MedAI 2024 note also uses the name FedGPM for learning-rate optimization. "
                "These are not counted as a second copy of FedProTIP."
            ),
        },
        "sources": {
            "FOT": "https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf53f12ab4ea81d942f16162f7f37e6c-Abstract-Conference.html",
            "FedProTIP": "https://arxiv.org/abs/2509.21606",
            "GPM": "https://arxiv.org/abs/2103.09762",
            "C2-AFCL": "https://proceedings.mlr.press/v306/li26gx.html",
            "FedSubMuon": "https://arxiv.org/abs/2609.06073",
            "FedCALM": "https://arxiv.org/abs/2610.23434",
            "FedMuon": "https://proceedings.iclr.cc/paper_files/paper/2026/hash/427f20d90386fd27804f1831d6a3d48f-Abstract-Conference.html",
            "Muon-OGD": "https://arxiv.org/abs/2605.08949",
            "When Muon Meets Task Interference": "https://arxiv.org/abs/2608.27518",
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    theory = theory_audit()
    (OUT / "theory.json").write_text(json.dumps(theory, indent=2))
    print("wrote", OUT / "theory.json", flush=True)
    if not theory["onesided_lemma_checks"]["passed"]:
        (OUT / "discriminate.json").write_text(
            json.dumps(
                {
                    "status": "void",
                    "reason": "one-sided verification failed; diagnostic was not rerun",
                    "failures": theory["onesided_lemma_checks"]["failures"],
                },
                indent=2,
            )
        )
        raise SystemExit(1)
    experiment = run_discriminate()
    (OUT / "discriminate.json").write_text(json.dumps(experiment, indent=2))
    print("wrote", OUT / "discriminate.json", flush=True)
    (OUT / "provenance.json").write_text(json.dumps(provenance(), indent=2))
    print("wrote", OUT / "provenance.json", flush=True)
    (OUT / "literature.json").write_text(json.dumps(literature_screen(), indent=2))
    print("wrote", OUT / "literature.json", flush=True)


if __name__ == "__main__":
    main()
