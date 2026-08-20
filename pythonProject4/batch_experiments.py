#!/usr/bin/env python3
"""
Batch runner for the MNIST merge lab.

Runs a suite of experiments across one or more random seeds, then writes:
- per-run JSON files
- a flat CSV summary
- aggregate JSON/CSV by experiment and merge method
- a Markdown report
- SVG comparison plots
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence

from batch_report_utils import (
    METHOD_LABELS,
    aggregate_rows,
    generate_plots,
    generate_summary_markdown,
    write_csv,
)
from mnist_merge_lab import build_parser, run_experiment


DEFAULT_MERGE_METHODS = [
    "single_best",
    "simple_avg",
    "sample_weighted_avg",
    "epoch_weighted_avg",
    "regmean_equal",
    "regmean_epoch_weighted",
]


EXPERIMENT_PRESETS: Dict[str, Dict[str, object]] = {
    "phase0_iid_equal": {
        "description": "IID split with equal training budget.",
        "split_mode": "iid",
        "client_epochs": [5] * 10,
        "centralized_epochs": 5,
    },
    "phase1_iid_budget_hetero": {
        "description": "IID split with heterogeneous local training epochs.",
        "split_mode": "iid",
        "client_epochs": [1, 2, 3, 5, 8, 10, 15, 20, 30, 50],
        "centralized_epochs": 10,
    },
    "phase2_dirichlet_mild": {
        "description": "Dirichlet label skew with equal training budget.",
        "split_mode": "dirichlet",
        "dirichlet_alpha": 1.0,
        "client_epochs": [5] * 10,
        "centralized_epochs": 5,
    },
    "phase2_dirichlet_medium": {
        "description": "Stronger Dirichlet label skew with equal training budget.",
        "split_mode": "dirichlet",
        "dirichlet_alpha": 0.5,
        "client_epochs": [5] * 10,
        "centralized_epochs": 5,
    },
    "phase3_pair_label_equal": {
        "description": "Two-label grouping per client with equal training budget.",
        "split_mode": "pair-label",
        "client_epochs": [5] * 10,
        "centralized_epochs": 5,
    },
    "phase4_dirichlet_budget_hetero": {
        "description": "Dirichlet label skew with heterogeneous local training epochs.",
        "split_mode": "dirichlet",
        "dirichlet_alpha": 0.5,
        "client_epochs": [1, 2, 3, 5, 8, 10, 15, 20, 30, 50],
        "centralized_epochs": 10,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch experiments for MNIST merge lab")
    parser.add_argument("--data-root", type=Path, default=Path("./work/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/mnist_merge_lab/batch"))
    parser.add_argument(
        "--suite",
        type=str,
        default="core",
        choices=["core", "full"],
        help="core skips the most expensive presets; full runs everything.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional subset of experiment preset names.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--calibration-size", type=int, default=256)
    parser.add_argument("--regmean-lambda", type=float, default=1e-3)
    parser.add_argument(
        "--merge-methods",
        type=str,
        nargs="+",
        default=DEFAULT_MERGE_METHODS,
    )
    return parser.parse_args()


def experiment_names_for_suite(suite: str) -> List[str]:
    if suite == "full":
        return list(EXPERIMENT_PRESETS.keys())
    return [
        "phase0_iid_equal",
        "phase1_iid_budget_hetero",
        "phase2_dirichlet_medium",
        "phase4_dirichlet_budget_hetero",
    ]


def make_base_namespace(args: argparse.Namespace):
    namespace = build_parser().parse_args([])
    namespace.data_root = args.data_root
    namespace.device = args.device
    namespace.train_batch_size = args.train_batch_size
    namespace.eval_batch_size = args.eval_batch_size
    namespace.lr = args.lr
    namespace.weight_decay = args.weight_decay
    namespace.hidden_sizes = args.hidden_sizes
    namespace.calibration_size = args.calibration_size
    namespace.regmean_lambda = args.regmean_lambda
    namespace.merge_methods = args.merge_methods
    return namespace


def flatten_result_rows(experiment_name: str, seed: int, results: Dict[str, object]) -> List[Dict[str, object]]:
    config = results["config"]
    rows: List[Dict[str, object]] = []
    for method_name, payload in results["merges"].items():
        if method_name == "single_best":
            metrics = payload["metrics"]
            row = {
                "experiment": experiment_name,
                "seed": seed,
                "method": method_name,
                "split_mode": config["split_mode"],
                "dirichlet_alpha": config.get("dirichlet_alpha"),
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"],
                "macro_accuracy": metrics["macro_accuracy"],
                "worst_class_accuracy": metrics["worst_class_accuracy"],
                "mean_client_local_accuracy": "",
                "std_client_local_accuracy": "",
            }
        else:
            metrics = payload["global"]
            local_values = list(payload["per_client_local_accuracy"].values())
            row = {
                "experiment": experiment_name,
                "seed": seed,
                "method": method_name,
                "split_mode": config["split_mode"],
                "dirichlet_alpha": config.get("dirichlet_alpha"),
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"],
                "macro_accuracy": metrics["macro_accuracy"],
                "worst_class_accuracy": metrics["worst_class_accuracy"],
                "mean_client_local_accuracy": statistics.mean(local_values),
                "std_client_local_accuracy": statistics.pstdev(local_values) if len(local_values) > 1 else 0.0,
            }
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    selected_experiments = args.experiments or experiment_names_for_suite(args.suite)
    unknown = [name for name in selected_experiments if name not in EXPERIMENT_PRESETS]
    if unknown:
        raise ValueError(f"Unknown experiment preset(s): {', '.join(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    flat_rows: List[Dict[str, object]] = []
    all_results: Dict[str, Dict[str, object]] = {}
    base_namespace = make_base_namespace(args)

    for experiment_name in selected_experiments:
        preset = deepcopy(EXPERIMENT_PRESETS[experiment_name])
        all_results[experiment_name] = {"description": preset["description"], "runs": {}}
        for seed in args.seeds:
            run_namespace = deepcopy(base_namespace)
            run_namespace.seed = seed
            run_namespace.output_dir = runs_dir / experiment_name / f"seed_{seed}"
            run_namespace.num_clients = 10
            run_namespace.skip_centralized = False
            for key, value in preset.items():
                if key == "description":
                    continue
                setattr(run_namespace, key, value)

            results = run_experiment(run_namespace)
            run_namespace.output_dir.mkdir(parents=True, exist_ok=True)
            result_path = run_namespace.output_dir / "results.json"
            result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

            all_results[experiment_name]["runs"][str(seed)] = results
            flat_rows.extend(flatten_result_rows(experiment_name, seed, results))

    summary_json_path = args.output_dir / "batch_results.json"
    summary_json_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    flat_csv_path = args.output_dir / "summary_rows.csv"
    write_csv(flat_csv_path, flat_rows)

    aggregated_rows = aggregate_rows(flat_rows)
    aggregate_csv_path = args.output_dir / "aggregate_summary.csv"
    write_csv(aggregate_csv_path, aggregated_rows)

    aggregate_json_path = args.output_dir / "aggregate_summary.json"
    aggregate_json_path.write_text(json.dumps(aggregated_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = args.output_dir / "report.md"
    descriptions = {name: str(EXPERIMENT_PRESETS[name]["description"]) for name in selected_experiments}
    generate_summary_markdown(report_path, selected_experiments, aggregated_rows, descriptions)
    generate_plots(args.output_dir, selected_experiments, aggregated_rows)

    print(json.dumps(
        {
            "selected_experiments": selected_experiments,
            "seeds": args.seeds,
            "output_dir": str(args.output_dir),
            "files": [
                str(summary_json_path),
                str(flat_csv_path),
                str(aggregate_csv_path),
                str(aggregate_json_path),
                str(report_path),
                str(args.output_dir / "accuracy_comparison.svg"),
                str(args.output_dir / "regmean_vs_simple_delta.svg"),
            ],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
