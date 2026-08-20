#!/usr/bin/env python3
"""
Runner for CNN merge experiments on CIFAR.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

from cnn_merge_lab import CNNExperimentConfig, run_cnn_experiment, save_json
from report_utils_v2 import aggregate_rows, build_report_markdown, flatten_summary_row, write_bar_svg, write_csv, write_line_svg


GROUP_NOTES = {
    "cnn_alpha_sweep": "CIFAR Dirichlet alpha sweep on a small CNN to test how merge quality changes with stronger non-IID class skew.",
    "cnn_calibration_sweep": "CIFAR calibration sweep under heterogeneous training budgets to test how much calibration data RegMean needs.",
    "cnn_feature_skew": "CIFAR feature-skew comparison to test whether RegMean helps when clients observe different visual styles.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CNN merge experiments")
    parser.add_argument("--data-root", type=Path, default=Path("./work/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/cnn_merge_suite"))
    parser.add_argument("--dataset-name", type=str, choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--regmean-lambda", type=float, default=1e-3)
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=["cnn_alpha_sweep", "cnn_calibration_sweep", "cnn_feature_skew"],
    )
    return parser.parse_args()


def make_base_config(args: argparse.Namespace, seed: int) -> CNNExperimentConfig:
    return CNNExperimentConfig(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        device=args.device,
        seed=seed,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        width=args.width,
        regmean_lambda=args.regmean_lambda,
    )


def alpha_sweep_configs(base: CNNExperimentConfig) -> Dict[str, CNNExperimentConfig]:
    configs = {}
    for alpha in [5.0, 1.0, 0.5, 0.1]:
        config = deepcopy(base)
        config.split_mode = "dirichlet"
        config.dirichlet_alpha = alpha
        config.client_epochs = tuple([5] * config.num_clients)
        config.centralized_epochs = 5
        configs[f"alpha_{alpha}"] = config
    return configs


def calibration_sweep_configs(base: CNNExperimentConfig) -> Dict[str, CNNExperimentConfig]:
    configs = {}
    for size in [16, 32, 64, 128, 256, 512]:
        config = deepcopy(base)
        config.split_mode = "dirichlet"
        config.dirichlet_alpha = 0.5
        config.client_epochs = tuple([1, 2, 3, 5, 8, 10, 15, 20, 30, 50])
        config.centralized_epochs = 10
        config.calibration_size = size
        configs[f"calib_{size}"] = config
    return configs


def feature_skew_configs(base: CNNExperimentConfig) -> Dict[str, CNNExperimentConfig]:
    configs = {}
    for mode in ["rotation", "color"]:
        config = deepcopy(base)
        config.split_mode = "iid"
        config.feature_skew_mode = mode
        config.client_epochs = tuple([5] * config.num_clients)
        config.centralized_epochs = 5
        configs[f"feature_{mode}"] = config
    return configs


def render_alpha_plot(output_dir: Path, aggregate_rows: List[Dict[str, object]]) -> None:
    labels = ["alpha_5.0", "alpha_1.0", "alpha_0.5", "alpha_0.1"]
    methods = ["simple_avg", "epoch_weighted_avg", "regmean_equal", "regmean_epoch_weighted"]
    colors = ["#2563eb", "#059669", "#dc2626", "#d97706"]
    series = []
    for method, color in zip(methods, colors):
        lookup = {str(row["setting"]): float(row["accuracy_mean"]) for row in aggregate_rows if row["group"] == "cnn_alpha_sweep" and row["method"] == method and row["accuracy_mean"] != ""}
        series.append({"label": method, "color": color, "values": [lookup.get(label, 0.0) for label in labels]})
    write_line_svg(output_dir / "cnn_alpha_sweep_accuracy.svg", "CNN alpha sweep accuracy", labels, series, "Accuracy")
    simple = {str(row["setting"]): float(row["accuracy_mean"]) for row in aggregate_rows if row["group"] == "cnn_alpha_sweep" and row["method"] == "simple_avg" and row["accuracy_mean"] != ""}
    regmean = {str(row["setting"]): float(row["accuracy_mean"]) for row in aggregate_rows if row["group"] == "cnn_alpha_sweep" and row["method"] == "regmean_epoch_weighted" and row["accuracy_mean"] != ""}
    deltas = [regmean.get(label, 0.0) - simple.get(label, 0.0) for label in labels]
    write_bar_svg(output_dir / "cnn_alpha_sweep_regmean_delta.svg", "CNN RegMean minus simple across alpha", labels, deltas, "Accuracy delta")


def render_calibration_plot(output_dir: Path, aggregate_rows: List[Dict[str, object]]) -> None:
    labels = ["calib_16", "calib_32", "calib_64", "calib_128", "calib_256", "calib_512"]
    methods = ["simple_avg", "epoch_weighted_avg", "regmean_equal", "regmean_epoch_weighted"]
    colors = ["#2563eb", "#059669", "#dc2626", "#d97706"]
    series = []
    for method, color in zip(methods, colors):
        lookup = {str(row["setting"]): float(row["accuracy_mean"]) for row in aggregate_rows if row["group"] == "cnn_calibration_sweep" and row["method"] == method and row["accuracy_mean"] != ""}
        series.append({"label": method, "color": color, "values": [lookup.get(label, 0.0) for label in labels]})
    write_line_svg(output_dir / "cnn_calibration_sweep_accuracy.svg", "CNN calibration sweep accuracy", labels, series, "Accuracy")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flat_rows: List[Dict[str, object]] = []

    for seed in args.seeds:
        base = make_base_config(args, seed)
        group_configs = {
            "cnn_alpha_sweep": alpha_sweep_configs(base),
            "cnn_calibration_sweep": calibration_sweep_configs(base),
            "cnn_feature_skew": feature_skew_configs(base),
        }
        for group in args.groups:
            for setting, config in group_configs[group].items():
                print(f"[cnn-run] seed={seed} group={group} setting={setting}", flush=True)
                result = run_cnn_experiment(config)
                save_json(args.output_dir / group / setting / f"seed_{seed}.json", result)
                flat_rows.extend(
                    [flatten_summary_row(group, setting, seed, method, metrics) for method, metrics in result["merges"].items()]
                )

    write_csv(args.output_dir / "summary_rows.csv", flat_rows)
    aggregate = aggregate_rows(flat_rows)
    write_csv(args.output_dir / "aggregate_summary.csv", aggregate)
    (args.output_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    build_report_markdown(args.output_dir / "report.md", aggregate, GROUP_NOTES)

    if "cnn_alpha_sweep" in args.groups:
        render_alpha_plot(args.output_dir, aggregate)
    if "cnn_calibration_sweep" in args.groups:
        render_calibration_plot(args.output_dir, aggregate)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "groups": args.groups,
                "files": [
                    str(args.output_dir / "summary_rows.csv"),
                    str(args.output_dir / "aggregate_summary.csv"),
                    str(args.output_dir / "aggregate_summary.json"),
                    str(args.output_dir / "report.md"),
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
