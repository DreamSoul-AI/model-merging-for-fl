#!/usr/bin/env python3
"""
Second-pass experiment automation for:
1. Dirichlet alpha sweep
2. Calibration size sweep
3. Per-class recovery analysis
4. Per-client recovery analysis
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence

from core_merge_lab import ExperimentConfig, run_single_experiment, save_json
from report_utils_v2 import (
    aggregate_rows,
    build_report_markdown,
    flatten_summary_row,
    write_bar_svg,
    write_csv,
    write_line_svg,
)


GROUP_NOTES = {
    "alpha_sweep": "Fixed local training budget, sweep non-IID strength through Dirichlet alpha.",
    "calibration_sweep": "Fixed a representative non-IID setting, sweep calibration set size for RegMean statistics.",
    "per_class_recovery": "Measure which classes are recovered by RegMean relative to simple average.",
    "per_client_recovery": "Measure which client-local distributions are recovered by RegMean relative to simple average.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run second-pass MNIST merge experiments")
    parser.add_argument("--data-root", type=Path, default=Path("./work/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/second_pass_suite"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--regmean-lambda", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=["alpha_sweep", "calibration_sweep", "per_class_recovery", "per_client_recovery"],
    )
    return parser.parse_args()


def make_base_config(args: argparse.Namespace, seed: int) -> ExperimentConfig:
    return ExperimentConfig(
        data_root=args.data_root,
        device=args.device,
        seed=seed,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_sizes=tuple(args.hidden_sizes),
        regmean_lambda=args.regmean_lambda,
    )


def run_group_setting(
    output_dir: Path,
    group: str,
    setting: str,
    config: ExperimentConfig,
) -> Dict[str, object]:
    result = run_single_experiment(config)
    save_json(output_dir / group / setting / f"seed_{config.seed}.json", result)
    return result


def build_summary_rows(group: str, setting: str, seed: int, result: Dict[str, object]) -> List[Dict[str, object]]:
    return [
        flatten_summary_row(group, setting, seed, method, metrics)
        for method, metrics in result["merges"].items()
    ]


def mean_metric_for_method(rows: Sequence[Dict[str, object]], group: str, method: str, field: str) -> Dict[str, float]:
    selected = [row for row in rows if row["group"] == group and row["method"] == method]
    return {str(row["setting"]): float(row[f"{field}_mean"]) for row in selected if row[f"{field}_mean"] != ""}


def make_alpha_sweep_configs(base: ExperimentConfig) -> Dict[str, ExperimentConfig]:
    configs = {}
    for alpha in [5.0, 1.0, 0.5, 0.1]:
        config = deepcopy(base)
        config.split_mode = "dirichlet"
        config.dirichlet_alpha = alpha
        config.client_epochs = tuple([5] * 10)
        config.centralized_epochs = 5
        configs[f"alpha_{alpha}"] = config
    return configs


def make_calibration_sweep_configs(base: ExperimentConfig) -> Dict[str, ExperimentConfig]:
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


def make_per_class_configs(base: ExperimentConfig) -> Dict[str, ExperimentConfig]:
    config = deepcopy(base)
    config.split_mode = "dirichlet"
    config.dirichlet_alpha = 0.5
    config.client_epochs = tuple([5] * 10)
    config.centralized_epochs = 5
    return {"dirichlet_medium": config}


def make_per_client_configs(base: ExperimentConfig) -> Dict[str, ExperimentConfig]:
    config = deepcopy(base)
    config.split_mode = "dirichlet"
    config.dirichlet_alpha = 0.5
    config.client_epochs = tuple([1, 2, 3, 5, 8, 10, 15, 20, 30, 50])
    config.centralized_epochs = 10
    return {"dirichlet_budget_hetero": config}


def render_alpha_plots(output_dir: Path, aggregate_rows: Sequence[Dict[str, object]]) -> None:
    labels = ["alpha_5.0", "alpha_1.0", "alpha_0.5", "alpha_0.1"]
    series = []
    methods = ["simple_avg", "epoch_weighted_avg", "regmean_equal", "regmean_epoch_weighted"]
    colors = ["#2563eb", "#059669", "#dc2626", "#d97706"]
    for method, color in zip(methods, colors):
        lookup = mean_metric_for_method(aggregate_rows, "alpha_sweep", method, "accuracy")
        series.append({"label": method, "color": color, "values": [lookup.get(label, 0.0) for label in labels]})
    write_line_svg(output_dir / "alpha_sweep_accuracy.svg", "Alpha sweep accuracy", labels, series, "Accuracy")

    simple = mean_metric_for_method(aggregate_rows, "alpha_sweep", "simple_avg", "accuracy")
    regmean = mean_metric_for_method(aggregate_rows, "alpha_sweep", "regmean_epoch_weighted", "accuracy")
    deltas = [regmean.get(label, 0.0) - simple.get(label, 0.0) for label in labels]
    write_bar_svg(output_dir / "alpha_sweep_regmean_delta.svg", "RegMean minus simple across alpha", labels, deltas, "Accuracy delta")


def render_calibration_plots(output_dir: Path, aggregate_rows: Sequence[Dict[str, object]]) -> None:
    labels = ["calib_16", "calib_32", "calib_64", "calib_128", "calib_256", "calib_512"]
    methods = ["simple_avg", "epoch_weighted_avg", "regmean_equal", "regmean_epoch_weighted"]
    colors = ["#2563eb", "#059669", "#dc2626", "#d97706"]
    series = []
    for method, color in zip(methods, colors):
        lookup = mean_metric_for_method(aggregate_rows, "calibration_sweep", method, "accuracy")
        series.append({"label": method, "color": color, "values": [lookup.get(label, 0.0) for label in labels]})
    write_line_svg(output_dir / "calibration_sweep_accuracy.svg", "Calibration sweep accuracy", labels, series, "Accuracy")


def summarize_recovery_outputs(
    output_dir: Path,
    group: str,
    setting: str,
    seed_results: Sequence[Dict[str, object]],
    key_mode: str,
) -> None:
    keys = []
    recovery_rows = []
    for result in seed_results:
        simple = result["merges"]["simple_avg"]
        regmean = result["merges"]["regmean_epoch_weighted"]
        if key_mode == "class":
            simple_metrics = simple["global"]
            regmean_metrics = regmean["global"]
            keys = [f"class_{i}_acc" for i in range(10)]
            for key in keys:
                recovery_rows.append({"key": key, "delta": regmean_metrics[key] - simple_metrics[key]})
        else:
            simple_metrics = simple["per_client_local_accuracy"]
            regmean_metrics = regmean["per_client_local_accuracy"]
            keys = list(simple_metrics.keys())
            for key in keys:
                recovery_rows.append({"key": key, "delta": regmean_metrics[key] - simple_metrics[key]})

    grouped: Dict[str, List[float]] = {}
    for row in recovery_rows:
        grouped.setdefault(row["key"], []).append(row["delta"])
    summary = [{"item": key, "delta_mean": sum(values) / len(values)} for key, values in grouped.items()]
    summary.sort(key=lambda item: item["item"])
    write_csv(output_dir / group / setting / f"{key_mode}_recovery.csv", summary)
    write_bar_svg(
        output_dir / group / setting / f"{key_mode}_recovery.svg",
        f"{group} {key_mode} recovery",
        [row["item"] for row in summary],
        [float(row["delta_mean"]) for row in summary],
        "RegMean - simple",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flat_rows: List[Dict[str, object]] = []
    grouped_seed_results: Dict[str, Dict[str, List[Dict[str, object]]]] = {}

    for seed in args.seeds:
        base = make_base_config(args, seed)
        group_to_configs = {
            "alpha_sweep": make_alpha_sweep_configs(base),
            "calibration_sweep": make_calibration_sweep_configs(base),
            "per_class_recovery": make_per_class_configs(base),
            "per_client_recovery": make_per_client_configs(base),
        }
        for group in args.groups:
            configs = group_to_configs[group]
            for setting, config in configs.items():
                print(f"[run] seed={seed} group={group} setting={setting}", flush=True)
                result = run_group_setting(args.output_dir, group, setting, config)
                flat_rows.extend(build_summary_rows(group, setting, seed, result))
                grouped_seed_results.setdefault(group, {}).setdefault(setting, []).append(result)

    write_csv(args.output_dir / "summary_rows.csv", flat_rows)
    aggregate = aggregate_rows(flat_rows)
    write_csv(args.output_dir / "aggregate_summary.csv", aggregate)
    (args.output_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    build_report_markdown(args.output_dir / "report.md", aggregate, GROUP_NOTES)

    if "alpha_sweep" in args.groups:
        render_alpha_plots(args.output_dir, aggregate)
    if "calibration_sweep" in args.groups:
        render_calibration_plots(args.output_dir, aggregate)
    if "per_class_recovery" in args.groups:
        summarize_recovery_outputs(args.output_dir, "per_class_recovery", "dirichlet_medium", grouped_seed_results["per_class_recovery"]["dirichlet_medium"], "class")
    if "per_client_recovery" in args.groups:
        summarize_recovery_outputs(
            args.output_dir,
            "per_client_recovery",
            "dirichlet_budget_hetero",
            grouped_seed_results["per_client_recovery"]["dirichlet_budget_hetero"],
            "client",
        )

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "files": [
                    str(args.output_dir / "summary_rows.csv"),
                    str(args.output_dir / "aggregate_summary.csv"),
                    str(args.output_dir / "aggregate_summary.json"),
                    str(args.output_dir / "report.md"),
                ],
                "groups": args.groups,
                "seeds": args.seeds,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
