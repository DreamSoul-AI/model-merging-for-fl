#!/usr/bin/env python3
"""
Standalone plot builder for advanced MNIST+MLP experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

from report_utils_v2 import aggregate_rows, write_bar_svg, write_csv, write_line_svg


COLORS = {
    "simple_avg": "#2563eb",
    "sample_weighted_avg": "#0891b2",
    "epoch_weighted_avg": "#059669",
    "val_loss_weighted_avg": "#8b5cf6",
    "regmean_equal": "#dc2626",
    "regmean_sample_weighted": "#ea580c",
    "regmean_epoch_weighted": "#d97706",
    "regmean_val_loss_weighted": "#b45309",
    "centralized": "#7c3aed",
    "regmean_all_layers": "#dc2626",
    "regmean_hidden_layers": "#0891b2",
    "regmean_last_layer": "#d97706",
    "sample_weighted_last_layer": "#059669",
    "simple_bias": "#2563eb",
    "regmean_bias": "#dc2626",
    "simple_ridge": "#0891b2",
    "regmean_ridge": "#d97706",
    "simple_lastlayer_tune": "#7c3aed",
    "regmean_lastlayer_tune": "#b45309",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build plots for advanced MNIST results")
    parser.add_argument("--results-dir", type=Path, default=Path("./outputs/mnist_advanced_suite"))
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def group_rows(rows: Sequence[Dict[str, object]], group: str) -> List[Dict[str, object]]:
    return [row for row in rows if str(row["group"]) == group]


def mean_lookup(rows: Sequence[Dict[str, object]], metric: str = "accuracy_mean") -> Dict[str, float]:
    return {
        str(row["method"]): float(row[metric])
        for row in rows
        if row.get(metric, "") != ""
    }


def save_aggregate_if_needed(results_dir: Path) -> List[Dict[str, object]]:
    summary_path = results_dir / "summary_rows.csv"
    aggregate_path = results_dir / "aggregate_summary.csv"
    if aggregate_path.exists():
        return read_rows(aggregate_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing both {summary_path} and {aggregate_path}")
    summary_rows = read_rows(summary_path)
    aggregate = aggregate_rows(summary_rows)
    write_csv(aggregate_path, aggregate)
    (results_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    return aggregate


def plot_extreme_non_iid(results_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    lookup = mean_lookup(group_rows(rows, "extreme_non_iid"))
    methods = ["simple_avg", "sample_weighted_avg", "epoch_weighted_avg", "regmean_equal", "regmean_epoch_weighted", "centralized"]
    labels = methods
    values = [lookup.get(method, 0.0) for method in methods]
    write_bar_svg(results_dir / "extreme_non_iid_accuracy.svg", "Extreme non-IID accuracy", labels, values, "Accuracy")

    delta_methods = ["regmean_equal", "regmean_epoch_weighted"]
    deltas = [lookup.get(method, 0.0) - lookup.get("simple_avg", 0.0) for method in delta_methods]
    write_bar_svg(results_dir / "extreme_non_iid_regmean_delta.svg", "Extreme non-IID RegMean minus simple", delta_methods, deltas, "Accuracy delta")


def plot_layerwise_ablation(results_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    lookup = mean_lookup(group_rows(rows, "layerwise_ablation"))
    methods = ["simple_avg", "regmean_all_layers", "regmean_hidden_layers", "regmean_last_layer", "sample_weighted_last_layer", "centralized"]
    values = [lookup.get(method, 0.0) for method in methods]
    write_bar_svg(results_dir / "layerwise_ablation_accuracy.svg", "Layer-wise ablation accuracy", methods, values, "Accuracy")


def plot_weighting_ablation(results_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    lookup = mean_lookup(group_rows(rows, "weighting_ablation"))
    methods = [
        "simple_avg",
        "sample_weighted_avg",
        "epoch_weighted_avg",
        "val_loss_weighted_avg",
        "regmean_equal",
        "regmean_sample_weighted",
        "regmean_epoch_weighted",
        "regmean_val_loss_weighted",
    ]
    values = [lookup.get(method, 0.0) for method in methods]
    write_bar_svg(results_dir / "weighting_ablation_accuracy.svg", "Weighting ablation accuracy", methods, values, "Accuracy")


def plot_few_shot_recovery(results_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    lookup = mean_lookup(group_rows(rows, "few_shot_recovery"))
    labels = ["base", "1-shot", "5-shot", "10-shot"]
    series = [
        {
            "label": "simple_bias",
            "color": COLORS["simple_bias"],
            "values": [
                lookup.get("simple_avg", 0.0),
                lookup.get("simple_bias_1shot", 0.0),
                lookup.get("simple_bias_5shot", 0.0),
                lookup.get("simple_bias_10shot", 0.0),
            ],
        },
        {
            "label": "regmean_bias",
            "color": COLORS["regmean_bias"],
            "values": [
                lookup.get("regmean_equal", 0.0),
                lookup.get("regmean_bias_1shot", 0.0),
                lookup.get("regmean_bias_5shot", 0.0),
                lookup.get("regmean_bias_10shot", 0.0),
            ],
        },
        {
            "label": "simple_ridge",
            "color": COLORS["simple_ridge"],
            "values": [
                lookup.get("simple_avg", 0.0),
                lookup.get("simple_ridge_1shot", 0.0),
                lookup.get("simple_ridge_5shot", 0.0),
                lookup.get("simple_ridge_10shot", 0.0),
            ],
        },
        {
            "label": "regmean_ridge",
            "color": COLORS["regmean_ridge"],
            "values": [
                lookup.get("regmean_equal", 0.0),
                lookup.get("regmean_ridge_1shot", 0.0),
                lookup.get("regmean_ridge_5shot", 0.0),
                lookup.get("regmean_ridge_10shot", 0.0),
            ],
        },
        {
            "label": "simple_lastlayer_tune",
            "color": COLORS["simple_lastlayer_tune"],
            "values": [
                lookup.get("simple_avg", 0.0),
                lookup.get("simple_lastlayer_tune_1shot", 0.0),
                lookup.get("simple_lastlayer_tune_5shot", 0.0),
                lookup.get("simple_lastlayer_tune_10shot", 0.0),
            ],
        },
        {
            "label": "regmean_lastlayer_tune",
            "color": COLORS["regmean_lastlayer_tune"],
            "values": [
                lookup.get("regmean_equal", 0.0),
                lookup.get("regmean_lastlayer_tune_1shot", 0.0),
                lookup.get("regmean_lastlayer_tune_5shot", 0.0),
                lookup.get("regmean_lastlayer_tune_10shot", 0.0),
            ],
        },
        {
            "label": "centralized",
            "color": COLORS["centralized"],
            "values": [lookup.get("centralized", 0.0)] * 4,
        },
    ]
    write_line_svg(results_dir / "few_shot_recovery_accuracy.svg", "Few-shot recovery accuracy", labels, series, "Accuracy")


def plot_gap_decomposition(results_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    lookup = mean_lookup(group_rows(rows, "gap_decomposition"))
    labels = ["merge", "head_refit", "short_tune"]
    series = [
        {
            "label": "simple_avg",
            "color": COLORS["simple_avg"],
            "values": [
                lookup.get("simple_avg", 0.0),
                lookup.get("simple_avg_head_refit", 0.0),
                lookup.get("simple_avg_short_finetune", 0.0),
            ],
        },
        {
            "label": "regmean_equal",
            "color": COLORS["regmean_equal"],
            "values": [
                lookup.get("regmean_equal", 0.0),
                lookup.get("regmean_equal_head_refit", 0.0),
                lookup.get("regmean_equal_short_finetune", 0.0),
            ],
        },
        {
            "label": "centralized",
            "color": COLORS["centralized"],
            "values": [lookup.get("centralized", 0.0)] * 3,
        },
    ]
    write_line_svg(results_dir / "gap_decomposition_accuracy.svg", "Gap decomposition accuracy", labels, series, "Accuracy")


def main() -> None:
    args = parse_args()
    aggregate_rows_data = save_aggregate_if_needed(args.results_dir)

    plot_few_shot_recovery(args.results_dir, aggregate_rows_data)
    plot_extreme_non_iid(args.results_dir, aggregate_rows_data)
    plot_gap_decomposition(args.results_dir, aggregate_rows_data)
    plot_layerwise_ablation(args.results_dir, aggregate_rows_data)
    plot_weighting_ablation(args.results_dir, aggregate_rows_data)

    print(
        json.dumps(
            {
                "results_dir": str(args.results_dir),
                "files": [
                    str(args.results_dir / "few_shot_recovery_accuracy.svg"),
                    str(args.results_dir / "extreme_non_iid_accuracy.svg"),
                    str(args.results_dir / "extreme_non_iid_regmean_delta.svg"),
                    str(args.results_dir / "gap_decomposition_accuracy.svg"),
                    str(args.results_dir / "layerwise_ablation_accuracy.svg"),
                    str(args.results_dir / "weighting_ablation_accuracy.svg"),
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
