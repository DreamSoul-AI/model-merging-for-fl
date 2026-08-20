#!/usr/bin/env python3
"""
Standalone plot and markdown builder for gap diagnostics.
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
    "regmean_equal": "#dc2626",
    "regmean_sample_weighted": "#d97706",
    "regmean_hidden_layers": "#7c3aed",
    "centralized": "#059669",
    "simple_avg_bias_cal": "#60a5fa",
    "simple_avg_ridge_head": "#1d4ed8",
    "simple_avg_short_tune": "#0f766e",
    "regmean_equal_bias_cal": "#f87171",
    "regmean_equal_ridge_head": "#b91c1c",
    "regmean_equal_short_tune": "#c2410c",
    "sample_weighted_avg_bias_cal": "#22c55e",
    "sample_weighted_avg_ridge_head": "#15803d",
    "sample_weighted_avg_short_tune": "#166534",
    "regmean_sample_weighted_ridge_head": "#92400e",
    "regmean_hidden_layers_bias_cal": "#8b5cf6",
    "regmean_hidden_layers_short_tune": "#6d28d9",
    "regmean_hidden_layers_temperature": "#4c1d95",
    "regmean_hidden_layers_vector_scaling": "#581c87",
    "regmean_hidden_layers_last_layer_tune": "#4338ca",
    "regmean_hidden_layers_hidden_head_tune": "#312e81",
    "sample_weighted_bias_cal": "#06b6d4",
    "sample_weighted_temperature": "#0284c7",
    "sample_weighted_vector_scaling": "#0369a1",
    "sample_weighted_last_layer_tune": "#0f766e",
    "sample_weighted_hidden_head_tune": "#14532d",
    "sample_weighted_middle_layer_tune": "#15803d",
    "sample_weighted_full_tune": "#166534",
    "regmean_last_layer": "#a16207",
    "regmean_middle_last": "#b45309",
    "regmean_all_layers": "#c2410c",
    "regmean_hidden_last_layer_tune": "#5b21b6",
    "regmean_hidden_middle_layer_tune": "#6d28d9",
    "regmean_hidden_full_tune": "#7c3aed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build plots and markdown for gap diagnostics")
    parser.add_argument("--results-dir", type=Path, default=Path("./outputs/gap_diagnostics_suite"))
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def save_aggregate_if_needed(results_dir: Path) -> List[Dict[str, object]]:
    summary_path = results_dir / "summary_rows.csv"
    aggregate_path = results_dir / "aggregate_summary.csv"
    if aggregate_path.exists():
        return read_csv_rows(aggregate_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing both {summary_path} and {aggregate_path}")
    aggregate = aggregate_rows(read_csv_rows(summary_path))
    write_csv(aggregate_path, aggregate)
    (results_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    return aggregate


def read_diagnostics(results_dir: Path) -> Dict[str, object]:
    diagnostics_path = results_dir / "diagnostics.json"
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"Missing diagnostics file: {diagnostics_path}")
    return json.loads(diagnostics_path.read_text(encoding="utf-8"))


def rows_for_group(rows: Sequence[Dict[str, object]], group: str) -> List[Dict[str, object]]:
    return [row for row in rows if str(row["group"]) == group]


def mean_lookup(rows: Sequence[Dict[str, object]], metric: str = "accuracy_mean") -> Dict[str, float]:
    return {str(row["method"]): float(row[metric]) for row in rows if row.get(metric, "") != ""}


def diagnostic_mean(diagnostics_group: Dict[str, object], metric: str, methods: Sequence[str]) -> Dict[str, float]:
    sums = {method: 0.0 for method in methods}
    counts = {method: 0 for method in methods}
    for _, seed_payload in diagnostics_group.items():
        for method in methods:
            if method in seed_payload and metric in seed_payload[method]:
                sums[method] += float(seed_payload[method][metric])
                counts[method] += 1
    return {method: (sums[method] / counts[method] if counts[method] else 0.0) for method in methods}


def sorted_shot_settings(rows: Sequence[Dict[str, object]], group: str) -> List[str]:
    return sorted(
        {str(row["setting"]) for row in rows if str(row["group"]) == group},
        key=lambda x: int(x.split("shot", 1)[0]) if "shot" in x else x,
    )


def build_markdown(results_dir: Path, aggregate_rows_data: Sequence[Dict[str, object]], diagnostics: Dict[str, object]) -> None:
    lines = ["# Gap Diagnostics Summary", ""]

    gap_rows = rows_for_group(aggregate_rows_data, "gap_sources")
    if gap_rows:
        lines.append("## Gap Sources")
        lines.append("")
        lines.append("| Method | Acc mean | Macro mean | Worst-class mean |")
        lines.append("|---|---:|---:|---:|")
        for row in gap_rows:
            lines.append(
                f"| {row['method']} | {float(row['accuracy_mean']):.4f} | {float(row['macro_accuracy_mean']):.4f} | {float(row['worst_class_accuracy_mean']):.4f} |"
            )
        lines.append("")

    function_group = diagnostics.get("function_space", {})
    if function_group:
        methods = ["simple_avg", "sample_weighted_avg", "regmean_equal", "regmean_sample_weighted", "regmean_hidden_layers", "centralized"]
        kl = diagnostic_mean(function_group, "symmetric_kl_to_centralized", methods)
        agree = diagnostic_mean(function_group, "prediction_agreement_to_centralized", methods)
        lines.append("## Function-space Diagnostics")
        lines.append("")
        lines.append("| Method | Symmetric KL to centralized | Prediction agreement |")
        lines.append("|---|---:|---:|")
        for method in methods:
            lines.append(f"| {method} | {kl[method]:.4f} | {agree[method]:.4f} |")
        lines.append("")

    repr_group = diagnostics.get("representation_space", {})
    if repr_group:
        methods = ["simple_avg", "sample_weighted_avg", "regmean_equal", "regmean_sample_weighted", "regmean_hidden_layers", "centralized"]
        pen = diagnostic_mean(repr_group, "penultimate_cosine_to_centralized", methods)
        l0 = diagnostic_mean(repr_group, "layer0_cosine_to_centralized", methods)
        l1 = diagnostic_mean(repr_group, "layer1_cosine_to_centralized", methods)
        l2 = diagnostic_mean(repr_group, "layer2_cosine_to_centralized", methods)
        lines.append("## Representation-space Diagnostics")
        lines.append("")
        lines.append("| Method | Penultimate cosine | Layer0 cosine | Layer1 cosine | Layer2 cosine |")
        lines.append("|---|---:|---:|---:|---:|")
        for method in methods:
            lines.append(f"| {method} | {pen[method]:.4f} | {l0[method]:.4f} | {l1[method]:.4f} | {l2[method]:.4f} |")
        lines.append("")

    reduction_rows = rows_for_group(aggregate_rows_data, "gap_reduction")
    if reduction_rows:
        lines.append("## Gap Reduction Candidates")
        lines.append("")
        lines.append("| Method | Acc mean | Macro mean | Worst-class mean |")
        lines.append("|---|---:|---:|---:|")
        for row in reduction_rows:
            lines.append(
                f"| {row['method']} | {float(row['accuracy_mean']):.4f} | {float(row['macro_accuracy_mean']):.4f} | {float(row['worst_class_accuracy_mean']):.4f} |"
            )
        lines.append("")

    calibration_rows = rows_for_group(aggregate_rows_data, "systematic_calibration")
    if calibration_rows:
        lines.append("## Systematic Calibration")
        lines.append("")
        for setting in sorted({str(row["setting"]) for row in calibration_rows}):
            lines.append(f"### {setting}")
            lines.append("")
            lines.append("| Method | Acc mean | Macro mean | Worst-class mean |")
            lines.append("|---|---:|---:|---:|")
            for row in [row for row in calibration_rows if str(row["setting"]) == setting]:
                lines.append(
                    f"| {row['method']} | {float(row['accuracy_mean']):.4f} | {float(row['macro_accuracy_mean']):.4f} | {float(row['worst_class_accuracy_mean']):.4f} |"
                )
            lines.append("")

    if function_group and repr_group:
        methods = [
            "sample_weighted_avg",
            "regmean_sample_weighted",
            "regmean_hidden_layers",
            "regmean_hidden_layers_last_layer_tune",
            "regmean_hidden_layers_hidden_head_tune",
        ]
        kl = diagnostic_mean(function_group, "symmetric_kl_to_centralized", methods)
        pen = diagnostic_mean(repr_group, "penultimate_cosine_to_centralized", methods)
        deep_drop = diagnostic_mean(repr_group, "deep_representation_drop", methods)
        lines.append("## Mainline Hypothesis Check")
        lines.append("")
        lines.append("| Method | Symmetric KL | Penultimate cosine | Deep-drop (layer0 - penultimate) |")
        lines.append("|---|---:|---:|---:|")
        for method in methods:
            lines.append(f"| {method} | {kl[method]:.4f} | {pen[method]:.4f} | {deep_drop[method]:.4f} |")
        lines.append("")

    calibration_strength_rows = rows_for_group(aggregate_rows_data, "calibration_strength")
    if calibration_strength_rows:
        lines.append("## Calibration Strength")
        lines.append("")
        for setting in sorted_shot_settings(aggregate_rows_data, "calibration_strength"):
            lines.append(f"### {setting}")
            lines.append("")
            lines.append("| Method | Acc mean | Macro mean | Worst-class mean |")
            lines.append("|---|---:|---:|---:|")
            for row in [row for row in calibration_strength_rows if str(row["setting"]) == setting]:
                lines.append(
                    f"| {row['method']} | {float(row['accuracy_mean']):.4f} | {float(row['macro_accuracy_mean']):.4f} | {float(row['worst_class_accuracy_mean']):.4f} |"
                )
            lines.append("")

    layerwise_alignment_group = diagnostics.get("layerwise_alignment", {})
    if layerwise_alignment_group:
        methods = [
            "simple_avg",
            "sample_weighted_avg",
            "sample_weighted_avg_last_layer_tune",
            "sample_weighted_avg_hidden_head_tune",
            "regmean_hidden_layers",
            "regmean_hidden_layers_last_layer_tune",
            "regmean_hidden_layers_hidden_head_tune",
            "centralized",
        ]
        layer2_cka = diagnostic_mean(layerwise_alignment_group, "layer2_cka_to_centralized", methods)
        pen_cka = diagnostic_mean(layerwise_alignment_group, "penultimate_cka_to_centralized", methods)
        center = diagnostic_mean(layerwise_alignment_group, "penultimate_center_cosine_to_centralized", methods)
        lines.append("## Layerwise Alignment")
        lines.append("")
        lines.append("| Method | Layer2 CKA | Penultimate CKA | Penultimate class-center cosine |")
        lines.append("|---|---:|---:|---:|")
        for method in methods:
            lines.append(f"| {method} | {layer2_cka[method]:.4f} | {pen_cka[method]:.4f} | {center[method]:.4f} |")
        lines.append("")

    layerwise_intervention_rows = rows_for_group(aggregate_rows_data, "layerwise_intervention")
    if layerwise_intervention_rows:
        lines.append("## Layerwise Intervention")
        lines.append("")
        for setting in sorted({str(row["setting"]) for row in layerwise_intervention_rows}):
            lines.append(f"### {setting}")
            lines.append("")
            lines.append("| Method | Acc mean | Macro mean | Worst-class mean |")
            lines.append("|---|---:|---:|---:|")
            for row in [row for row in layerwise_intervention_rows if str(row["setting"]) == setting]:
                lines.append(
                    f"| {row['method']} | {float(row['accuracy_mean']):.4f} | {float(row['macro_accuracy_mean']):.4f} | {float(row['worst_class_accuracy_mean']):.4f} |"
                )
            lines.append("")

    function_repr_bridge_group = diagnostics.get("function_repr_bridge", {})
    if function_repr_bridge_group:
        methods = [
            "sample_weighted_avg",
            "sample_weighted_avg_last_layer_tune",
            "sample_weighted_avg_hidden_head_tune",
            "regmean_hidden_layers",
            "regmean_hidden_layers_last_layer_tune",
            "regmean_hidden_layers_hidden_head_tune",
        ]
        kl = diagnostic_mean(function_repr_bridge_group, "symmetric_kl_to_centralized", methods)
        pen = diagnostic_mean(function_repr_bridge_group, "penultimate_cka_to_centralized", methods)
        acc = diagnostic_mean(function_repr_bridge_group, "subset_accuracy", methods)
        lines.append("## Function-Representation Bridge")
        lines.append("")
        lines.append("| Method | Subset accuracy | Symmetric KL | Penultimate CKA |")
        lines.append("|---|---:|---:|---:|")
        for method in methods:
            lines.append(f"| {method} | {acc[method]:.4f} | {kl[method]:.4f} | {pen[method]:.4f} |")
        lines.append("")

    (results_dir / "diagnostics_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_gap_sources(results_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    lookup = mean_lookup(rows_for_group(rows, "gap_sources"))
    methods = [
        "simple_avg",
        "simple_avg_bias_cal",
        "simple_avg_ridge_head",
        "simple_avg_short_tune",
        "regmean_equal",
        "regmean_equal_bias_cal",
        "regmean_equal_ridge_head",
        "regmean_equal_short_tune",
        "sample_weighted_avg",
        "sample_weighted_avg_bias_cal",
        "sample_weighted_avg_ridge_head",
        "sample_weighted_avg_short_tune",
        "centralized",
    ]
    write_bar_svg(
        results_dir / "gap_sources_accuracy.svg",
        "Gap source interventions accuracy",
        methods,
        [lookup.get(method, 0.0) for method in methods],
        "Accuracy",
    )


def plot_gap_reduction(results_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    lookup = mean_lookup(rows_for_group(rows, "gap_reduction"))
    methods = [
        "simple_avg",
        "sample_weighted_avg",
        "regmean_equal",
        "regmean_sample_weighted",
        "regmean_hidden_layers",
        "simple_avg_ridge_head",
        "regmean_sample_weighted_ridge_head",
        "regmean_hidden_layers_bias_cal",
        "regmean_hidden_layers_short_tune",
        "centralized",
    ]
    write_bar_svg(
        results_dir / "gap_reduction_accuracy.svg",
        "Gap reduction candidates",
        methods,
        [lookup.get(method, 0.0) for method in methods],
        "Accuracy",
    )


def plot_function_space(results_dir: Path, diagnostics: Dict[str, object]) -> None:
    function_group = diagnostics.get("function_space", {})
    methods = [
        "simple_avg",
        "sample_weighted_avg",
        "regmean_equal",
        "regmean_sample_weighted",
        "regmean_hidden_layers",
        "regmean_hidden_layers_last_layer_tune",
        "regmean_hidden_layers_hidden_head_tune",
        "centralized",
    ]
    kl = diagnostic_mean(function_group, "symmetric_kl_to_centralized", methods)
    agree = diagnostic_mean(function_group, "prediction_agreement_to_centralized", methods)
    write_bar_svg(
        results_dir / "function_space_symmetric_kl.svg",
        "Function-space symmetric KL to centralized",
        methods,
        [kl[method] for method in methods],
        "Symmetric KL",
    )
    write_bar_svg(
        results_dir / "function_space_agreement.svg",
        "Prediction agreement to centralized",
        methods,
        [agree[method] for method in methods],
        "Agreement",
    )


def plot_representation_space(results_dir: Path, diagnostics: Dict[str, object]) -> None:
    repr_group = diagnostics.get("representation_space", {})
    methods = [
        "simple_avg",
        "sample_weighted_avg",
        "regmean_equal",
        "regmean_sample_weighted",
        "regmean_hidden_layers",
        "regmean_hidden_layers_last_layer_tune",
        "regmean_hidden_layers_hidden_head_tune",
        "centralized",
    ]
    series = []
    for metric, label, color in [
        ("layer0_cosine_to_centralized", "layer0", "#2563eb"),
        ("layer1_cosine_to_centralized", "layer1", "#059669"),
        ("layer2_cosine_to_centralized", "layer2", "#dc2626"),
        ("penultimate_cosine_to_centralized", "penultimate", "#d97706"),
    ]:
        values = diagnostic_mean(repr_group, metric, methods)
        series.append({"label": label, "color": color, "values": [values[method] for method in methods]})
    write_line_svg(
        results_dir / "representation_space_cosine.svg",
        "Representation cosine to centralized",
        methods,
        series,
        "Cosine",
    )
    deep_drop = diagnostic_mean(repr_group, "deep_representation_drop", methods)
    write_bar_svg(
        results_dir / "representation_space_deep_drop.svg",
        "Deeper-layer mismatch proxy (layer0 cosine minus penultimate cosine)",
        methods,
        [deep_drop[method] for method in methods],
        "Deep-drop",
    )


def plot_systematic_calibration(results_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    group_rows = rows_for_group(rows, "systematic_calibration")
    if not group_rows:
        return
    settings = sorted({str(row["setting"]) for row in group_rows}, key=lambda x: int(x.split("shot", 1)[0]))
    lookup = {
        (str(row["setting"]), str(row["method"])): float(row["accuracy_mean"])
        for row in group_rows
        if row.get("accuracy_mean", "") != ""
    }
    regmean_series = [
        {"label": "merge", "color": COLORS["regmean_hidden_layers"], "values": [lookup.get((setting, "regmean_hidden_layers"), 0.0) for setting in settings]},
        {"label": "bias", "color": COLORS["regmean_hidden_layers_bias_cal"], "values": [lookup.get((setting, "regmean_hidden_layers_bias_cal"), 0.0) for setting in settings]},
        {"label": "temp", "color": COLORS["regmean_hidden_layers_temperature"], "values": [lookup.get((setting, "regmean_hidden_layers_temperature"), 0.0) for setting in settings]},
        {"label": "vector", "color": COLORS["regmean_hidden_layers_vector_scaling"], "values": [lookup.get((setting, "regmean_hidden_layers_vector_scaling"), 0.0) for setting in settings]},
        {"label": "last-layer", "color": COLORS["regmean_hidden_layers_last_layer_tune"], "values": [lookup.get((setting, "regmean_hidden_layers_last_layer_tune"), 0.0) for setting in settings]},
        {"label": "hidden+head", "color": COLORS["regmean_hidden_layers_hidden_head_tune"], "values": [lookup.get((setting, "regmean_hidden_layers_hidden_head_tune"), 0.0) for setting in settings]},
    ]
    sample_series = [
        {"label": "merge", "color": COLORS["sample_weighted_avg"], "values": [lookup.get((setting, "sample_weighted_avg"), 0.0) for setting in settings]},
        {"label": "bias", "color": COLORS["sample_weighted_bias_cal"], "values": [lookup.get((setting, "sample_weighted_bias_cal"), 0.0) for setting in settings]},
        {"label": "temp", "color": COLORS["sample_weighted_temperature"], "values": [lookup.get((setting, "sample_weighted_temperature"), 0.0) for setting in settings]},
        {"label": "vector", "color": COLORS["sample_weighted_vector_scaling"], "values": [lookup.get((setting, "sample_weighted_vector_scaling"), 0.0) for setting in settings]},
        {"label": "last-layer", "color": COLORS["sample_weighted_last_layer_tune"], "values": [lookup.get((setting, "sample_weighted_last_layer_tune"), 0.0) for setting in settings]},
        {"label": "hidden+head", "color": COLORS["sample_weighted_hidden_head_tune"], "values": [lookup.get((setting, "sample_weighted_hidden_head_tune"), 0.0) for setting in settings]},
    ]
    write_line_svg(
        results_dir / "systematic_calibration_regmean_hidden_layers.svg",
        "RegMean hidden-layers calibration ladder",
        settings,
        regmean_series,
        "Accuracy",
    )
    write_line_svg(
        results_dir / "systematic_calibration_sample_weighted.svg",
        "Sample-weighted merge calibration ladder",
        settings,
        sample_series,
        "Accuracy",
    )


def plot_calibration_strength(results_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    group_rows = rows_for_group(rows, "calibration_strength")
    if not group_rows:
        return
    settings = sorted_shot_settings(rows, "calibration_strength")
    lookup = {
        (str(row["setting"]), str(row["method"])): float(row["accuracy_mean"])
        for row in group_rows
        if row.get("accuracy_mean", "") != ""
    }
    regmean_series = [
        {"label": "merge", "color": COLORS["regmean_hidden_layers"], "values": [lookup.get((setting, "regmean_hidden_layers"), 0.0) for setting in settings]},
        {"label": "temp", "color": COLORS["regmean_hidden_layers_temperature"], "values": [lookup.get((setting, "regmean_hidden_layers_temperature"), 0.0) for setting in settings]},
        {"label": "bias", "color": COLORS["regmean_hidden_layers_bias_cal"], "values": [lookup.get((setting, "regmean_hidden_layers_bias_cal"), 0.0) for setting in settings]},
        {"label": "vector", "color": COLORS["regmean_hidden_layers_vector_scaling"], "values": [lookup.get((setting, "regmean_hidden_layers_vector_scaling"), 0.0) for setting in settings]},
        {"label": "last-layer", "color": COLORS["regmean_hidden_layers_last_layer_tune"], "values": [lookup.get((setting, "regmean_hidden_layers_last_layer_tune"), 0.0) for setting in settings]},
        {"label": "hidden+head", "color": COLORS["regmean_hidden_layers_hidden_head_tune"], "values": [lookup.get((setting, "regmean_hidden_layers_hidden_head_tune"), 0.0) for setting in settings]},
    ]
    sample_series = [
        {"label": "merge", "color": COLORS["sample_weighted_avg"], "values": [lookup.get((setting, "sample_weighted_avg"), 0.0) for setting in settings]},
        {"label": "temp", "color": COLORS["sample_weighted_temperature"], "values": [lookup.get((setting, "sample_weighted_temperature"), 0.0) for setting in settings]},
        {"label": "bias", "color": COLORS["sample_weighted_bias_cal"], "values": [lookup.get((setting, "sample_weighted_bias_cal"), 0.0) for setting in settings]},
        {"label": "vector", "color": COLORS["sample_weighted_vector_scaling"], "values": [lookup.get((setting, "sample_weighted_vector_scaling"), 0.0) for setting in settings]},
        {"label": "last-layer", "color": COLORS["sample_weighted_last_layer_tune"], "values": [lookup.get((setting, "sample_weighted_last_layer_tune"), 0.0) for setting in settings]},
        {"label": "hidden+head", "color": COLORS["sample_weighted_hidden_head_tune"], "values": [lookup.get((setting, "sample_weighted_hidden_head_tune"), 0.0) for setting in settings]},
    ]
    write_line_svg(results_dir / "calibration_strength_regmean_hidden_layers.svg", "Calibration strength on RegMean hidden-layers", settings, regmean_series, "Accuracy")
    write_line_svg(results_dir / "calibration_strength_sample_weighted.svg", "Calibration strength on sample-weighted average", settings, sample_series, "Accuracy")


def plot_layerwise_alignment(results_dir: Path, diagnostics: Dict[str, object]) -> None:
    group = diagnostics.get("layerwise_alignment", {})
    if not group:
        return
    methods = [
        "simple_avg",
        "sample_weighted_avg",
        "sample_weighted_avg_last_layer_tune",
        "sample_weighted_avg_hidden_head_tune",
        "regmean_hidden_layers",
        "regmean_hidden_layers_last_layer_tune",
        "regmean_hidden_layers_hidden_head_tune",
        "centralized",
    ]
    cka_series = []
    for metric, label, color in [
        ("layer0_cka_to_centralized", "layer0_cka", "#2563eb"),
        ("layer1_cka_to_centralized", "layer1_cka", "#059669"),
        ("layer2_cka_to_centralized", "layer2_cka", "#dc2626"),
        ("penultimate_cka_to_centralized", "penultimate_cka", "#d97706"),
    ]:
        values = diagnostic_mean(group, metric, methods)
        cka_series.append({"label": label, "color": color, "values": [values[method] for method in methods]})
    write_line_svg(results_dir / "layerwise_alignment_cka.svg", "Layerwise CKA to centralized", methods, cka_series, "CKA")

    center_series = []
    for metric, label, color in [
        ("layer0_center_cosine_to_centralized", "layer0_center", "#2563eb"),
        ("layer1_center_cosine_to_centralized", "layer1_center", "#059669"),
        ("layer2_center_cosine_to_centralized", "layer2_center", "#dc2626"),
        ("penultimate_center_cosine_to_centralized", "penultimate_center", "#d97706"),
    ]:
        values = diagnostic_mean(group, metric, methods)
        center_series.append({"label": label, "color": color, "values": [values[method] for method in methods]})
    write_line_svg(results_dir / "layerwise_alignment_center_cosine.svg", "Layerwise class-center cosine to centralized", methods, center_series, "Center cosine")


def plot_layerwise_intervention(results_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    group_rows = rows_for_group(rows, "layerwise_intervention")
    if not group_rows:
        return
    merge_lookup = mean_lookup([row for row in group_rows if str(row["setting"]) == "merge_layer_selection"])
    merge_methods = [
        "simple_avg",
        "sample_weighted_avg",
        "regmean_hidden_layers",
        "regmean_last_layer",
        "regmean_middle_last",
        "regmean_all_layers",
        "centralized",
    ]
    write_bar_svg(
        results_dir / "layerwise_intervention_merge_selection.svg",
        "Selective RegMean merge by layer set",
        merge_methods,
        [merge_lookup.get(method, 0.0) for method in merge_methods],
        "Accuracy",
    )
    tune_lookup = mean_lookup([row for row in group_rows if str(row["setting"]) == "tune_layer_selection"])
    tune_methods = [
        "sample_weighted_avg",
        "sample_weighted_last_layer_tune",
        "sample_weighted_middle_layer_tune",
        "sample_weighted_hidden_head_tune",
        "sample_weighted_full_tune",
        "regmean_hidden_layers",
        "regmean_hidden_last_layer_tune",
        "regmean_hidden_middle_layer_tune",
        "regmean_hidden_hidden_head_tune",
        "regmean_hidden_full_tune",
        "centralized",
    ]
    write_bar_svg(
        results_dir / "layerwise_intervention_tune_selection.svg",
        "Post-merge tuning by trainable layer set",
        tune_methods,
        [tune_lookup.get(method, 0.0) for method in tune_methods],
        "Accuracy",
    )


def plot_function_repr_bridge(results_dir: Path, diagnostics: Dict[str, object]) -> None:
    group = diagnostics.get("function_repr_bridge", {})
    if not group:
        return
    methods = [
        "sample_weighted_avg",
        "sample_weighted_avg_last_layer_tune",
        "sample_weighted_avg_hidden_head_tune",
        "regmean_hidden_layers",
        "regmean_hidden_layers_last_layer_tune",
        "regmean_hidden_layers_hidden_head_tune",
        "centralized",
    ]
    bridge_series = []
    for metric, label, color in [
        ("subset_accuracy", "subset_acc", "#2563eb"),
        ("prediction_agreement_to_centralized", "agreement", "#059669"),
        ("penultimate_cka_to_centralized", "penultimate_cka", "#d97706"),
    ]:
        values = diagnostic_mean(group, metric, methods)
        bridge_series.append({"label": label, "color": color, "values": [values[method] for method in methods]})
    write_line_svg(
        results_dir / "function_repr_bridge_metrics.svg",
        "Function-space and representation-space move together",
        methods,
        bridge_series,
        "Metric",
    )
    kl = diagnostic_mean(group, "symmetric_kl_to_centralized", methods)
    write_bar_svg(
        results_dir / "function_repr_bridge_kl.svg",
        "Function-space KL after layer-aware adaptation",
        methods,
        [kl[method] for method in methods],
        "Symmetric KL",
    )


def main() -> None:
    args = parse_args()
    aggregate_rows_data = save_aggregate_if_needed(args.results_dir)
    diagnostics = read_diagnostics(args.results_dir)

    build_markdown(args.results_dir, aggregate_rows_data, diagnostics)
    plot_gap_sources(args.results_dir, aggregate_rows_data)
    plot_gap_reduction(args.results_dir, aggregate_rows_data)
    plot_function_space(args.results_dir, diagnostics)
    plot_representation_space(args.results_dir, diagnostics)
    plot_systematic_calibration(args.results_dir, aggregate_rows_data)
    plot_calibration_strength(args.results_dir, aggregate_rows_data)
    plot_layerwise_alignment(args.results_dir, diagnostics)
    plot_layerwise_intervention(args.results_dir, aggregate_rows_data)
    plot_function_repr_bridge(args.results_dir, diagnostics)

    print(
        json.dumps(
            {
                "results_dir": str(args.results_dir),
                "files": [
                    str(args.results_dir / "diagnostics_report.md"),
                    str(args.results_dir / "gap_sources_accuracy.svg"),
                    str(args.results_dir / "gap_reduction_accuracy.svg"),
                    str(args.results_dir / "function_space_symmetric_kl.svg"),
                    str(args.results_dir / "function_space_agreement.svg"),
                    str(args.results_dir / "representation_space_cosine.svg"),
                    str(args.results_dir / "representation_space_deep_drop.svg"),
                    str(args.results_dir / "systematic_calibration_regmean_hidden_layers.svg"),
                    str(args.results_dir / "systematic_calibration_sample_weighted.svg"),
                    str(args.results_dir / "calibration_strength_regmean_hidden_layers.svg"),
                    str(args.results_dir / "calibration_strength_sample_weighted.svg"),
                    str(args.results_dir / "layerwise_alignment_cka.svg"),
                    str(args.results_dir / "layerwise_alignment_center_cosine.svg"),
                    str(args.results_dir / "layerwise_intervention_merge_selection.svg"),
                    str(args.results_dir / "layerwise_intervention_tune_selection.svg"),
                    str(args.results_dir / "function_repr_bridge_metrics.svg"),
                    str(args.results_dir / "function_repr_bridge_kl.svg"),
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
