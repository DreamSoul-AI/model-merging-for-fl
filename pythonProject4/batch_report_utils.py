#!/usr/bin/env python3
"""
Reporting helpers for MNIST merge lab batch experiments.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Dict, List, Sequence


METHOD_LABELS = {
    "centralized": "Centralized",
    "single_best": "Best single",
    "simple_avg": "Simple average",
    "sample_weighted_avg": "Sample weighted",
    "epoch_weighted_avg": "Epoch weighted",
    "regmean_equal": "RegMean equal",
    "regmean_epoch_weighted": "RegMean epoch weighted",
}


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    numeric_fields = [
        "accuracy",
        "loss",
        "macro_accuracy",
        "worst_class_accuracy",
        "mean_client_local_accuracy",
        "std_client_local_accuracy",
    ]
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        key = (row["experiment"], row["method"])
        grouped.setdefault(key, []).append(row)

    aggregated: List[Dict[str, object]] = []
    for (experiment_name, method_name), bucket in grouped.items():
        record: Dict[str, object] = {
            "experiment": experiment_name,
            "method": method_name,
            "num_runs": len(bucket),
        }
        for field in numeric_fields:
            values = [float(item[field]) for item in bucket if item[field] != ""]
            if values:
                record[f"{field}_mean"] = statistics.mean(values)
                record[f"{field}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            else:
                record[f"{field}_mean"] = ""
                record[f"{field}_std"] = ""
        aggregated.append(record)

    aggregated.sort(
        key=lambda item: (
            str(item["experiment"]),
            method_sort_key(str(item["method"])),
        )
    )
    return aggregated


def safe_slug(text: str) -> str:
    return text.replace("/", "-").replace(" ", "-")


def format_metric(value: object) -> str:
    if value == "":
        return "-"
    return f"{float(value):.4f}"


def method_sort_key(method_name: str) -> int:
    order = [
        "centralized",
        "single_best",
        "simple_avg",
        "sample_weighted_avg",
        "epoch_weighted_avg",
        "regmean_equal",
        "regmean_epoch_weighted",
    ]
    return order.index(method_name) if method_name in order else len(order)


def generate_summary_markdown(
    output_path: Path,
    selected_experiments: Sequence[str],
    aggregated_rows: Sequence[Dict[str, object]],
    experiment_descriptions: Dict[str, str] | None = None,
) -> None:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in aggregated_rows:
        grouped.setdefault(str(row["experiment"]), []).append(row)

    lines: List[str] = [
        "# MNIST Merge Lab Batch Report",
        "",
        "## Included presets",
        "",
    ]
    for name in selected_experiments:
        description = experiment_descriptions.get(name, "") if experiment_descriptions else ""
        if description:
            lines.append(f"- `{name}`: {description}")
        else:
            lines.append(f"- `{name}`")

    lines.extend(["", "## Aggregate comparison", ""])

    for experiment_name in selected_experiments:
        lines.append(f"### {experiment_name}")
        lines.append("")
        lines.append("| Method | Acc mean | Acc std | Macro mean | Worst-class mean | Local-acc mean |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        experiment_rows = sorted(grouped.get(experiment_name, []), key=lambda item: method_sort_key(str(item["method"])))
        for row in experiment_rows:
            lines.append(
                "| "
                f"{METHOD_LABELS.get(str(row['method']), str(row['method']))} | "
                f"{format_metric(row['accuracy_mean'])} | "
                f"{format_metric(row['accuracy_std'])} | "
                f"{format_metric(row['macro_accuracy_mean'])} | "
                f"{format_metric(row['worst_class_accuracy_mean'])} | "
                f"{format_metric(row['mean_client_local_accuracy_mean'])} |"
            )
        lines.append("")

        baseline = next((row for row in experiment_rows if row["method"] == "simple_avg"), None)
        regmean = next((row for row in experiment_rows if row["method"] == "regmean_epoch_weighted"), None)
        if baseline and regmean and baseline["accuracy_mean"] != "" and regmean["accuracy_mean"] != "":
            delta = float(regmean["accuracy_mean"]) - float(baseline["accuracy_mean"])
            lines.append(f"RegMean epoch weighted vs simple average accuracy delta: `{delta:+.4f}`")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _chart_palette(index: int) -> str:
    palette = ["#2563eb", "#059669", "#dc2626", "#d97706", "#7c3aed", "#0891b2", "#4b5563"]
    return palette[index % len(palette)]


def generate_grouped_bar_svg(
    output_path: Path,
    title: str,
    x_labels: Sequence[str],
    series: Sequence[Dict[str, object]],
    y_label: str,
) -> None:
    width = 1100
    height = 560
    margin_left = 90
    margin_right = 30
    margin_top = 60
    margin_bottom = 110
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    max_value = max([float(value) for item in series for value in item["values"]] + [1.0])
    y_max = max(1.0, math.ceil(max_value * 20) / 20)
    group_width = plot_width / max(1, len(x_labels))
    series_width = group_width * 0.78
    bar_width = series_width / max(1, len(series))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }',
        ".axis { stroke: #6b7280; stroke-width: 1; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        ".label { font-size: 12px; }",
        ".title { font-size: 20px; font-weight: 600; }",
        ".legend { font-size: 12px; }",
        "</style>",
        f'<text x="{margin_left}" y="30" class="title">{title}</text>',
    ]

    for tick_index in range(6):
        value = y_max * tick_index / 5
        y = margin_top + plot_height - (value / y_max) * plot_height
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="grid" />')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" class="label">{value:.2f}</text>')

    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" class="axis" />')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" class="axis" />')

    for group_index, label in enumerate(x_labels):
        group_left = margin_left + group_index * group_width + (group_width - series_width) / 2
        for series_index, item in enumerate(series):
            value = float(item["values"][group_index])
            bar_height = 0 if y_max == 0 else (value / y_max) * plot_height
            x = group_left + series_index * bar_width
            y = margin_top + plot_height - bar_height
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.92:.1f}" height="{bar_height:.1f}" fill="{item["color"]}" rx="2" />'
            )
            parts.append(
                f'<text x="{x + bar_width * 0.46:.1f}" y="{max(y - 6, margin_top + 12):.1f}" text-anchor="middle" class="label">{value:.3f}</text>'
            )
        text_x = group_left + series_width / 2
        parts.append(
            f'<text x="{text_x:.1f}" y="{height - margin_bottom + 30}" text-anchor="end" transform="rotate(-25 {text_x:.1f},{height - margin_bottom + 30})" class="label">{label}</text>'
        )

    legend_x = margin_left
    legend_y = height - 28
    for series_index, item in enumerate(series):
        item_x = legend_x + series_index * 150
        parts.append(f'<rect x="{item_x}" y="{legend_y - 10}" width="14" height="14" fill="{item["color"]}" rx="2" />')
        parts.append(f'<text x="{item_x + 20}" y="{legend_y + 1}" class="legend">{item["label"]}</text>')

    parts.append(f'<text x="20" y="{margin_top + plot_height / 2:.1f}" transform="rotate(-90 20,{margin_top + plot_height / 2:.1f})" class="label">{y_label}</text>')
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def generate_delta_bar_svg(
    output_path: Path,
    title: str,
    x_labels: Sequence[str],
    deltas: Sequence[float],
    y_label: str,
) -> None:
    width = 980
    height = 520
    margin_left = 90
    margin_right = 30
    margin_top = 60
    margin_bottom = 110
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_abs = max([abs(value) for value in deltas] + [0.01])
    y_max = math.ceil(max_abs * 1000) / 1000

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }',
        ".axis { stroke: #6b7280; stroke-width: 1; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        ".label { font-size: 12px; }",
        ".title { font-size: 20px; font-weight: 600; }",
        "</style>",
        f'<text x="{margin_left}" y="30" class="title">{title}</text>',
    ]

    for tick_index in range(5):
        value = (-y_max) + tick_index * (2 * y_max / 4)
        y = margin_top + plot_height - ((value + y_max) / (2 * y_max)) * plot_height
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="grid" />')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" class="label">{value:+.3f}</text>')

    zero_y = margin_top + plot_height / 2
    parts.append(f'<line x1="{margin_left}" y1="{zero_y:.1f}" x2="{width - margin_right}" y2="{zero_y:.1f}" class="axis" />')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" class="axis" />')

    group_width = plot_width / max(1, len(x_labels))
    bar_width = group_width * 0.5
    for index, (label, delta) in enumerate(zip(x_labels, deltas)):
        x = margin_left + index * group_width + (group_width - bar_width) / 2
        height_px = (abs(delta) / (2 * y_max)) * plot_height
        y = zero_y - height_px if delta >= 0 else zero_y
        color = "#059669" if delta >= 0 else "#dc2626"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height_px:.1f}" fill="{color}" rx="2" />')
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{(y - 6 if delta >= 0 else y + height_px + 16):.1f}" text-anchor="middle" class="label">{delta:+.4f}</text>')
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - margin_bottom + 32}" text-anchor="end" transform="rotate(-25 {x + bar_width / 2:.1f},{height - margin_bottom + 32})" class="label">{label}</text>'
        )

    parts.append(f'<text x="20" y="{margin_top + plot_height / 2:.1f}" transform="rotate(-90 20,{margin_top + plot_height / 2:.1f})" class="label">{y_label}</text>')
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def generate_plots(output_dir: Path, selected_experiments: Sequence[str], aggregated_rows: Sequence[Dict[str, object]]) -> None:
    grouped: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in aggregated_rows:
        grouped.setdefault(str(row["experiment"]), {})[str(row["method"])] = row

    experiment_labels = [safe_slug(name) for name in selected_experiments]
    accuracy_series_methods = ["simple_avg", "epoch_weighted_avg", "regmean_equal", "regmean_epoch_weighted", "centralized"]
    series = []
    for index, method_name in enumerate(accuracy_series_methods):
        values = []
        for experiment_name in selected_experiments:
            entry = grouped.get(experiment_name, {}).get(method_name)
            values.append(float(entry["accuracy_mean"]) if entry and entry["accuracy_mean"] != "" else 0.0)
        series.append(
            {
                "label": METHOD_LABELS.get(method_name, method_name),
                "values": values,
                "color": _chart_palette(index),
            }
        )
    generate_grouped_bar_svg(
        output_dir / "accuracy_comparison.svg",
        title="Test accuracy by experiment",
        x_labels=experiment_labels,
        series=series,
        y_label="Accuracy",
    )

    deltas = []
    for experiment_name in selected_experiments:
        baseline = grouped.get(experiment_name, {}).get("simple_avg")
        regmean = grouped.get(experiment_name, {}).get("regmean_epoch_weighted")
        if baseline and regmean and baseline["accuracy_mean"] != "" and regmean["accuracy_mean"] != "":
            deltas.append(float(regmean["accuracy_mean"]) - float(baseline["accuracy_mean"]))
        else:
            deltas.append(0.0)
    generate_delta_bar_svg(
        output_dir / "regmean_vs_simple_delta.svg",
        title="RegMean epoch weighted minus simple average",
        x_labels=experiment_labels,
        deltas=deltas,
        y_label="Accuracy delta",
    )
