#!/usr/bin/env python3
"""
CSV, Markdown, and SVG helpers for the second-pass experiment suite.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Dict, List, Sequence


METHOD_LABELS = {
    "simple_avg": "Simple average",
    "sample_weighted_avg": "Sample weighted average",
    "epoch_weighted_avg": "Epoch weighted",
    "val_loss_weighted_avg": "Val-loss weighted average",
    "regmean_equal": "RegMean equal",
    "regmean_epoch_weighted": "RegMean epoch weighted",
    "regmean_sample_weighted": "RegMean sample weighted",
    "regmean_val_loss_weighted": "RegMean val-loss weighted",
    "regmean_hidden_layers": "RegMean hidden layers",
    "simple_avg_bias_cal": "Simple avg + bias-only",
    "simple_avg_ridge_head": "Simple avg + ridge head",
    "simple_avg_short_tune": "Simple avg + short tune",
    "sample_weighted_avg_bias_cal": "Sample weighted + bias-only",
    "sample_weighted_avg_ridge_head": "Sample weighted + ridge head",
    "sample_weighted_avg_short_tune": "Sample weighted + short tune",
    "regmean_equal_bias_cal": "RegMean equal + bias-only",
    "regmean_equal_ridge_head": "RegMean equal + ridge head",
    "regmean_equal_short_tune": "RegMean equal + short tune",
    "regmean_sample_weighted_ridge_head": "RegMean sample weighted + ridge head",
    "regmean_hidden_layers_bias_cal": "RegMean hidden + bias-only",
    "regmean_hidden_layers_short_tune": "RegMean hidden + short tune",
    "regmean_hidden_layers_temperature": "RegMean hidden + temperature",
    "regmean_hidden_layers_vector_scaling": "RegMean hidden + vector scaling",
    "regmean_hidden_layers_last_layer_tune": "RegMean hidden + last-layer tune",
    "regmean_hidden_layers_hidden_head_tune": "RegMean hidden + hidden+head tune",
    "regmean_last_layer": "RegMean last layer",
    "regmean_middle_last": "RegMean middle+last",
    "regmean_all_layers": "RegMean all layers",
    "sample_weighted_bias_cal": "Sample weighted + bias-only",
    "sample_weighted_temperature": "Sample weighted + temperature",
    "sample_weighted_vector_scaling": "Sample weighted + vector scaling",
    "sample_weighted_last_layer_tune": "Sample weighted + last-layer tune",
    "sample_weighted_hidden_head_tune": "Sample weighted + hidden+head tune",
    "sample_weighted_middle_layer_tune": "Sample weighted + middle-layer tune",
    "sample_weighted_full_tune": "Sample weighted + full tune",
    "regmean_hidden_last_layer_tune": "RegMean hidden + last-layer tune",
    "regmean_hidden_middle_layer_tune": "RegMean hidden + middle-layer tune",
    "regmean_hidden_hidden_head_tune": "RegMean hidden + hidden+head tune",
    "regmean_hidden_full_tune": "RegMean hidden + full tune",
    "centralized": "Centralized",
    "single_best": "Best single",
}


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def flatten_summary_row(group: str, setting: str, seed: int, method: str, metrics: Dict[str, object]) -> Dict[str, object]:
    if method == "single_best":
        global_metrics = metrics["metrics"]
        local_mean = ""
        local_std = ""
    else:
        global_metrics = metrics["global"]
        local_values = list(metrics["per_client_local_accuracy"].values())
        local_mean = statistics.mean(local_values)
        local_std = statistics.pstdev(local_values) if len(local_values) > 1 else 0.0
    return {
        "group": group,
        "setting": setting,
        "seed": seed,
        "method": method,
        "accuracy": global_metrics["accuracy"],
        "loss": global_metrics["loss"],
        "macro_accuracy": global_metrics["macro_accuracy"],
        "worst_class_accuracy": global_metrics["worst_class_accuracy"],
        "mean_client_local_accuracy": local_mean,
        "std_client_local_accuracy": local_std,
    }


def aggregate_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    numeric_fields = [
        "accuracy",
        "loss",
        "macro_accuracy",
        "worst_class_accuracy",
        "mean_client_local_accuracy",
        "std_client_local_accuracy",
    ]
    for row in rows:
        grouped.setdefault((row["group"], row["setting"], row["method"]), []).append(row)

    aggregates: List[Dict[str, object]] = []
    for (group, setting, method), bucket in grouped.items():
        item: Dict[str, object] = {"group": group, "setting": setting, "method": method, "num_runs": len(bucket)}
        for field in numeric_fields:
            values = [float(row[field]) for row in bucket if row[field] != ""]
            item[f"{field}_mean"] = statistics.mean(values) if values else ""
            item[f"{field}_std"] = statistics.pstdev(values) if len(values) > 1 else (0.0 if values else "")
        aggregates.append(item)
    return sorted(aggregates, key=lambda row: (str(row["group"]), str(row["setting"]), str(row["method"])))


def _fmt(value: object) -> str:
    if value == "":
        return "-"
    return f"{float(value):.4f}"


def build_report_markdown(path: Path, summary_rows: Sequence[Dict[str, object]], notes: Dict[str, str]) -> None:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in summary_rows:
        grouped.setdefault(str(row["group"]), []).append(row)

    lines = ["# Second-pass Merge Experiment Report", ""]
    for group, rows in grouped.items():
        lines.append(f"## {group}")
        if group in notes:
            lines.append("")
            lines.append(notes[group])
        lines.append("")
        settings = sorted({str(row["setting"]) for row in rows})
        for setting in settings:
            lines.append(f"### {setting}")
            lines.append("")
            lines.append("| Method | Acc mean | Macro mean | Worst-class mean | Local mean |")
            lines.append("|---|---:|---:|---:|---:|")
            setting_rows = [row for row in rows if row["setting"] == setting]
            for row in setting_rows:
                lines.append(
                    f"| {METHOD_LABELS.get(str(row['method']), str(row['method']))} | "
                    f"{_fmt(row['accuracy_mean'])} | "
                    f"{_fmt(row['macro_accuracy_mean'])} | "
                    f"{_fmt(row['worst_class_accuracy_mean'])} | "
                    f"{_fmt(row['mean_client_local_accuracy_mean'])} |"
                )
            base = next((row for row in setting_rows if row["method"] == "simple_avg"), None)
            regmean = next((row for row in setting_rows if row["method"] == "regmean_epoch_weighted"), None)
            if base and regmean and base["accuracy_mean"] != "" and regmean["accuracy_mean"] != "":
                lines.append("")
                lines.append(
                    "RegMean epoch weighted minus simple average accuracy: "
                    f"`{float(regmean['accuracy_mean']) - float(base['accuracy_mean']):+.4f}`"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _palette(i: int) -> str:
    colors = ["#2563eb", "#059669", "#dc2626", "#d97706", "#7c3aed", "#0891b2"]
    return colors[i % len(colors)]


def write_line_svg(path: Path, title: str, x_labels: Sequence[str], series: Sequence[Dict[str, object]], y_label: str) -> None:
    width, height = 1100, 560
    ml, mr, mt, mb = 90, 40, 60, 110
    pw, ph = width - ml - mr, height - mt - mb
    all_values = [float(v) for item in series for v in item["values"]]
    y_min = min(all_values) if all_values else 0.0
    y_max = max(all_values) if all_values else 1.0
    y_min = min(y_min, 0.0)
    y_max = max(y_max, 1.0 if y_max <= 1.0 else y_max)
    if abs(y_max - y_min) < 1e-8:
        y_max = y_min + 1.0

    def y_to_px(v: float) -> float:
        return mt + ph - ((v - y_min) / (y_max - y_min)) * ph

    def x_to_px(i: int) -> float:
        if len(x_labels) == 1:
            return ml + pw / 2
        return ml + i * pw / (len(x_labels) - 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }',
        ".axis { stroke: #6b7280; stroke-width: 1; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        ".label { font-size: 12px; }",
        ".title { font-size: 20px; font-weight: 600; }",
        "</style>",
        f'<text x="{ml}" y="30" class="title">{title}</text>',
    ]
    for tick in range(6):
        value = y_min + tick * (y_max - y_min) / 5
        y = y_to_px(value)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width - mr}" y2="{y:.1f}" class="grid" />')
        parts.append(f'<text x="{ml - 10}" y="{y + 4:.1f}" text-anchor="end" class="label">{value:.3f}</text>')
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" class="axis" />')
    parts.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{width - mr}" y2="{mt + ph}" class="axis" />')

    for idx, label in enumerate(x_labels):
        x = x_to_px(idx)
        parts.append(
            f'<text x="{x:.1f}" y="{height - 45}" text-anchor="end" transform="rotate(-25 {x:.1f},{height - 45})" class="label">{label}</text>'
        )

    for series_idx, item in enumerate(series):
        points = " ".join(f"{x_to_px(i):.1f},{y_to_px(float(v)):.1f}" for i, v in enumerate(item["values"]))
        color = item["color"]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}" />')
        for i, v in enumerate(item["values"]):
            x = x_to_px(i)
            y = y_to_px(float(v))
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" />')
            parts.append(f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" class="label">{float(v):.3f}</text>')
        lx = ml + series_idx * 160
        ly = height - 18
        parts.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 18}" y2="{ly - 4}" stroke="{color}" stroke-width="3" />')
        parts.append(f'<text x="{lx + 24}" y="{ly}" class="label">{item["label"]}</text>')
    parts.append(f'<text x="20" y="{mt + ph / 2:.1f}" transform="rotate(-90 20,{mt + ph / 2:.1f})" class="label">{y_label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_bar_svg(path: Path, title: str, x_labels: Sequence[str], values: Sequence[float], y_label: str) -> None:
    width, height = 980, 520
    ml, mr, mt, mb = 90, 30, 60, 110
    pw, ph = width - ml - mr, height - mt - mb
    max_abs = max([abs(v) for v in values] + [0.01])
    y_max = math.ceil(max_abs * 1000) / 1000

    def y_to_px(v: float) -> float:
        return mt + ph - ((v + y_max) / (2 * y_max)) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }',
        ".axis { stroke: #6b7280; stroke-width: 1; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        ".label { font-size: 12px; }",
        ".title { font-size: 20px; font-weight: 600; }",
        "</style>",
        f'<text x="{ml}" y="30" class="title">{title}</text>',
    ]
    for tick in range(5):
        value = -y_max + tick * (2 * y_max / 4)
        y = y_to_px(value)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width - mr}" y2="{y:.1f}" class="grid" />')
        parts.append(f'<text x="{ml - 10}" y="{y + 4:.1f}" text-anchor="end" class="label">{value:+.3f}</text>')
    zero_y = y_to_px(0.0)
    parts.append(f'<line x1="{ml}" y1="{zero_y:.1f}" x2="{width - mr}" y2="{zero_y:.1f}" class="axis" />')
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" class="axis" />')

    group_w = pw / max(1, len(x_labels))
    bar_w = group_w * 0.52
    for idx, (label, value) in enumerate(zip(x_labels, values)):
        x = ml + idx * group_w + (group_w - bar_w) / 2
        h = (abs(value) / (2 * y_max)) * ph
        y = zero_y - h if value >= 0 else zero_y
        color = "#059669" if value >= 0 else "#dc2626"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="2" />')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{(y - 6 if value >= 0 else y + h + 16):.1f}" text-anchor="middle" class="label">{value:+.4f}</text>')
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{height - 45}" text-anchor="end" transform="rotate(-25 {x + bar_w / 2:.1f},{height - 45})" class="label">{label}</text>'
        )
    parts.append(f'<text x="20" y="{mt + ph / 2:.1f}" transform="rotate(-90 20,{mt + ph / 2:.1f})" class="label">{y_label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
