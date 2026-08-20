#!/usr/bin/env python3
"""
Rebuild report tables and plots from a completed batch directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from batch_report_utils import (
    aggregate_rows,
    generate_plots,
    generate_summary_markdown,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate summaries and plots from batch outputs")
    parser.add_argument("--batch-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, object]] = []
        for row in reader:
            rows.append(dict(row))
        return rows


def main() -> None:
    args = parse_args()
    rows_path = args.batch_dir / "summary_rows.csv"
    if not rows_path.exists():
        raise FileNotFoundError(f"Missing summary rows file: {rows_path}")

    rows = read_rows(rows_path)
    aggregated_rows = aggregate_rows(rows)
    write_csv(args.batch_dir / "aggregate_summary.csv", aggregated_rows)
    (args.batch_dir / "aggregate_summary.json").write_text(
        json.dumps(aggregated_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    experiments = []
    for row in aggregated_rows:
        name = str(row["experiment"])
        if name not in experiments:
            experiments.append(name)

    generate_summary_markdown(args.batch_dir / "report.md", experiments, aggregated_rows)
    generate_plots(args.batch_dir, experiments, aggregated_rows)

    print(
        json.dumps(
            {
                "batch_dir": str(args.batch_dir),
                "files": [
                    str(args.batch_dir / "aggregate_summary.csv"),
                    str(args.batch_dir / "aggregate_summary.json"),
                    str(args.batch_dir / "report.md"),
                    str(args.batch_dir / "accuracy_comparison.svg"),
                    str(args.batch_dir / "regmean_vs_simple_delta.svg"),
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
