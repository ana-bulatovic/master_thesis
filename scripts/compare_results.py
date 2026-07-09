#!/usr/bin/env python3
"""
Build comparison table and HTML report from existing evaluation JSON files.

Examples:
  python scripts/compare_results.py
  python scripts/compare_results.py --files results/evaluation_20260708_033040.json
  python scripts/compare_results.py --pattern "evaluation_*.json" --output results/my_comparison
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

from path_setup import PROJECT_ROOT
from evaluation.comparison_report import (
    load_rows_from_directory,
    load_rows_from_json,
    write_csv,
    write_html_report,
)

RESULTS_DIR = PROJECT_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation JSON files into comparison table/charts"
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Specific evaluation JSON files to include",
    )
    parser.add_argument(
        "--pattern",
        default="evaluation_*.json",
        help="Glob pattern when scanning results/ directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output base path without extension (default: results/aggregated_YYYYMMDD_HHMMSS)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: List[dict] = []

    if args.files:
        for file_path in args.files:
            path = Path(file_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            rows.extend(load_rows_from_json(path))
    else:
        rows = load_rows_from_directory(RESULTS_DIR, pattern=args.pattern)

    if not rows:
        print("No evaluation results found. Run run_evaluation.py or run_comparison.py first.")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        base = Path(args.output)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
    else:
        base = RESULTS_DIR / f"aggregated_{timestamp}"

    csv_path = base.with_suffix(".csv")
    html_path = base.with_suffix(".html")
    json_path = base.with_suffix(".json")

    write_csv(rows, csv_path)
    write_html_report(rows, html_path, title="Agregirana uporedna analiza")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "num_rows": len(rows),
        "csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "html": str(html_path.relative_to(PROJECT_ROOT)),
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Rows:  {len(rows)}")
    print(f"CSV:   {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"HTML:  {html_path.relative_to(PROJECT_ROOT)}")
    print(f"JSON:  {json_path.relative_to(PROJECT_ROOT)}")
    print("\nOpen the HTML file in a browser for table and charts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
