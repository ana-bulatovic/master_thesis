#!/usr/bin/env python3
"""Prepare datasets for master thesis experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data.dataset_loader import DatasetLoader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, preprocess, and save thesis datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "sarcasm", "sentiment", "summarization-amazon", "summarization-xsum"],
        default="all",
        help="Which dataset to prepare",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "datasets.yaml"),
        help="Path to datasets.yaml",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    loader = DatasetLoader(config_path=args.config)

    if args.dataset == "all":
        results = loader.prepare_all()
    elif args.dataset == "sarcasm":
        results = {"sarcasm": loader.prepare_isarcasmeval()}
    elif args.dataset == "sentiment":
        results = {"sentiment": loader.prepare_amazon_sentiment()}
    elif args.dataset == "summarization-amazon":
        results = {"summarization_amazon": loader.prepare_amazon_summarization()}
    elif args.dataset == "summarization-xsum":
        results = {"summarization_xsum": loader.prepare_xsum()}
    else:
        raise ValueError(f"Unsupported dataset option: {args.dataset}")

    print("\nDataset preparation completed.")
    for name, metadata in results.items():
        print(f"\n{name}:")
        if isinstance(metadata, dict) and "splits" in metadata:
            for split_name, split_info in metadata["splits"].items():
                count = split_info.get("count") or split_info.get("stats", {}).get(
                    "output_count", "?"
                )
                print(f"  - {split_name}: {count} samples -> {split_info.get('path')}")
        else:
            print(f"  metadata saved")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
