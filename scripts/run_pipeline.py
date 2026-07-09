#!/usr/bin/env python3
"""
Run the full NLP pipeline:
  - Sarcasm: fine-tuned DistilBERT (or LLM if configured)
  - Sentiment + Summarization: Ollama LLM

Usage:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --text "Oh great, another amazing product that broke after two days."
  python scripts/run_pipeline.py --file reviews.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from path_setup import PROJECT_ROOT
from pipeline.pipeline import NLPPipeline

DEFAULT_SAMPLES = [
    "This smartphone has an excellent camera and long battery life. I am very happy with this purchase.",
    "Oh great, another amazing product that broke after two days. Best purchase ever.",
    "Terrible quality. The laptop keeps freezing. Complete waste of money.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full thesis NLP pipeline")
    parser.add_argument("--text", default=None, help="Single review text")
    parser.add_argument("--file", default=None, help="Text file with one review per line")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less console output during processing",
    )
    return parser.parse_args()


def load_reviews(args: argparse.Namespace) -> list[str]:
    if args.text:
        return [args.text.strip()]

    if args.file:
        path = Path(args.file)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        lines = path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]

    return DEFAULT_SAMPLES


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    reviews = load_reviews(args)

    print("=" * 80)
    print("MASTER RAD — FULL NLP PIPELINE")
    print("=" * 80)

    pipeline = NLPPipeline(config_path=str(config_path))
    pipeline.print_config_summary()

    if not pipeline.check_setup():
        print("\nSetup failed. Fix the issues above and try again.")
        return 1

    results = pipeline.process_batch(reviews, verbose=not args.quiet)
    pipeline.save_results(results, "pipeline_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
