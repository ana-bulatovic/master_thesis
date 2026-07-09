#!/usr/bin/env python3
"""
Evaluate a fine-tuned sarcasm model on the test split.

Usage:
  python scripts/evaluate_finetuned_sarcasm.py
  python scripts/evaluate_finetuned_sarcasm.py --model-dir models/sarcasm_distilbert/final --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from path_setup import PROJECT_ROOT
from data.dataset_loader import DatasetLoader
from evaluation.evaluator import Evaluator
from models.sarcasm_classifier import FinetunedSarcasmClassifier

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "sarcasm_distilbert" / "final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned sarcasm classifier")
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Path to saved fine-tuned model directory",
    )
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Model not found: {model_dir}")
        print("Train first: python scripts/train_sarcasm.py")
        return 1

    classifier = FinetunedSarcasmClassifier(model_dir)
    loader = DatasetLoader()
    records = loader.load_task_split("sarcasm", split=args.split)
    if args.limit:
        records = records[: args.limit]

    predictions: List[str] = []
    references: List[str] = []
    samples: List[Dict[str, Any]] = []

    print(f"Evaluating fine-tuned model on {len(records)} samples...")

    for index, record in enumerate(records, 1):
        text = record["text"]
        reference = record["label"]
        result = classifier.predict(text)
        predicted = result["label"]
        correct = predicted == reference

        predictions.append(predicted)
        references.append(reference)
        samples.append(
            {
                "index": index,
                "text": text,
                "reference": reference,
                "predicted": predicted,
                "correct": correct,
                "confidence": result["confidence"],
            }
        )

    metrics = Evaluator.evaluate_classification(predictions, references)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"evaluation_finetuned_sarcasm_{run_id}.json"

    metadata_path = model_dir / "training_metadata.json"
    base_model = "unknown"
    if metadata_path.exists():
        base_model = json.loads(metadata_path.read_text(encoding="utf-8")).get("base_model", "unknown")

    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "approach": "fine-tuned",
        "model": str(model_dir.relative_to(PROJECT_ROOT)),
        "base_model": base_model,
        "prompting_technique": "fine-tuned",
        "split": args.split,
        "limit": args.limit or len(records),
        "tasks": [
            {
                "task": "sarcasm",
                "metrics": metrics,
                "samples": samples,
            }
        ],
        "output_file": str(output_file.relative_to(PROJECT_ROOT)),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nResults:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"\nSaved to: {output_file.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
