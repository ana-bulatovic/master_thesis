#!/usr/bin/env python3
"""
Compare fine-tuned sarcasm classifier vs LLM prompting on the SAME test samples.

Usage:
  python scripts/compare_sarcasm_approaches.py --limit 50
  python scripts/compare_sarcasm_approaches.py --llm-model llama3.1:latest --technique few-shot --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from path_setup import PROJECT_ROOT
from data.dataset_loader import DatasetLoader
from evaluation.comparison_report import write_csv, write_html_report
from evaluation.evaluator import Evaluator
from models.sarcasm_classifier import FinetunedSarcasmClassifier
from pipeline.pipeline import NLPPipeline
from run_evaluation import sarcasm_prediction_to_label

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "sarcasm_distilbert" / "final"


def evaluate_finetuned(
    classifier: FinetunedSarcasmClassifier,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    predictions: List[str] = []
    references: List[str] = []
    samples: List[Dict[str, Any]] = []

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
    return {"metrics": metrics, "samples": samples}


def evaluate_llm(
    pipeline: NLPPipeline,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    predictions: List[str] = []
    references: List[str] = []
    samples: List[Dict[str, Any]] = []

    for index, record in enumerate(records, 1):
        text = record["text"]
        reference = record["label"]
        result = pipeline.sarcasm_detector.detect(
            text, model=pipeline.config.pipeline_config.sarcasm_model
        )
        predicted = sarcasm_prediction_to_label(result["is_sarcastic"])
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
                "model_response": result.get("response", ""),
            }
        )

    metrics = Evaluator.evaluate_classification(predictions, references)
    return {"metrics": metrics, "samples": samples}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fine-tuned vs LLM sarcasm detection")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--llm-model", default=None, help="Override LLM model from config.yaml")
    parser.add_argument(
        "--technique",
        choices=["zero-shot", "few-shot", "chain-of-thought"],
        default=None,
    )
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)

    if not model_dir.exists():
        print(f"Fine-tuned model not found: {model_dir}")
        print("Train first: python scripts/train_sarcasm.py")
        return 1

    config_path = PROJECT_ROOT / "config" / "config.yaml"
    pipeline = NLPPipeline(config_path=str(config_path))

    if args.llm_model:
        pipeline.config.model_config.name = args.llm_model
        pipeline.config.pipeline_config.sarcasm_model = args.llm_model
    if args.technique:
        pipeline.config.pipeline_config.prompting_technique = args.technique
    pipeline._initialize_components()

    if not pipeline.check_setup():
        print("Ollama is not available. Start with: ollama serve")
        return 1

    loader = DatasetLoader()
    records = loader.load_task_split("sarcasm", split=args.split)[: args.limit]

    print(f"Comparing approaches on {len(records)} identical samples...")

    classifier = FinetunedSarcasmClassifier(model_dir)
    finetuned_result = evaluate_finetuned(classifier, records)
    llm_result = evaluate_llm(pipeline, records)

    llm_model = pipeline.config.pipeline_config.sarcasm_model
    technique = pipeline.config.pipeline_config.prompting_technique
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = [
        {
            "run_id": run_id,
            "model": str(model_dir.relative_to(PROJECT_ROOT)),
            "technique": "fine-tuned",
            "task": "sarcasm",
            "source": None,
            "variant": "-",
            "split": args.split,
            "limit": args.limit,
            **finetuned_result["metrics"],
            "rouge1_f": None,
            "rouge2_f": None,
            "rougeL_f": None,
        },
        {
            "run_id": run_id,
            "model": llm_model,
            "technique": technique,
            "task": "sarcasm",
            "source": None,
            "variant": "prompting",
            "split": args.split,
            "limit": args.limit,
            **llm_result["metrics"],
            "rouge1_f": None,
            "rouge2_f": None,
            "rougeL_f": None,
        },
    ]

    output_json = RESULTS_DIR / f"sarcasm_comparison_{run_id}.json"
    output_csv = RESULTS_DIR / f"sarcasm_comparison_{run_id}.csv"
    output_html = RESULTS_DIR / f"sarcasm_comparison_{run_id}.html"

    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "type": "sarcasm_approach_comparison",
        "split": args.split,
        "limit": args.limit,
        "fine_tuned": {
            "model_dir": str(model_dir.relative_to(PROJECT_ROOT)),
            "metrics": finetuned_result["metrics"],
            "samples": finetuned_result["samples"],
        },
        "llm_prompting": {
            "model": llm_model,
            "technique": technique,
            "metrics": llm_result["metrics"],
            "samples": llm_result["samples"],
        },
        "comparison_rows": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(rows, output_csv)
    write_html_report(
        rows,
        output_html,
        title="Uporedna analiza: fine-tuned vs LLM prompting (sarkazam)",
    )

    print("\nFine-tuned:")
    print(f"  F1: {finetuned_result['metrics']['f1']:.4f}")
    print(f"  Accuracy: {finetuned_result['metrics']['accuracy']:.4f}")

    print("\nLLM prompting:")
    print(f"  F1: {llm_result['metrics']['f1']:.4f}")
    print(f"  Accuracy: {llm_result['metrics']['accuracy']:.4f}")

    print(f"\nJSON:  {output_json.relative_to(PROJECT_ROOT)}")
    print(f"CSV:   {output_csv.relative_to(PROJECT_ROOT)}")
    print(f"HTML:  {output_html.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
