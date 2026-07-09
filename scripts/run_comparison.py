#!/usr/bin/env python3
"""
Run comparative evaluation across multiple models, techniques, and pipeline variants
on the SAME dataset samples.

Examples:
  python scripts/run_comparison.py --task sentiment --techniques few-shot,chain-of-thought --limit 20
  python scripts/run_comparison.py --task summarization --source amazon --limit 10
  python scripts/run_comparison.py --task all --models llama3.2:3b --techniques few-shot,chain-of-thought --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from path_setup import PROJECT_ROOT

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from data.dataset_loader import DatasetLoader
from evaluation.comparison_report import (
    flatten_comparison_report,
    write_csv,
    write_html_report,
)
from pipeline.pipeline import NLPPipeline

VARIANT_LABELS = {
    "without_sentiment": "baseline",
    "with_sentiment": "+sentiment",
    "with_sarcasm": "+sarcasm",
    "with_sentiment_and_sarcasm": "+sentiment+sarcasm",
    "sentiment_baseline": "sentiment BEZ detekcije sarkazma",
    "sentiment_with_sarcasm_detection": "sentiment SA detekcijom sarkazma",
}


def friendly_variant(variant: str | None) -> str:
    if not variant:
        return "-"
    return VARIANT_LABELS.get(variant, variant)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


# Reuse evaluation logic from run_evaluation.py
from run_evaluation import (  # noqa: E402
    LOG_DIR,
    RESULTS_DIR,
    evaluate_sarcasm,
    evaluate_sentiment,
    evaluate_summarization,
    setup_logging,
)

SUMMARIZATION_VARIANTS: List[Tuple[bool, bool, str]] = [
    (False, False, "without_sentiment"),
    (True, False, "with_sentiment"),
    (False, True, "with_sarcasm"),
    (True, True, "with_sentiment_and_sarcasm"),
]


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def configure_pipeline(pipeline: NLPPipeline, model: str, technique: str) -> None:
    pipeline.config.model_config.name = model
    pipeline.config.pipeline_config.sarcasm_model = model
    pipeline.config.pipeline_config.sentiment_model = model
    pipeline.config.pipeline_config.summarization_model = model
    pipeline.config.pipeline_config.prompting_technique = technique
    pipeline._initialize_components()


def experiment_row_from_task_result(
    model: str,
    technique: str,
    task_result: Dict[str, Any],
) -> Dict[str, Any]:
    task = task_result["task"]
    row: Dict[str, Any] = {
        "model": model,
        "technique": technique,
        "task": task,
        "source": task_result.get("source"),
        "variant": friendly_variant(task_result.get("variant")),
    }

    if task in ("sarcasm", "sentiment"):
        metrics = task_result.get("metrics", {})
        row.update(
            {
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
            }
        )
    elif task == "summarization":
        rouge = task_result.get("metrics", {}).get("rouge", {})
        row.update(
            {
                "rouge1_f": rouge.get("rouge1_f"),
                "rouge2_f": rouge.get("rouge2_f"),
                "rougeL_f": rouge.get("rougeL_f"),
            }
        )

    row["details"] = task_result
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare models and pipeline variants on the same data"
    )
    parser.add_argument(
        "--task",
        choices=["sarcasm", "sentiment", "summarization", "all"],
        default="sentiment",
        help="Which task(s) to compare (sentiment: with vs without sarcasm detection)",
    )
    parser.add_argument(
        "--source",
        choices=["amazon", "xsum"],
        default="amazon",
        help="Summarization dataset source",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of samples (same for all experiments)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated models, e.g. tinyllama:latest,llama3.1,mistral",
    )
    parser.add_argument(
        "--techniques",
        default="few-shot",
        help="Comma-separated prompting techniques",
    )
    parser.add_argument(
        "--variants",
        choices=["all", "baseline", "sentiment", "sarcasm", "full"],
        default="all",
        help="Summarization pipeline variants to test",
    )
    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Skip HTML report generation",
    )
    return parser.parse_args()


def get_summarization_variants(mode: str) -> List[Tuple[bool, bool, str]]:
    mapping = {
        "all": SUMMARIZATION_VARIANTS,
        "baseline": [(False, False, "without_sentiment")],
        "sentiment": [(True, False, "with_sentiment")],
        "sarcasm": [(False, True, "with_sarcasm")],
        "full": [(True, True, "with_sentiment_and_sarcasm")],
    }
    return mapping[mode]


def resolve_models(pipeline: NLPPipeline, models_arg: str | None) -> List[str]:
    if models_arg:
        return parse_csv_list(models_arg)

    available = pipeline.client.list_models()
    if available:
        return available[:3]

    return [pipeline.config.model_config.name]


def main() -> int:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"comparison_{run_id}.log"
    output_json = RESULTS_DIR / f"comparison_{run_id}.json"
    output_csv = RESULTS_DIR / f"comparison_{run_id}.csv"
    output_html = RESULTS_DIR / f"comparison_{run_id}.html"

    logger = setup_logging(log_file)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config_path = PROJECT_ROOT / "config" / "config.yaml"
    pipeline = NLPPipeline(config_path=str(config_path))

    if not pipeline.check_setup():
        logger.error("Ollama is not available. Start with: ollama serve")
        return 1

    models = resolve_models(pipeline, args.models)
    techniques = parse_csv_list(args.techniques)
    tasks = (
        ["sarcasm", "sentiment", "summarization"]
        if args.task == "all"
        else [args.task]
    )

    loader = DatasetLoader()
    dataset_cache: Dict[str, List[Dict[str, Any]]] = {}
    experiments: List[Dict[str, Any]] = []
    total_runs = len(models) * len(techniques) * len(tasks)

    logger.info("Comparison run: %s", run_id)
    logger.info("Models: %s", models)
    logger.info("Techniques: %s", techniques)
    logger.info("Tasks: %s", tasks)
    logger.info("Same limit for all runs: %d", args.limit)

    run_counter = 0
    for model in models:
        for technique in techniques:
            configure_pipeline(pipeline, model, technique)
            logger.info("")
            logger.info("=== Model: %s | Technique: %s ===", model, technique)

            for task in tasks:
                run_counter += 1
                logger.info(
                    ">>> [%d] Task: %s (model=%s, technique=%s)",
                    run_counter,
                    task.upper(),
                    model,
                    technique,
                )

                if task not in dataset_cache:
                    if task == "summarization":
                        dataset_cache[task] = loader.load_task_split(
                            task, split=args.split, source=args.source
                        )
                    else:
                        dataset_cache[task] = loader.load_task_split(
                            task, split=args.split
                        )

                records = dataset_cache[task][: args.limit]

                if task == "sarcasm":
                    task_result = evaluate_sarcasm(pipeline, records, logger)
                    experiments.append(
                        experiment_row_from_task_result(model, technique, task_result)
                    )

                elif task == "sentiment":
                    for use_sarcasm_flag in (False, True):
                        variant_name = (
                            "sentiment_with_sarcasm_detection"
                            if use_sarcasm_flag
                            else "sentiment_baseline"
                        )
                        logger.info("    Variant: %s", friendly_variant(variant_name))
                        task_result = evaluate_sentiment(
                            pipeline, records, logger, use_sarcasm=use_sarcasm_flag
                        )
                        experiments.append(
                            experiment_row_from_task_result(model, technique, task_result)
                        )

                elif task == "summarization":
                    for use_sentiment, use_sarcasm, variant_name in get_summarization_variants(
                        args.variants
                    ):
                        logger.info("    Variant: %s", variant_name)
                        task_result = evaluate_summarization(
                            pipeline,
                            records,
                            source=args.source,
                            logger=logger,
                            use_sentiment=use_sentiment,
                            use_sarcasm=use_sarcasm,
                        )
                        experiments.append(
                            experiment_row_from_task_result(model, technique, task_result)
                        )

    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "type": "comparison",
        "dataset": {
            "task": args.task,
            "source": args.source if "summarization" in tasks else None,
            "split": args.split,
            "limit": args.limit,
            "note": "All experiments use the same first N samples from the split.",
        },
        "models": models,
        "techniques": techniques,
        "variants_mode": args.variants,
        "experiments": experiments,
        "output_file": str(output_json.relative_to(PROJECT_ROOT)),
        "log_file": str(log_file.relative_to(PROJECT_ROOT)),
    }

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    flat_rows = flatten_comparison_report(report)
    write_csv(flat_rows, output_csv)

    if not args.skip_html:
        write_html_report(
            flat_rows,
            output_html,
            title="Uporedna analiza pipeline-a",
        )

    sentiment_rows = [row for row in flat_rows if row.get("task") == "sentiment"]
    if sentiment_rows:
        logger.info("")
        logger.info("SENTIMENT: sa vs bez detekcije sarkazma")
        logger.info("-" * 70)
        for row in sentiment_rows:
            logger.info(
                "  %s | %s | %s | F1=%s | Acc=%s",
                row.get("technique"),
                row.get("variant"),
                row.get("model"),
                _format_metric(row.get("f1")),
                _format_metric(row.get("accuracy")),
            )

    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 70)
    logger.info("Experiments: %d", len(experiments))
    logger.info("JSON:  %s", output_json.relative_to(PROJECT_ROOT))
    logger.info("CSV:   %s", output_csv.relative_to(PROJECT_ROOT))
    if not args.skip_html:
        logger.info("HTML:  %s", output_html.relative_to(PROJECT_ROOT))
    logger.info("Log:   %s", log_file.relative_to(PROJECT_ROOT))
    logger.info("")
    logger.info("Open HTML in browser for table + charts.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
