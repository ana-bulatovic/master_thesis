#!/usr/bin/env python3
"""
Run LLM evaluation on prepared datasets and save results with a readable log.

Usage examples:
  python scripts/run_evaluation.py --task sarcasm --limit 10
  python scripts/run_evaluation.py --task sentiment --limit 20
  python scripts/run_evaluation.py --task summarization --source amazon --limit 5 --with-sentiment --with-sarcasm
  python scripts/run_evaluation.py --task all --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from path_setup import PROJECT_ROOT
from config import Config
from data.dataset_loader import DatasetLoader
from evaluation.evaluator import Evaluator
from pipeline.pipeline import NLPPipeline

LOG_DIR = PROJECT_ROOT / "results" / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"


def setup_logging(log_file: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_evaluation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def truncate(text: str, max_len: int = 120) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def sarcasm_prediction_to_label(is_sarcastic: bool) -> str:
    return "sarcastic" if is_sarcastic else "non-sarcastic"


def evaluate_sarcasm(
    pipeline: NLPPipeline,
    records: List[Dict[str, Any]],
    logger: logging.Logger,
) -> Dict[str, Any]:
    predictions: List[str] = []
    references: List[str] = []
    samples: List[Dict[str, Any]] = []

    logger.info("Running sarcasm detection on %d samples...", len(records))

    for index, record in enumerate(records, 1):
        text = record["text"]
        reference = record["label"]

        result = pipeline.detect_sarcasm(text)
        if not result:
            raise RuntimeError(
                "Sarcasm detection is not configured. "
                "Set sarcasm_backend in config.yaml to 'fine-tuned' or 'llm'."
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

        status = "OK" if correct else "MISS"
        logger.info(
            "[%d/%d] %s | ref=%s pred=%s | %s",
            index,
            len(records),
            status,
            reference,
            predicted,
            truncate(text, 80),
        )

    metrics = Evaluator.evaluate_classification(predictions, references)
    return {"task": "sarcasm", "metrics": metrics, "samples": samples}


def evaluate_sentiment(
    pipeline: NLPPipeline,
    records: List[Dict[str, Any]],
    logger: logging.Logger,
    use_sarcasm: bool = False,
) -> Dict[str, Any]:
    predictions: List[str] = []
    references: List[str] = []
    samples: List[Dict[str, Any]] = []

    variant = "with_sarcasm" if use_sarcasm else "baseline"
    logger.info(
        "Running sentiment classification (%s) on %d samples...",
        variant,
        len(records),
    )

    for index, record in enumerate(records, 1):
        text = record["text"]
        reference = record["label"]
        sarcasm = None

        if use_sarcasm:
            sarcasm_result = pipeline.detect_sarcasm(text)
            if sarcasm_result:
                sarcasm = sarcasm_result["is_sarcastic"]

        result = pipeline.sentiment_classifier.classify(
            text,
            model=pipeline.config.pipeline_config.sentiment_model,
            sarcasm=sarcasm,
            sarcasm_aware=use_sarcasm and sarcasm is not None,
        )
        predicted = result["sentiment"]
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
                "sarcasm_used": sarcasm if use_sarcasm else None,
                "model_response": result.get("response", ""),
            }
        )

        status = "OK" if correct else "MISS"
        logger.info(
            "[%d/%d] %s | ref=%s pred=%s | %s",
            index,
            len(records),
            status,
            reference,
            predicted,
            truncate(text, 80),
        )

    metrics = Evaluator.evaluate_classification(
        predictions, references, labels=["positive", "negative", "neutral"]
    )
    return {"task": "sentiment", "variant": variant, "metrics": metrics, "samples": samples}


def evaluate_summarization(
    pipeline: NLPPipeline,
    records: List[Dict[str, Any]],
    source: str,
    logger: logging.Logger,
    use_sentiment: bool = False,
    use_sarcasm: bool = False,
) -> Dict[str, Any]:
    predictions: List[str] = []
    references: List[str] = []
    samples: List[Dict[str, Any]] = []

    if use_sentiment and use_sarcasm:
        variant = "with_sentiment_and_sarcasm"
        summarizer = pipeline.summarizer_with_sentiment_and_sarcasm
    elif use_sentiment:
        variant = "with_sentiment"
        summarizer = pipeline.summarizer_with_sentiment
    elif use_sarcasm:
        variant = "with_sarcasm"
        summarizer = pipeline.summarizer_with_sarcasm
    else:
        variant = "without_sentiment"
        summarizer = pipeline.summarizer_without_sentiment

    logger.info(
        "Running summarization (%s, source=%s) on %d samples...",
        variant,
        source,
        len(records),
    )

    for index, record in enumerate(records, 1):
        text = record["text"]
        reference = record["summary"]
        sentiment = None
        sarcasm = None

        if use_sentiment:
            sentiment = record.get("label")
            if not sentiment or sentiment in ("sarcastic", "non-sarcastic"):
                if use_sarcasm:
                    sarcasm_result = pipeline.detect_sarcasm(text)
                    sarcasm_for_sentiment = (
                        sarcasm_result["is_sarcastic"] if sarcasm_result else None
                    )
                    sentiment_result = pipeline.sentiment_classifier.classify(
                        text,
                        model=pipeline.config.pipeline_config.sentiment_model,
                        sarcasm=sarcasm_for_sentiment,
                        sarcasm_aware=sarcasm_for_sentiment is not None,
                    )
                else:
                    sentiment_result = pipeline.sentiment_classifier.classify(
                        text, model=pipeline.config.pipeline_config.sentiment_model
                    )
                sentiment = sentiment_result["sentiment"]

        if use_sarcasm:
            if "sarcasm" in record and record["sarcasm"] is not None:
                sarcasm = bool(record["sarcasm"])
            elif record.get("label") in ("sarcastic", "non-sarcastic"):
                sarcasm = record["label"] == "sarcastic"
            else:
                sarcasm_result = pipeline.detect_sarcasm(text)
                if sarcasm_result:
                    sarcasm = sarcasm_result["is_sarcastic"]

        result = summarizer.summarize(
            text,
            sentiment=sentiment,
            sarcasm=sarcasm,
            model=pipeline.config.pipeline_config.summarization_model,
        )
        predicted = result["summary"]

        rouge = Evaluator.compute_rouge(predicted, reference)
        predictions.append(predicted)
        references.append(reference)
        samples.append(
            {
                "index": index,
                "text": text,
                "reference_summary": reference,
                "predicted_summary": predicted,
                "rouge1_f": rouge["rouge1_f"],
                "rougeL_f": rouge["rougeL_f"],
                "sentiment_used": sentiment if use_sentiment else None,
                "sarcasm_used": sarcasm if use_sarcasm else None,
            }
        )

        logger.info(
            "[%d/%d] ROUGE-L=%.3f | ref: %s",
            index,
            len(records),
            rouge["rougeL_f"],
            truncate(reference, 60),
        )
        logger.info("         pred: %s", truncate(predicted, 60))

    aggregate = Evaluator.evaluate_summarization(
        predictions, references, use_rouge=True, use_bertscore=False
    )

    return {
        "task": "summarization",
        "source": source,
        "variant": variant,
        "metrics": aggregate,
        "samples": samples,
    }


def print_summary(report: Dict[str, Any], logger: logging.Logger) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info("Run ID: %s", report["run_id"])
    logger.info("Model: %s", report["model"])
    logger.info("Prompting: %s", report["prompting_technique"])
    logger.info("Limit per task: %s", report["limit"])
    logger.info("")

    for task_result in report["tasks"]:
        task = task_result["task"]
        logger.info("--- %s ---", task.upper())

        if task in ("sarcasm", "sentiment"):
            metrics = task_result["metrics"]
            logger.info("  Variant:   %s", task_result.get("variant", "-"))
            logger.info("  Accuracy:  %.3f", metrics["accuracy"])
            logger.info("  Precision: %.3f", metrics["precision"])
            logger.info("  Recall:    %.3f", metrics["recall"])
            logger.info("  F1:        %.3f", metrics["f1"])

            correct = sum(1 for s in task_result["samples"] if s["correct"])
            total = len(task_result["samples"])
            logger.info("  Correct:   %d / %d", correct, total)

            logger.info("  Preview (first 3):")
            for sample in task_result["samples"][:3]:
                mark = "OK" if sample["correct"] else "X"
                logger.info(
                    "    [%s] ref=%s pred=%s | %s",
                    mark,
                    sample["reference"],
                    sample["predicted"],
                    truncate(sample["text"], 70),
                )

        elif task == "summarization":
            rouge = task_result["metrics"]["rouge"]
            logger.info("  Source:    %s", task_result.get("source"))
            logger.info("  Variant:   %s", task_result.get("variant"))
            logger.info("  ROUGE-1:   %.3f", rouge["rouge1_f"])
            logger.info("  ROUGE-2:   %.3f", rouge["rouge2_f"])
            logger.info("  ROUGE-L:   %.3f", rouge["rougeL_f"])

            logger.info("  Preview (first 2):")
            for sample in task_result["samples"][:2]:
                logger.info("    ref:  %s", truncate(sample["reference_summary"], 70))
                logger.info("    pred: %s", truncate(sample["predicted_summary"], 70))
                logger.info("    ROUGE-L: %.3f", sample["rougeL_f"])
                logger.info("")

    logger.info("=" * 70)
    logger.info("Full JSON: %s", report["output_file"])
    logger.info("Log file:  %s", report["log_file"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM on prepared datasets")
    parser.add_argument(
        "--task",
        choices=["sarcasm", "sentiment", "summarization", "all"],
        default="all",
        help="Which task to evaluate",
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
        help="Dataset split to use",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of samples per task (keep small for llama2)",
    )
    parser.add_argument(
        "--technique",
        choices=["zero-shot", "few-shot", "chain-of-thought"],
        default=None,
        help="Override prompting technique from config",
    )
    parser.add_argument(
        "--with-sentiment",
        action="store_true",
        help="Also run sentiment-aware summarization",
    )
    parser.add_argument(
        "--with-sarcasm",
        action="store_true",
        help="Also run sarcasm-aware summarization",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"run_{run_id}.log"
    output_file = RESULTS_DIR / f"evaluation_{run_id}.json"

    logger = setup_logging(log_file)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config_path = PROJECT_ROOT / "config" / "config.yaml"
    pipeline = NLPPipeline(config_path=str(config_path))

    if args.technique:
        pipeline.config.pipeline_config.prompting_technique = args.technique
        pipeline._initialize_components()

    logger.info("Starting evaluation run: %s", run_id)
    logger.info("Config: %s", config_path)

    if not pipeline.check_setup():
        logger.error("Ollama is not available. Start with: ollama serve")
        return 1

    loader = DatasetLoader()
    task_results: List[Dict[str, Any]] = []

    tasks_to_run = (
        ["sarcasm", "sentiment", "summarization"]
        if args.task == "all"
        else [args.task]
    )

    for task in tasks_to_run:
        logger.info("")
        logger.info(">>> TASK: %s", task.upper())

        if task == "sarcasm":
            records = loader.load_task_split("sarcasm", split=args.split)
            records = records[: args.limit]
            task_results.append(evaluate_sarcasm(pipeline, records, logger))

        elif task == "sentiment":
            records = loader.load_task_split("sentiment", split=args.split)
            records = records[: args.limit]
            task_results.append(evaluate_sentiment(pipeline, records, logger, use_sarcasm=False))
            if args.with_sarcasm:
                task_results.append(evaluate_sentiment(pipeline, records, logger, use_sarcasm=True))

        elif task == "summarization":
            records = loader.load_task_split(
                "summarization", split=args.split, source=args.source
            )
            records = records[: args.limit]

            summarization_variants = [(False, False)]
            if args.with_sentiment:
                summarization_variants.append((True, False))
            if args.with_sarcasm:
                summarization_variants.append((False, True))
            if args.with_sentiment and args.with_sarcasm:
                summarization_variants.append((True, True))

            for use_sentiment, use_sarcasm in summarization_variants:
                task_results.append(
                    evaluate_summarization(
                        pipeline,
                        records,
                        source=args.source,
                        logger=logger,
                        use_sentiment=use_sentiment,
                        use_sarcasm=use_sarcasm,
                    )
                )

    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model": pipeline.config.model_config.name,
        "prompting_technique": pipeline.config.pipeline_config.prompting_technique,
        "split": args.split,
        "limit": args.limit,
        "tasks": task_results,
        "output_file": str(output_file.relative_to(PROJECT_ROOT)),
        "log_file": str(log_file.relative_to(PROJECT_ROOT)),
    }

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print_summary(report, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
