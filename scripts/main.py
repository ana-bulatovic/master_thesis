#!/usr/bin/env python3
"""Main entry point for the NLP master thesis pipeline."""

import sys

from path_setup import PROJECT_ROOT  # noqa: F401
from pipeline.pipeline import NLPPipeline
from evaluation.evaluator import Evaluator

SAMPLE_REVIEWS = [
    "Ovaj proizvod je čudesna kupnja! Odličan kvalitet i brza dostava. Veoma sam zadovoljan.",
    "Ponos sam što sam kupio ovo! Nije ni loše, nije specijalno. Prosečan proizvod.",
    "Najgori proizvod koji sam ikada kupio! Otišao je u kvar posle dva dana. Totalno razočaran.",
    "Divno, poslali su mi proizvod u komadima. 'Odličan' kvalitet!",
    "Proizvod je dobar, funkcioniše kako treba. Nema razloga za žalbu.",
]


def demonstrate_pipeline() -> bool:
    print("=" * 80)
    print("NLP PIPELINE: Sarcasm Detection, Sentiment Classification & Summarization")
    print("=" * 80)

    pipeline = NLPPipeline(config_path=str(PROJECT_ROOT / "config" / "config.yaml"))
    pipeline.print_config_summary()

    if not pipeline.check_setup():
        print("\nSetup failed. Please ensure Ollama is running and fine-tuned model exists.")
        return False

    results = pipeline.process_batch(SAMPLE_REVIEWS)

    pipeline.save_results(results, "pipeline_results.json")
    return True


def demonstrate_individual_tasks() -> bool:
    pipeline = NLPPipeline(config_path=str(PROJECT_ROOT / "config" / "config.yaml"))
    if not pipeline.check_setup():
        return False

    test_review = "Oh great, another amazing product that broke after two days."

    sarcasm_result = pipeline.detect_sarcasm(test_review)
    if sarcasm_result:
        print(f"Sarcasm: {sarcasm_result['is_sarcastic']}")

    sentiment_result = pipeline.sentiment_classifier.classify(test_review)
    print(f"Sentiment: {sentiment_result['sentiment']}")

    summary_result = pipeline.summarizer_without_sentiment.summarize(test_review)
    print(f"Summary: {summary_result['summary']}")
    return True


def run_evaluation_example():
    predictions = ["positive", "negative", "positive", "neutral", "negative"]
    references = ["positive", "negative", "neutral", "neutral", "negative"]
    metrics = Evaluator.evaluate_classification(predictions, references)
    print(metrics)


def main():
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "individual":
            demonstrate_individual_tasks()
        elif command == "evaluate":
            run_evaluation_example()
        else:
            print("Unknown command. Use: individual | evaluate")
    else:
        demonstrate_pipeline()


if __name__ == "__main__":
    main()
