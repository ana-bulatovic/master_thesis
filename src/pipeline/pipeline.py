from typing import List, Dict, Optional
import json
import os
from datetime import datetime

from config import Config
from llm.ollama_client import OllamaClient
from pipeline.sarcasm_detector import SarcasmDetector
from pipeline.sentiment_classifier import SentimentClassifier
from pipeline.summarizer import TextSummarizer
from evaluation.evaluator import Evaluator


class NLPPipeline:
    """Main pipeline orchestrating sarcasm detection, sentiment classification, and summarization"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = Config(config_path)
        self.client = OllamaClient(self.config.model_config)

        self.sarcasm_detector = None
        self.sentiment_classifier = None
        self.summarizer_without_sentiment = None
        self.summarizer_with_sentiment = None
        self.summarizer_with_sarcasm = None
        self.summarizer_with_sentiment_and_sarcasm = None

        self._initialize_components()
        os.makedirs(self.config.pipeline_config.output_dir, exist_ok=True)

    def _initialize_components(self):
        technique = self.config.pipeline_config.prompting_technique
        pipeline_config = self.config.pipeline_config

        if pipeline_config.use_sarcasm_detection or pipeline_config.use_sarcasm_for_summarization:
            self.sarcasm_detector = SarcasmDetector(self.client, prompting_technique=technique)

        self.sentiment_classifier = SentimentClassifier(self.client, prompting_technique=technique)
        self.summarizer_without_sentiment = TextSummarizer(
            self.client,
            prompting_technique=technique,
            use_sentiment_info=False,
            use_sarcasm_info=False,
        )
        self.summarizer_with_sentiment = TextSummarizer(
            self.client,
            prompting_technique=technique,
            use_sentiment_info=True,
            use_sarcasm_info=False,
        )
        self.summarizer_with_sarcasm = TextSummarizer(
            self.client,
            prompting_technique=technique,
            use_sentiment_info=False,
            use_sarcasm_info=True,
        )
        self.summarizer_with_sentiment_and_sarcasm = TextSummarizer(
            self.client,
            prompting_technique=technique,
            use_sentiment_info=True,
            use_sarcasm_info=True,
        )

    def _detect_sarcasm(self, review: str) -> Optional[Dict]:
        if not self.sarcasm_detector:
            return None

        return self.sarcasm_detector.detect(
            review, model=self.config.pipeline_config.sarcasm_model
        )

    def _pipeline_variant_label(self) -> str:
        pipeline_config = self.config.pipeline_config
        uses_sarcasm = pipeline_config.use_sarcasm_for_summarization
        uses_sentiment = pipeline_config.use_sentiment_for_summarization

        if uses_sarcasm and uses_sentiment:
            return "sarcasm_sentiment_summarization"
        if uses_sarcasm:
            return "sarcasm_summarization"
        if uses_sentiment:
            return "sentiment_summarization"
        return "summarization_only"

    def check_setup(self) -> bool:
        print("Checking Ollama connection...")

        if not self.client.check_connection():
            print("Error: Ollama server is not running!")
            print("Make sure Ollama is started with: ollama serve")
            return False

        print("Ollama connection established")
        available_models = self.client.list_models()
        print(f"\nAvailable models: {available_models}")

        required_models = [
            self.config.pipeline_config.sarcasm_model,
            self.config.pipeline_config.sentiment_model,
            self.config.pipeline_config.summarization_model,
        ]

        for model in required_models:
            if not any(model in available_name for available_name in available_models):
                print(f"Warning: Model '{model}' not found in available models")

        return True

    def process_review(self, review: str) -> Dict:
        pipeline_config = self.config.pipeline_config
        result = {
            "original_review": review,
            "timestamp": datetime.now().isoformat(),
            "pipeline_variant": self._pipeline_variant_label(),
        }

        sarcasm_result = None
        if pipeline_config.use_sarcasm_detection or pipeline_config.use_sarcasm_for_summarization:
            sarcasm_result = self._detect_sarcasm(review)
        result["sarcasm_detection"] = sarcasm_result

        sentiment_result = self.sentiment_classifier.classify(
            review, model=pipeline_config.sentiment_model
        )
        result["sentiment_classification"] = sentiment_result

        sarcasm_flag = sarcasm_result["is_sarcastic"] if sarcasm_result else None
        sentiment_label = sentiment_result["sentiment"]

        result["summarization_without_sentiment"] = self.summarizer_without_sentiment.summarize(
            review,
            sentiment=None,
            sarcasm=None,
            model=pipeline_config.summarization_model,
        )

        if pipeline_config.use_sentiment_for_summarization:
            result["summarization_with_sentiment"] = self.summarizer_with_sentiment.summarize(
                review,
                sentiment=sentiment_label,
                sarcasm=None,
                model=pipeline_config.summarization_model,
            )
        else:
            result["summarization_with_sentiment"] = None

        if pipeline_config.use_sarcasm_for_summarization:
            result["summarization_with_sarcasm"] = self.summarizer_with_sarcasm.summarize(
                review,
                sentiment=None,
                sarcasm=sarcasm_flag,
                model=pipeline_config.summarization_model,
            )
        else:
            result["summarization_with_sarcasm"] = None

        if pipeline_config.use_sentiment_for_summarization and pipeline_config.use_sarcasm_for_summarization:
            result["summarization_with_sentiment_and_sarcasm"] = (
                self.summarizer_with_sentiment_and_sarcasm.summarize(
                    review,
                    sentiment=sentiment_label,
                    sarcasm=sarcasm_flag,
                    model=pipeline_config.summarization_model,
                )
            )
        else:
            result["summarization_with_sentiment_and_sarcasm"] = None

        return result

    def process_batch(self, reviews: List[str]) -> List[Dict]:
        results = []
        print(f"\nProcessing {len(reviews)} reviews...")

        for index, review in enumerate(reviews, 1):
            print(f"  [{index}/{len(reviews)}] Processing review...")
            results.append(self.process_review(review))

        return results

    def save_results(self, results: List[Dict], filename: str = "results.json"):
        output_path = os.path.join(self.config.pipeline_config.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_path}")

    def print_result(self, result: Dict):
        print("\n" + "=" * 80)
        print("REVIEW:", result["original_review"][:100] + "...")

        if result.get("sarcasm_detection"):
            print(f"\nSarcasm: {result['sarcasm_detection']['is_sarcastic']}")

        print(f"\nSentiment: {result['sentiment_classification']['sentiment'].upper()}")
        print(f"\nSummary (standard): {result['summarization_without_sentiment']['summary']}")

        if result.get("summarization_with_sentiment"):
            print(
                "Summary (sentiment-aware): "
                f"{result['summarization_with_sentiment']['summary']}"
            )

        if result.get("summarization_with_sarcasm"):
            print(
                "Summary (sarcasm-aware): "
                f"{result['summarization_with_sarcasm']['summary']}"
            )

        if result.get("summarization_with_sentiment_and_sarcasm"):
            print(
                "Summary (sentiment + sarcasm): "
                f"{result['summarization_with_sentiment_and_sarcasm']['summary']}"
            )
        print("=" * 80)
