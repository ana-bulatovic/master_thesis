from typing import List, Dict, Optional
import json
import os
from datetime import datetime
from pathlib import Path

from config import Config
from llm.ollama_client import OllamaClient
from models.sarcasm_classifier import FinetunedSarcasmClassifier
from pipeline.sarcasm_detector import SarcasmDetector
from pipeline.sentiment_classifier import SentimentClassifier
from pipeline.summarizer import TextSummarizer
from paths import PROJECT_ROOT


class NLPPipeline:
    """Main pipeline orchestrating sarcasm detection, sentiment classification, and summarization"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = Config(config_path)
        self.client = OllamaClient(self.config.model_config)

        self.sarcasm_detector = None
        self.finetuned_sarcasm_classifier = None
        self.sentiment_classifier = None
        self.summarizer_without_sentiment = None
        self.summarizer_with_sentiment = None
        self.summarizer_with_sarcasm = None
        self.summarizer_with_sentiment_and_sarcasm = None

        self._initialize_components()
        os.makedirs(self.config.pipeline_config.output_dir, exist_ok=True)

    def _uses_sarcasm(self) -> bool:
        pipeline_config = self.config.pipeline_config
        return pipeline_config.use_sarcasm_detection or pipeline_config.use_sarcasm_for_summarization

    def _initialize_components(self):
        technique = self.config.pipeline_config.prompting_technique
        pipeline_config = self.config.pipeline_config

        if self._uses_sarcasm():
            if pipeline_config.sarcasm_backend == "fine-tuned":
                model_path = PROJECT_ROOT / pipeline_config.sarcasm_finetuned_path
                self.finetuned_sarcasm_classifier = FinetunedSarcasmClassifier(model_path)
            else:
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

    def detect_sarcasm(self, review: str) -> Optional[Dict]:
        return self._detect_sarcasm(review)

    def _detect_sarcasm(self, review: str) -> Optional[Dict]:
        pipeline_config = self.config.pipeline_config

        if pipeline_config.sarcasm_backend == "fine-tuned":
            if not self.finetuned_sarcasm_classifier:
                return None
            return self.finetuned_sarcasm_classifier.predict(review)

        if not self.sarcasm_detector:
            return None

        return self.sarcasm_detector.detect(
            review, model=pipeline_config.sarcasm_model
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

    def print_config_summary(self) -> None:
        pipeline_config = self.config.pipeline_config
        print("\n" + "=" * 80)
        print("PIPELINE CONFIGURATION")
        print("=" * 80)
        print(f"  Prompting technique:     {pipeline_config.prompting_technique}")
        print(f"  Sentiment model (Ollama):  {pipeline_config.sentiment_model}")
        print(f"  Summarization (Ollama):    {pipeline_config.summarization_model}")

        if self._uses_sarcasm():
            if pipeline_config.sarcasm_backend == "fine-tuned":
                print(f"  Sarcasm backend:           fine-tuned DistilBERT")
                print(f"  Sarcasm model path:        {pipeline_config.sarcasm_finetuned_path}")
            else:
                print(f"  Sarcasm backend:           LLM (Ollama)")
                print(f"  Sarcasm model (Ollama):    {pipeline_config.sarcasm_model}")

        print(f"  Sentiment in summary:      {pipeline_config.use_sentiment_for_summarization}")
        print(f"  Sarcasm in summary:        {pipeline_config.use_sarcasm_for_summarization}")
        print("=" * 80)

    def check_setup(self) -> bool:
        print("Checking pipeline setup...")
        pipeline_config = self.config.pipeline_config
        ok = True

        if self._uses_sarcasm() and pipeline_config.sarcasm_backend == "fine-tuned":
            model_path = PROJECT_ROOT / pipeline_config.sarcasm_finetuned_path
            if model_path.exists():
                print(f"OK  Fine-tuned sarcasm model: {model_path.relative_to(PROJECT_ROOT)}")
            else:
                print(f"ERR Fine-tuned sarcasm model not found: {model_path}")
                print("    Train first: python scripts/train_sarcasm.py")
                ok = False

        print("\nChecking Ollama connection (sentiment + summarization)...")
        if not self.client.check_connection():
            print("ERR Ollama server is not running!")
            print("    Start with: ollama serve")
            ok = False
        else:
            print("OK  Ollama connection established")
            available_models = self.client.list_models()
            print(f"    Available models: {', '.join(available_models) or '(none)'}")

            for label, model in [
                ("Sentiment", pipeline_config.sentiment_model),
                ("Summarization", pipeline_config.summarization_model),
            ]:
                if not any(model in available_name for available_name in available_models):
                    print(f"WARN Model '{model}' ({label}) not found in Ollama")
                else:
                    print(f"OK  {label} model: {model}")

            if pipeline_config.sarcasm_backend == "llm" and self._uses_sarcasm():
                model = pipeline_config.sarcasm_model
                if not any(model in available_name for available_name in available_models):
                    print(f"WARN Sarcasm LLM model '{model}' not found in Ollama")

        return ok

    def process_review(self, review: str, verbose: bool = True) -> Dict:
        pipeline_config = self.config.pipeline_config
        result = {
            "original_review": review,
            "timestamp": datetime.now().isoformat(),
            "pipeline_variant": self._pipeline_variant_label(),
            "sarcasm_backend": pipeline_config.sarcasm_backend,
        }

        if verbose:
            print("\n" + "-" * 80)
            preview = review if len(review) <= 120 else review[:117] + "..."
            print(f"INPUT: {preview}")

        sarcasm_result = None
        if self._uses_sarcasm():
            if verbose:
                backend = (
                    "fine-tuned DistilBERT"
                    if pipeline_config.sarcasm_backend == "fine-tuned"
                    else f"LLM ({pipeline_config.sarcasm_model})"
                )
                print(f"  [1/3] Sarcasm detection ({backend})...")
            sarcasm_result = self._detect_sarcasm(review)
            if verbose and sarcasm_result:
                label = "SARCASTIC" if sarcasm_result["is_sarcastic"] else "NON-SARCASTIC"
                extra = ""
                if "confidence" in sarcasm_result:
                    extra = f" (confidence: {sarcasm_result['confidence']:.2%})"
                print(f"        -> {label}{extra}")
        result["sarcasm_detection"] = sarcasm_result

        if verbose:
            print(f"  [2/3] Sentiment classification (LLM: {pipeline_config.sentiment_model})...")
        sentiment_result = self.sentiment_classifier.classify(
            review, model=pipeline_config.sentiment_model
        )
        result["sentiment_classification"] = sentiment_result
        if verbose:
            print(f"        -> {sentiment_result['sentiment'].upper()}")

        sarcasm_flag = sarcasm_result["is_sarcastic"] if sarcasm_result else None
        sentiment_label = sentiment_result["sentiment"]

        if verbose:
            print(f"  [3/3] Summarization (LLM: {pipeline_config.summarization_model})...")

        result["summarization_without_sentiment"] = self.summarizer_without_sentiment.summarize(
            review,
            sentiment=None,
            sarcasm=None,
            model=pipeline_config.summarization_model,
        )
        if verbose:
            print("        -> baseline done")

        if pipeline_config.use_sentiment_for_summarization:
            result["summarization_with_sentiment"] = self.summarizer_with_sentiment.summarize(
                review,
                sentiment=sentiment_label,
                sarcasm=None,
                model=pipeline_config.summarization_model,
            )
            if verbose:
                print("        -> +sentiment done")
        else:
            result["summarization_with_sentiment"] = None

        if pipeline_config.use_sarcasm_for_summarization:
            result["summarization_with_sarcasm"] = self.summarizer_with_sarcasm.summarize(
                review,
                sentiment=None,
                sarcasm=sarcasm_flag,
                model=pipeline_config.summarization_model,
            )
            if verbose:
                print("        -> +sarcasm done")
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
            if verbose:
                print("        -> +sentiment+sarcasm done")
        else:
            result["summarization_with_sentiment_and_sarcasm"] = None

        if verbose:
            self._print_result_summary(result)

        return result

    def process_batch(self, reviews: List[str], verbose: bool = True) -> List[Dict]:
        results = []
        total = len(reviews)
        print(f"\nProcessing {total} review(s)...")

        for index, review in enumerate(reviews, 1):
            if verbose:
                print(f"\n{'#' * 80}")
                print(f"REVIEW {index}/{total}")
                print(f"{'#' * 80}")
            results.append(self.process_review(review, verbose=verbose))

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"COMPLETED: {total} review(s) processed")
            print(f"{'=' * 80}")

        return results

    def save_results(self, results: List[Dict], filename: str = "results.json"):
        output_path = os.path.join(self.config.pipeline_config.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}")

    def _print_result_summary(self, result: Dict) -> None:
        print("\n  RESULT:")
        if result.get("sarcasm_detection"):
            sarcasm = result["sarcasm_detection"]
            status = "SARCASTIC" if sarcasm["is_sarcastic"] else "NON-SARCASTIC"
            line = f"    Sarcasm:   {status}"
            if sarcasm.get("confidence") is not None:
                line += f" ({sarcasm['confidence']:.1%})"
            print(line)

        sentiment = result["sentiment_classification"]["sentiment"].upper()
        print(f"    Sentiment: {sentiment}")
        print(f"    Summary (baseline):          {result['summarization_without_sentiment']['summary']}")

        if result.get("summarization_with_sentiment"):
            print(
                "    Summary (+sentiment):        "
                f"{result['summarization_with_sentiment']['summary']}"
            )
        if result.get("summarization_with_sarcasm"):
            print(
                "    Summary (+sarcasm):          "
                f"{result['summarization_with_sarcasm']['summary']}"
            )
        if result.get("summarization_with_sentiment_and_sarcasm"):
            print(
                "    Summary (+sentiment+sarcasm):"
                f" {result['summarization_with_sentiment_and_sarcasm']['summary']}"
            )

    def print_result(self, result: Dict):
        print("\n" + "=" * 80)
        preview = result["original_review"]
        if len(preview) > 100:
            preview = preview[:97] + "..."
        print(f"REVIEW: {preview}")
        self._print_result_summary(result)
        print("=" * 80)
