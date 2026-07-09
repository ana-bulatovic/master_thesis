"""NLP pipeline task modules."""

from .pipeline import NLPPipeline
from .sarcasm_detector import SarcasmDetector
from .sentiment_classifier import SentimentClassifier
from .summarizer import TextSummarizer

__all__ = [
    "NLPPipeline",
    "SarcasmDetector",
    "SentimentClassifier",
    "TextSummarizer",
]
