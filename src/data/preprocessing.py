"""Text preprocessing and dataset preparation utilities."""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

STANDARD_FIELDS = ("text", "label", "summary")


def rating_to_sentiment(rating: float) -> Optional[str]:
    """Map Amazon star rating to sentiment label."""
    if rating is None or pd.isna(rating):
        return None

    try:
        stars = int(round(float(rating)))
    except (TypeError, ValueError):
        return None

    if stars <= 2:
        return "negative"
    if stars == 3:
        return "neutral"
    if stars >= 4:
        return "positive"
    return None


def sarcasm_label_to_str(value: Any) -> Optional[str]:
    """Convert iSarcasmEval numeric/boolean label to string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "sarcastic", "true", "yes"}:
            return "sarcastic"
        if normalized in {"0", "non-sarcastic", "not sarcastic", "false", "no"}:
            return "non-sarcastic"
        return normalized

    try:
        return "sarcastic" if int(value) == 1 else "non-sarcastic"
    except (TypeError, ValueError):
        return None


class TextPreprocessor:
    """Clean and validate text records for NLP tasks."""

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(
        self,
        min_text_length: int = 10,
        min_summary_length: int = 3,
        remove_duplicates: bool = True,
        lowercase_labels: bool = True,
    ):
        self.min_text_length = min_text_length
        self.min_summary_length = min_summary_length
        self.remove_duplicates = remove_duplicates
        self.lowercase_labels = lowercase_labels

    def clean_text(self, text: Any) -> str:
        """Apply basic text normalization."""
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""

        cleaned = str(text)
        cleaned = html.unescape(cleaned)
        cleaned = self.HTML_TAG_PATTERN.sub(" ", cleaned)
        cleaned = self.URL_PATTERN.sub(" ", cleaned)
        cleaned = cleaned.replace("\u00a0", " ")
        cleaned = self.WHITESPACE_PATTERN.sub(" ", cleaned).strip()
        return cleaned

    def normalize_record(self, record: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Normalize a record to the standard schema."""
        text = self.clean_text(record.get("text"))
        summary = self.clean_text(record.get("summary")) if record.get("summary") is not None else None
        label = record.get("label")

        if summary == "":
            summary = None

        if label is not None:
            label = str(label).strip()
            if self.lowercase_labels:
                label = label.lower()

        return {
            "text": text,
            "label": label,
            "summary": summary,
        }

    def is_valid_record(
        self,
        record: Dict[str, Optional[str]],
        require_label: bool = False,
        require_summary: bool = False,
    ) -> bool:
        """Validate record against task requirements."""
        text = record.get("text") or ""
        label = record.get("label")
        summary = record.get("summary")

        if len(text) < self.min_text_length:
            return False
        if require_label and not label:
            return False
        if require_summary:
            if not summary or len(summary) < self.min_summary_length:
                return False
        return True

    def preprocess_records(
        self,
        records: List[Dict[str, Any]],
        require_label: bool = False,
        require_summary: bool = False,
    ) -> Tuple[List[Dict[str, Optional[str]]], Dict[str, int]]:
        """Clean, validate, deduplicate, and report preprocessing stats."""
        stats = {
            "input_count": len(records),
            "missing_text": 0,
            "missing_label": 0,
            "missing_summary": 0,
            "too_short_text": 0,
            "too_short_summary": 0,
            "duplicates_removed": 0,
            "output_count": 0,
        }

        processed: List[Dict[str, Optional[str]]] = []
        seen_texts = set()

        for raw in records:
            normalized = self.normalize_record(raw)

            if not normalized["text"]:
                stats["missing_text"] += 1
                continue
            if len(normalized["text"]) < self.min_text_length:
                stats["too_short_text"] += 1
                continue
            if require_label and not normalized["label"]:
                stats["missing_label"] += 1
                continue
            if require_summary:
                if not normalized["summary"]:
                    stats["missing_summary"] += 1
                    continue
                if len(normalized["summary"]) < self.min_summary_length:
                    stats["too_short_summary"] += 1
                    continue

            if self.remove_duplicates:
                key = normalized["text"].lower()
                if key in seen_texts:
                    stats["duplicates_removed"] += 1
                    continue
                seen_texts.add(key)

            processed.append(normalized)

        stats["output_count"] = len(processed)
        return processed, stats

    def split_records(
        self,
        records: List[Dict[str, Optional[str]]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
    ) -> Dict[str, List[Dict[str, Optional[str]]]]:
        """Create train/validation/test splits."""
        if not records:
            return {"train": [], "validation": [], "test": []}

        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

        if len(records) < 3:
            return {"train": records, "validation": [], "test": []}

        train_records, temp_records = train_test_split(
            records,
            test_size=(1.0 - train_ratio),
            random_state=random_seed,
            shuffle=True,
        )

        if not temp_records or val_ratio + test_ratio == 0:
            return {"train": train_records, "validation": [], "test": []}

        relative_test_size = test_ratio / (val_ratio + test_ratio)
        val_records, test_records = train_test_split(
            temp_records,
            test_size=relative_test_size,
            random_state=random_seed,
            shuffle=True,
        )

        return {
            "train": train_records,
            "validation": val_records,
            "test": test_records,
        }
