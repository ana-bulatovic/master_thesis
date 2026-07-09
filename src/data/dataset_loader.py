"""Unified dataset loading for sarcasm, sentiment, and summarization tasks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml

from paths import PROJECT_ROOT
from data.preprocessing import TextPreprocessor, rating_to_sentiment, sarcasm_label_to_str

logger = logging.getLogger(__name__)


@dataclass
class DatasetRecord:
    """Standard record format used across all tasks."""

    text: str
    label: Optional[str] = None
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "text": self.text,
            "label": self.label,
            "summary": self.summary,
        }


class DatasetLoader:
    """Load, preprocess, and persist thesis datasets."""

    ISARCSMEVAL_RAW_BASE = (
        "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main"
    )

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or PROJECT_ROOT / "config" / "datasets.yaml")
        self.config = self._load_config()
        self.preprocessor = TextPreprocessor(
            min_text_length=self.config["preprocessing"]["min_text_length"],
            min_summary_length=self.config["preprocessing"]["min_summary_length"],
            remove_duplicates=self.config["preprocessing"]["remove_duplicates"],
            lowercase_labels=self.config["preprocessing"]["lowercase_labels"],
        )
        self.raw_dir = PROJECT_ROOT / self.config["paths"]["raw_dir"]
        self.processed_dir = PROJECT_ROOT / self.config["paths"]["processed_dir"]

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _ensure_dirs(self, *parts: str) -> Path:
        path = self.processed_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _download_file(self, url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            logger.info("Using cached file: %s", destination)
            return destination

        logger.info("Downloading %s", url)
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return destination

    def _save_jsonl(self, records: List[Dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _save_metadata(self, metadata: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def load_jsonl(path: Path) -> List[Dict[str, Any]]:
        records = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

  # ------------------------------------------------------------------
  # iSarcasmEval
  # ------------------------------------------------------------------

    def _load_isarcasmeval_csv(self, relative_path: str) -> pd.DataFrame:
        raw_path = self.raw_dir / "sarcasm" / Path(relative_path).name
        url = f"{self.ISARCSMEVAL_RAW_BASE}/{relative_path}"
        self._download_file(url, raw_path)
        return pd.read_csv(raw_path)

    def load_isarcasmeval_raw(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load iSarcasmEval English train/test splits."""
        cfg = self.config["isarcasmeval"]
        train_df = self._load_isarcasmeval_csv(cfg["train_file"])
        test_df = self._load_isarcasmeval_csv(cfg["test_file"])

        train_text_col = cfg["text_column"]
        test_text_col = cfg.get("test_text_column", "text")
        label_col = cfg["label_column"]

        train_records = []
        for _, row in train_df.iterrows():
            text = row.get(train_text_col) or row.get("text") or row.get("tweet")
            label = sarcasm_label_to_str(row.get(label_col))
            if text and label:
                train_records.append(
                    DatasetRecord(text=str(text), label=label, summary=None).to_dict()
                )

        test_records = []
        for _, row in test_df.iterrows():
            text = row.get(test_text_col) or row.get("text") or row.get("tweet")
            label = sarcasm_label_to_str(row.get(label_col))
            if text and label:
                test_records.append(
                    DatasetRecord(text=str(text), label=label, summary=None).to_dict()
                )

        return {"train": train_records, "test": test_records}

    def prepare_isarcasmeval(self) -> Dict[str, Any]:
        """Prepare sarcasm detection dataset."""
        raw_splits = self.load_isarcasmeval_raw()
        output_dir = self._ensure_dirs("sarcasm")

        metadata = {
            "dataset": "iSarcasmEval",
            "task": "sarcasm_detection",
            "language": self.config["isarcasmeval"]["language"],
            "splits": {},
        }

        for split_name, records in raw_splits.items():
            processed, stats = self.preprocessor.preprocess_records(
                records, require_label=True, require_summary=False
            )
            out_path = output_dir / f"{split_name}.jsonl"
            self._save_jsonl(processed, out_path)
            metadata["splits"][split_name] = {
                "path": str(out_path.relative_to(PROJECT_ROOT)),
                "stats": stats,
            }
            logger.info(
                "iSarcasmEval %s: %s records saved to %s",
                split_name,
                len(processed),
                out_path,
            )

        metadata_path = output_dir / "metadata.json"
        self._save_metadata(metadata, metadata_path)
        return metadata

  # ------------------------------------------------------------------
  # Amazon Reviews 2023
  # ------------------------------------------------------------------

    def _load_amazon_hf_dataset(self):
        from datasets import load_dataset

        cfg = self.config["amazon_reviews_2023"]
        max_samples = cfg.get("max_samples")

        loaders = [
            (
                "fallback",
                lambda: load_dataset(
                    cfg["fallback_hf_dataset"],
                    data_dir=cfg["fallback_data_dir"],
                    split="train",
                ),
            ),
            (
                "primary",
                lambda: load_dataset(
                    cfg["hf_dataset"],
                    cfg["hf_config"],
                    split=cfg.get("hf_split", "full"),
                ),
            ),
        ]

        last_error = None
        for source_name, loader_fn in loaders:
            try:
                logger.info("Loading Amazon Reviews 2023 via %s loader", source_name)
                dataset = loader_fn()
                if max_samples:
                    dataset = dataset.select(range(min(max_samples, len(dataset))))
                return dataset
            except Exception as exc:
                last_error = exc
                logger.warning("Amazon loader '%s' failed: %s", source_name, exc)

        raise RuntimeError(f"Could not load Amazon Reviews 2023: {last_error}")

    def load_amazon_reviews_raw(self) -> List[Dict[str, Any]]:
        """Load Amazon Reviews 2023 records with rating, text, and title."""
        dataset = self._load_amazon_hf_dataset()
        cfg = self.config["amazon_reviews_2023"]

        records = []
        for row in dataset:
            rating = row.get(cfg["rating_column"])
            text = row.get(cfg["text_column"])
            title = row.get(cfg["title_column"])
            sentiment = rating_to_sentiment(rating)

            if not text:
                continue

            records.append(
                {
                    "text": text,
                    "label": sentiment,
                    "summary": title,
                    "metadata": {
                        "rating": rating,
                        "source": "amazon_reviews_2023",
                    },
                }
            )
        return records

    def prepare_amazon_sentiment(self) -> Dict[str, Any]:
        """Prepare Amazon Reviews for sentiment classification."""
        records = self.load_amazon_reviews_raw()
        split_cfg = self.config["splits"]

        processed, stats = self.preprocessor.preprocess_records(
            records, require_label=True, require_summary=False
        )
        splits = self.preprocessor.split_records(
            processed,
            train_ratio=split_cfg["train_ratio"],
            val_ratio=split_cfg["val_ratio"],
            test_ratio=split_cfg["test_ratio"],
            random_seed=split_cfg["random_seed"],
        )

        output_dir = self._ensure_dirs("sentiment")
        metadata = {
            "dataset": "Amazon Reviews 2023",
            "task": "sentiment_classification",
            "preprocessing_stats": stats,
            "splits": {},
        }

        for split_name, split_records in splits.items():
            # Sentiment task does not need summary field
            task_records = [
                {"text": r["text"], "label": r["label"], "summary": None}
                for r in split_records
            ]
            out_path = output_dir / f"{split_name}.jsonl"
            self._save_jsonl(task_records, out_path)
            metadata["splits"][split_name] = {
                "path": str(out_path.relative_to(PROJECT_ROOT)),
                "count": len(task_records),
            }

        self._save_metadata(metadata, output_dir / "metadata.json")
        return metadata

    def prepare_amazon_summarization(self) -> Dict[str, Any]:
        """Prepare Amazon Reviews for summarization (title as reference summary)."""
        records = self.load_amazon_reviews_raw()
        split_cfg = self.config["splits"]

        if self.config["amazon_reviews_2023"].get("require_title_for_summarization", True):
            records = [r for r in records if r.get("summary")]

        processed, stats = self.preprocessor.preprocess_records(
            records, require_label=False, require_summary=True
        )
        splits = self.preprocessor.split_records(
            processed,
            train_ratio=split_cfg["train_ratio"],
            val_ratio=split_cfg["val_ratio"],
            test_ratio=split_cfg["test_ratio"],
            random_seed=split_cfg["random_seed"],
        )

        output_dir = self._ensure_dirs("summarization", "amazon")
        metadata = {
            "dataset": "Amazon Reviews 2023",
            "task": "abstractive_summarization",
            "reference_summary": "review_title",
            "preprocessing_stats": stats,
            "splits": {},
        }

        for split_name, split_records in splits.items():
            task_records = [
                {"text": r["text"], "label": None, "summary": r["summary"]}
                for r in split_records
            ]
            out_path = output_dir / f"{split_name}.jsonl"
            self._save_jsonl(task_records, out_path)
            metadata["splits"][split_name] = {
                "path": str(out_path.relative_to(PROJECT_ROOT)),
                "count": len(task_records),
            }

        self._save_metadata(metadata, output_dir / "metadata.json")
        return metadata

  # ------------------------------------------------------------------
  # XSum
  # ------------------------------------------------------------------

    def load_xsum_raw(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load XSum with official train/validation/test splits."""
        from datasets import load_dataset

        cfg = self.config["xsum"]
        max_samples = cfg.get("max_samples")
        text_col = cfg["text_column"]
        summary_col = cfg["summary_column"]

        result: Dict[str, List[Dict[str, Any]]] = {}
        for split_name in ("train", "validation", "test"):
            dataset = load_dataset(cfg["hf_dataset"], split=split_name)
            if max_samples:
                per_split_limit = max(1, max_samples // 3)
                dataset = dataset.select(range(min(per_split_limit, len(dataset))))

            records = []
            for row in dataset:
                text = row.get(text_col)
                summary = row.get(summary_col)
                if text and summary:
                    records.append(
                        DatasetRecord(text=str(text), label=None, summary=str(summary)).to_dict()
                    )
            result[split_name] = records

        return result

    def prepare_xsum(self) -> Dict[str, Any]:
        """Prepare XSum benchmark for summarization."""
        raw_splits = self.load_xsum_raw()
        output_dir = self._ensure_dirs("summarization", "xsum")

        metadata = {
            "dataset": "XSum",
            "task": "abstractive_summarization",
            "splits": {},
        }

        for split_name, records in raw_splits.items():
            processed, stats = self.preprocessor.preprocess_records(
                records, require_label=False, require_summary=True
            )
            out_path = output_dir / f"{split_name}.jsonl"
            self._save_jsonl(processed, out_path)
            metadata["splits"][split_name] = {
                "path": str(out_path.relative_to(PROJECT_ROOT)),
                "stats": stats,
            }

        self._save_metadata(metadata, output_dir / "metadata.json")
        return metadata

  # ------------------------------------------------------------------
  # High-level API
  # ------------------------------------------------------------------

    def prepare_all(self) -> Dict[str, Any]:
        """Prepare all configured datasets."""
        results = {}

        if self.config["isarcasmeval"].get("enabled", True):
            results["sarcasm"] = self.prepare_isarcasmeval()

        if self.config["amazon_reviews_2023"].get("enabled", True):
            results["sentiment"] = self.prepare_amazon_sentiment()
            results["summarization_amazon"] = self.prepare_amazon_summarization()

        if self.config["xsum"].get("enabled", True):
            results["summarization_xsum"] = self.prepare_xsum()

        summary_path = self.processed_dir / "preparation_summary.json"
        self._save_metadata(results, summary_path)
        return results

    def load_task_split(
        self,
        task: str,
        split: str = "test",
        source: Optional[str] = None,
    ) -> List[Dict[str, Optional[str]]]:
        """
        Load a prepared split for evaluation.

        Args:
            task: sarcasm | sentiment | summarization
            split: train | validation | test
            source: amazon | xsum (only for summarization)
        """
        if task == "sarcasm":
            path = self.processed_dir / "sarcasm" / f"{split}.jsonl"
        elif task == "sentiment":
            path = self.processed_dir / "sentiment" / f"{split}.jsonl"
        elif task == "summarization":
            subdir = source or "amazon"
            path = self.processed_dir / "summarization" / subdir / f"{split}.jsonl"
        else:
            raise ValueError(f"Unknown task: {task}")

        if not path.exists():
            raise FileNotFoundError(
                f"Prepared split not found: {path}. Run scripts/prepare_datasets.py first."
            )
        return self.load_jsonl(path)

    def get_texts_and_labels(
        self, task: str, split: str = "test", source: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """Return (texts, labels) for classification tasks."""
        records = self.load_task_split(task, split=split, source=source)
        texts = [r["text"] for r in records]
        labels = [r["label"] for r in records if r.get("label")]
        if len(labels) != len(texts):
            raise ValueError("Some records are missing labels.")
        return texts, labels

    def get_texts_and_summaries(
        self, split: str = "test", source: str = "amazon"
    ) -> Tuple[List[str], List[str]]:
        """Return (texts, reference_summaries) for summarization."""
        records = self.load_task_split("summarization", split=split, source=source)
        texts = [r["text"] for r in records]
        summaries = [r["summary"] for r in records if r.get("summary")]
        if len(summaries) != len(texts):
            raise ValueError("Some records are missing reference summaries.")
        return texts, summaries
