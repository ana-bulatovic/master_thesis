#!/usr/bin/env python3
"""
Fine-tune a transformer model for sarcasm detection on iSarcasmEval.

Usage:
  python scripts/train_sarcasm.py
  python scripts/train_sarcasm.py --epochs 5 --batch-size 32
  python scripts/train_sarcasm.py --model-name bert-base-uncased --output-dir models/sarcasm_bert
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from path_setup import PROJECT_ROOT

DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "sarcasm" / "train.jsonl"
DEFAULT_TEST = PROJECT_ROOT / "data" / "processed" / "sarcasm" / "test.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "sarcasm_distilbert"
LABEL2ID = {"non-sarcastic": 0, "sarcastic": 1}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("text") and record.get("label") in LABEL2ID:
                records.append(record)
    return records


def build_dataset(records: List[Dict[str, str]]) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [record["text"] for record in records],
            "label": [LABEL2ID[record["label"]] for record in records],
        }
    )


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune sarcasm detection model")
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="Hugging Face base model",
    )
    parser.add_argument(
        "--train-file",
        default=str(DEFAULT_TRAIN),
        help="Path to train.jsonl",
    )
    parser.add_argument(
        "--test-file",
        default=str(DEFAULT_TEST),
        help="Path to test.jsonl (used for validation during training)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Where to save the fine-tuned model",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable mixed precision even if GPU is available",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    train_path = Path(args.train_file)
    test_path = Path(args.test_file)
    output_dir = Path(args.output_dir)
    final_dir = output_dir / "final"

    if not train_path.exists():
        print(f"Train file not found: {train_path}")
        print("Run first: python scripts/prepare_datasets.py --dataset sarcasm")
        return 1

    train_records = load_jsonl(train_path)
    if not train_records:
        print(f"No training records found in {train_path}")
        return 1

    eval_records = load_jsonl(test_path) if test_path.exists() else []
    print(f"Train samples: {len(train_records)}")
    print(f"Eval samples:  {len(eval_records)}")
    print(f"Base model:    {args.model_name}")
    print(f"Output dir:    {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    train_dataset = build_dataset(train_records)
    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    eval_dataset = None
    if eval_records:
        eval_dataset = build_dataset(eval_records)
        eval_dataset = eval_dataset.map(tokenize_batch, batched=True)
        eval_dataset = eval_dataset.rename_column("label", "labels")
        eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    use_fp16 = not args.no_fp16
    try:
        import torch

        if not torch.cuda.is_available():
            use_fp16 = False
    except ImportError:
        use_fp16 = False

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch" if eval_dataset is not None else "no",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to=[],
        fp16=use_fp16,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )

    print("\nStarting training...")
    trainer.train()

    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metadata = {
        "task": "sarcasm_detection",
        "approach": "fine-tuned",
        "base_model": args.model_name,
        "train_file": str(train_path.relative_to(PROJECT_ROOT)),
        "test_file": str(test_path.relative_to(PROJECT_ROOT)) if test_path.exists() else None,
        "train_samples": len(train_records),
        "eval_samples": len(eval_records),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "labels": list(LABEL2ID.keys()),
        "trained_at": datetime.now().isoformat(),
        "model_dir": str(final_dir.relative_to(PROJECT_ROOT)),
    }

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        metadata["eval_metrics"] = {
            key: float(value) for key, value in eval_metrics.items() if isinstance(value, (int, float))
        }
        print("\nValidation metrics:")
        for key, value in metadata["eval_metrics"].items():
            if key.startswith("eval_"):
                print(f"  {key}: {value:.4f}")

    metadata_path = final_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nModel saved to: {final_dir}")
    print(f"Metadata:       {metadata_path}")
    print("\nNext step:")
    print("  python scripts/evaluate_finetuned_sarcasm.py")
    print("  python scripts/compare_sarcasm_approaches.py --limit 50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
