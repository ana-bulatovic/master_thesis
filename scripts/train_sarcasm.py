#!/usr/bin/env python3
"""
Fine-tune a transformer model for sarcasm detection on iSarcasmEval.

Improvements over basic training:
  - class weights for imbalanced labels (sarcastic vs non-sarcastic)
  - validation split from train (test set stays untouched)
  - macro F1 and per-class metrics (focus on sarcastic recall)

Usage:
  python scripts/train_sarcasm.py
  python scripts/train_sarcasm.py --epochs 5 --batch-size 16
  python scripts/train_sarcasm.py --model-name roberta-base --output-dir models/sarcasm_roberta
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
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
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "sarcasm_bertweet"
DEFAULT_MODEL = "vinai/bertweet-base"
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


def build_dataset(texts: List[str], labels: List[int]) -> Dataset:
    return Dataset.from_dict({"text": texts, "label": labels})


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [
        total / (len(counts) * counts[class_id])
        for class_id in sorted(counts.keys())
    ]
    return torch.tensor(weights, dtype=torch.float)


def split_train_validation(
    records: List[Dict[str, str]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    texts = [record["text"] for record in records]
    labels = [LABEL2ID[record["label"]] for record in records]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )

    train_records = [
        {"text": text, "label": ID2LABEL[label]}
        for text, label in zip(train_texts, train_labels)
    ]
    val_records = [
        {"text": text, "label": ID2LABEL[label]}
        for text, label in zip(val_texts, val_labels)
    ]
    return train_records, val_records


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    sarcastic_recall = recall_score(
        labels,
        predictions,
        labels=[LABEL2ID["non-sarcastic"], LABEL2ID["sarcastic"]],
        average="binary",
        pos_label=LABEL2ID["sarcastic"],
        zero_division=0,
    )
    sarcastic_precision = precision_score(
        labels,
        predictions,
        labels=[LABEL2ID["non-sarcastic"], LABEL2ID["sarcastic"]],
        average="binary",
        pos_label=LABEL2ID["sarcastic"],
        zero_division=0,
    )
    sarcastic_f1 = f1_score(
        labels,
        predictions,
        labels=[LABEL2ID["non-sarcastic"], LABEL2ID["sarcastic"]],
        average="binary",
        pos_label=LABEL2ID["sarcastic"],
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_weighted": precision_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision_sarcastic": sarcastic_precision,
        "recall_sarcastic": sarcastic_recall,
        "f1_sarcastic": sarcastic_f1,
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune sarcasm detection model")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Hugging Face base model (default: vinai/bertweet-base)",
    )
    parser.add_argument(
        "--train-file",
        default=str(DEFAULT_TRAIN),
        help="Path to train.jsonl",
    )
    parser.add_argument(
        "--test-file",
        default=str(DEFAULT_TEST),
        help="Path to test.jsonl (held out; optional post-training eval only)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Where to save the fine-tuned model",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weights for imbalanced data",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable mixed precision even if GPU is available",
    )
    return parser.parse_args()


def prepare_dataset(records: List[Dict[str, str]], tokenizer, max_length: int) -> Dataset:
    dataset = build_dataset(
        [record["text"] for record in records],
        [LABEL2ID[record["label"]] for record in records],
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    dataset = dataset.map(tokenize_batch, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def print_label_distribution(records: List[Dict[str, str]], title: str) -> None:
    counts = Counter(record["label"] for record in records)
    print(f"  {title}: {dict(counts)} (total {len(records)})")


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

    all_train_records = load_jsonl(train_path)
    if not all_train_records:
        print(f"No training records found in {train_path}")
        return 1

    train_records, val_records = split_train_validation(
        all_train_records,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    test_records = load_jsonl(test_path) if test_path.exists() else []

    print("Dataset summary:")
    print_label_distribution(train_records, "Train")
    print_label_distribution(val_records, "Validation")
    if test_records:
        print_label_distribution(test_records, "Test (held out)")
    print(f"Base model:    {args.model_name}")
    print(f"Output dir:    {output_dir}")
    print(f"Class weights: {'off' if args.no_class_weights else 'on'}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = prepare_dataset(train_records, tokenizer, args.max_length)
    val_dataset = prepare_dataset(val_records, tokenizer, args.max_length)

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(
            [LABEL2ID[record["label"]] for record in train_records]
        )
        print(f"Class weights: non-sarcastic={class_weights[0]:.3f}, sarcastic={class_weights[1]:.3f}")

    use_fp16 = not args.no_fp16
    if not torch.cuda.is_available():
        use_fp16 = False

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_sarcastic",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to=[],
        fp16=use_fp16,
        seed=args.seed,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    val_metrics = trainer.evaluate()
    metadata = {
        "task": "sarcasm_detection",
        "approach": "fine-tuned",
        "base_model": args.model_name,
        "train_file": str(train_path.relative_to(PROJECT_ROOT)),
        "test_file": str(test_path.relative_to(PROJECT_ROOT)) if test_path.exists() else None,
        "train_samples": len(train_records),
        "validation_samples": len(val_records),
        "test_samples": len(test_records),
        "val_ratio": args.val_ratio,
        "class_weights_enabled": not args.no_class_weights,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "labels": list(LABEL2ID.keys()),
        "trained_at": datetime.now().isoformat(),
        "model_dir": str(final_dir.relative_to(PROJECT_ROOT)),
        "validation_metrics": {
            key.replace("eval_", ""): float(value)
            for key, value in val_metrics.items()
            if key.startswith("eval_") and isinstance(value, (int, float))
        },
    }

    print("\nValidation metrics:")
    for key, value in metadata["validation_metrics"].items():
        print(f"  {key}: {value:.4f}")

    if test_records:
        test_dataset = prepare_dataset(test_records, tokenizer, args.max_length)
        test_metrics_raw = trainer.evaluate(test_dataset)
        metadata["test_metrics"] = {
            key.replace("eval_", ""): float(value)
            for key, value in test_metrics_raw.items()
            if key.startswith("eval_") and isinstance(value, (int, float))
        }
        print("\nHeld-out test metrics:")
        for key, value in metadata["test_metrics"].items():
            print(f"  {key}: {value:.4f}")

    metadata_path = final_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nModel saved to: {final_dir}")
    print(f"Metadata:       {metadata_path}")
    print("\nNext steps:")
    print("  python scripts/evaluate_finetuned_sarcasm.py --model-dir", final_dir)
    print("  python scripts/compare_sarcasm_approaches.py --limit 50")
    print("\nUpdate config/config.yaml:")
    print(f"  sarcasm_finetuned_path: {final_dir.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
