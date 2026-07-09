"""Fine-tuned transformer classifier for sarcasm detection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FinetunedSarcasmClassifier:
    """Load and run a fine-tuned sequence classification model for sarcasm."""

    def __init__(self, model_path: str | Path, device: Optional[str] = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Fine-tuned model not found: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        config_labels = self.model.config.label2id

        self.label2id = {
            key: int(value) for key, value in config_labels.items()
        }

        self.id2label = {
            int(value): key for key, value in config_labels.items()
        }

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, max_length: int = 128) -> Dict[str, object]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
            predicted_id = int(torch.argmax(probabilities).item())

        label = self.id2label[predicted_id]
        confidence = float(probabilities[predicted_id].item())

        return {
            "is_sarcastic": label == "sarcastic",
            "label": label,
            "confidence": confidence,
            "response": label,
            "approach": "fine-tuned",
            "model_path": str(self.model_path),
        }

    def predict_batch(self, texts: List[str], max_length: int = 128) -> List[Dict[str, object]]:
        return [self.predict(text, max_length=max_length) for text in texts]
