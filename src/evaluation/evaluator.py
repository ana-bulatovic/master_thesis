from typing import List, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")


class Evaluator:
    """Evaluate classification and summarization tasks"""

    @staticmethod
    def evaluate_classification(
        predictions: List[str],
        references: List[str],
        labels: List[str] = None,
    ) -> Dict[str, float]:
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have same length")

        predictions = [str(prediction).lower().strip() for prediction in predictions]
        references = [str(reference).lower().strip() for reference in references]

        try:
            accuracy = accuracy_score(references, predictions)

            if labels:
                precision = precision_score(
                    references, predictions, labels=labels, average="weighted", zero_division=0
                )
                recall = recall_score(
                    references, predictions, labels=labels, average="weighted", zero_division=0
                )
                f1 = f1_score(
                    references, predictions, labels=labels, average="weighted", zero_division=0
                )
            else:
                precision = precision_score(
                    references, predictions, average="weighted", zero_division=0
                )
                recall = recall_score(
                    references, predictions, average="weighted", zero_division=0
                )
                f1 = f1_score(references, predictions, average="weighted", zero_division=0)

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        except Exception as exc:
            print(f"Error computing classification metrics: {exc}")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    @staticmethod
    def compute_rouge(prediction: str, reference: str, use_stemmer: bool = False) -> Dict[str, float]:
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer
            )
            scores = scorer.score(reference, prediction)

            return {
                "rouge1_f": scores["rouge1"].fmeasure,
                "rouge2_f": scores["rouge2"].fmeasure,
                "rougeL_f": scores["rougeL"].fmeasure,
            }
        except Exception as exc:
            print(f"Error computing ROUGE: {exc}")
            return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}

    @staticmethod
    def compute_bert_score(
        prediction: str, reference: str, model_type: str = "distilbert-base-uncased"
    ) -> Dict[str, float]:
        try:
            from bert_score import score

            precision, recall, f1 = score(
                [prediction], [reference], model_type=model_type, lang="en", verbose=False
            )

            return {
                "bert_precision": precision.item(),
                "bert_recall": recall.item(),
                "bert_f1": f1.item(),
            }
        except Exception as exc:
            print(f"Error computing BERTScore: {exc}")
            return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}

    @staticmethod
    def evaluate_summarization(
        predictions: List[str],
        references: List[str],
        use_rouge: bool = True,
        use_bertscore: bool = False,
    ) -> Dict[str, object]:
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have same length")

        results = {
            "num_samples": len(predictions),
            "rouge": {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0},
            "bertscore": {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0},
        }

        if use_rouge:
            rouge_scores = {"rouge1_f": [], "rouge2_f": [], "rougeL_f": []}
            for prediction, reference in zip(predictions, references):
                scores = Evaluator.compute_rouge(prediction, reference)
                for key, value in scores.items():
                    rouge_scores[key].append(value)

            results["rouge"] = {
                key: sum(values) / len(values) if values else 0.0
                for key, values in rouge_scores.items()
            }

        if use_bertscore:
            bertscore_scores = {
                "bert_precision": [],
                "bert_recall": [],
                "bert_f1": [],
            }
            for prediction, reference in zip(predictions, references):
                scores = Evaluator.compute_bert_score(prediction, reference)
                for key, value in scores.items():
                    bertscore_scores[key].append(value)

            results["bertscore"] = {
                key: sum(values) / len(values) if values else 0.0
                for key, values in bertscore_scores.items()
            }

        return results
