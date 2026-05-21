from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

class Evaluator:
    """Evaluate classification and summarization tasks"""
    
    @staticmethod
    def evaluate_classification(predictions: List[str], references: List[str], 
                               labels: List[str] = None) -> Dict[str, float]:
        """
        Evaluate classification task using standard metrics
        
        Args:
            predictions: List of predicted labels
            references: List of true labels
            labels: List of label names (for averaging)
        
        Returns:
            Dictionary with metrics (accuracy, precision, recall, f1)
        """
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have same length")
        
        # Convert to strings for comparison
        predictions = [str(p).lower().strip() for p in predictions]
        references = [str(r).lower().strip() for r in references]
        
        try:
            accuracy = accuracy_score(references, predictions)
            
            if labels:
                precision = precision_score(references, predictions, labels=labels, 
                                           average='weighted', zero_division=0)
                recall = recall_score(references, predictions, labels=labels, 
                                    average='weighted', zero_division=0)
                f1 = f1_score(references, predictions, labels=labels, 
                            average='weighted', zero_division=0)
            else:
                precision = precision_score(references, predictions, 
                                           average='weighted', zero_division=0)
                recall = recall_score(references, predictions, 
                                    average='weighted', zero_division=0)
                f1 = f1_score(references, predictions, 
                            average='weighted', zero_division=0)
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        except Exception as e:
            print(f"Error computing classification metrics: {e}")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    @staticmethod
    def compute_rouge(prediction: str, reference: str, use_stemmer: bool = False) -> Dict[str, float]:
        """
        Compute ROUGE scores for summarization
        
        Args:
            prediction: Generated summary
            reference: Reference summary
            use_stemmer: Whether to use stemmer
        
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                             use_stemmer=use_stemmer)
            scores = scorer.score(reference, prediction)
            
            return {
                "rouge1_f": scores['rouge1'].fmeasure,
                "rouge2_f": scores['rouge2'].fmeasure,
                "rougeL_f": scores['rougeL'].fmeasure,
            }
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    
    @staticmethod
    def compute_bert_score(prediction: str, reference: str, 
                          model_type: str = "distilbert-base-uncased") -> Dict[str, float]:
        """
        Compute BERTScore for summarization
        
        Args:
            prediction: Generated summary
            reference: Reference summary
            model_type: BERT model to use
        
        Returns:
            Dictionary with BERTScore values (precision, recall, f1)
        """
        try:
            from bert_score import score
            
            P, R, F1 = score([prediction], [reference], 
                            model_type=model_type, lang="en", verbose=False)
            
            return {
                "bert_precision": P.item(),
                "bert_recall": R.item(),
                "bert_f1": F1.item(),
            }
        except Exception as e:
            print(f"Error computing BERTScore: {e}")
            return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}
    
    @staticmethod
    def evaluate_summarization(predictions: List[str], references: List[str],
                              use_rouge: bool = True, use_bertscore: bool = False) -> Dict[str, any]:
        """
        Evaluate summarization task using ROUGE and/or BERTScore
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            use_rouge: Whether to compute ROUGE scores
            use_bertscore: Whether to compute BERTScore
        
        Returns:
            Dictionary with average metric scores
        """
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have same length")
        
        results = {
            "num_samples": len(predictions),
            "rouge": {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0},
            "bertscore": {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}
        }
        
        if use_rouge:
            rouge_scores = {
                "rouge1_f": [],
                "rouge2_f": [],
                "rougeL_f": [],
            }
            
            for pred, ref in zip(predictions, references):
                scores = Evaluator.compute_rouge(pred, ref)
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
            
            for pred, ref in zip(predictions, references):
                scores = Evaluator.compute_bert_score(pred, ref)
                for key, value in scores.items():
                    bertscore_scores[key].append(value)
            
            results["bertscore"] = {
                key: sum(values) / len(values) if values else 0.0 
                for key, values in bertscore_scores.items()
            }
        
        return results
