import re
from typing import Dict, List, Tuple
from ollama_client import OllamaClient
from config import ModelConfig

class SarcasmDetector:
    """Detect sarcasm in text using LLM"""
    
    def __init__(self, client: OllamaClient, prompting_technique: str = "few-shot"):
        self.client = client
        self.prompting_technique = prompting_technique
    
    def _get_zero_shot_prompt(self, text: str) -> str:
        """Generate zero-shot prompt for sarcasm detection"""
        return f"""You are a sarcasm detection expert. Analyze the following review and determine if it contains sarcasm.

Only answer with exactly Yes or No.
If the review expresses the opposite meaning from what it literally says, then it is sarcastic.

Review: {text}

Answer with only Yes or No."""
    
    def _get_few_shot_prompt(self, text: str) -> str:
        """Generate few-shot prompt with examples"""
        examples = [
            ("This product is absolutely amazing, it broke after one day!", "Yes"),
            ("I loved this phone, best purchase ever made!", "No"),
            ("Great quality, just like paying for garbage!", "Yes"),
            ("Perfect fit and flawless performance, except it stopped working immediately.", "Yes"),
            ("This is the best phone I've ever owned, if only it would turn on.", "Yes"),
            ("The service was great and the food was delicious.", "No"),
        ]
        
        prompt = "You are a sarcasm detection expert. Here are some examples:\n\n"
        
        for example_text, label in examples:
            prompt += f"Review: {example_text}\nSarcastic: {label}\n\n"
        
        prompt += f"Now analyze this review:\n\nReview: {text}\n\nAnswer with only Yes or No. Is this review sarcastic?"
        
        return prompt
    
    def _get_chain_of_thought_prompt(self, text: str) -> str:
        """Generate chain-of-thought prompt"""
        return f"""You are a sarcasm detection expert. Analyze the following review step by step.

Review: {text}

Step 1: Identify the literal meaning of the review.
Step 2: Identify the actual sentiment or intention.
Step 3: Compare literal vs actual meaning - is there a mismatch indicating sarcasm?
Step 4: Conclude.

Final answer (Yes/No): """
    
    def detect(self, text: str, model: str = None) -> Dict[str, any]:
        """
        Detect sarcasm in text
        
        Args:
            text: Text to analyze
            model: Model name (uses client default if None)
        
        Returns:
            Dictionary with prediction and confidence
        """
        # Select prompt based on technique
        if self.prompting_technique == "zero-shot":
            prompt = self._get_zero_shot_prompt(text)
        elif self.prompting_technique == "chain-of-thought":
            prompt = self._get_chain_of_thought_prompt(text)
        else:  # few-shot (default)
            prompt = self._get_few_shot_prompt(text)
        
        response = self.client.generate(prompt, model)
        
        # Parse response robustly for Yes/No answers
        response_lower = response.lower().strip()
        is_sarcastic = False
        yes_match = re.search(r"\byes\b", response_lower)
        no_match = re.search(r"\bno\b", response_lower)

        if yes_match and not no_match:
            is_sarcastic = True
        elif no_match and not yes_match:
            is_sarcastic = False
        elif response_lower.startswith("yes"):
            is_sarcastic = True
        elif response_lower.startswith("no"):
            is_sarcastic = False

        return {
            "text": text,
            "is_sarcastic": is_sarcastic,
            "response": response.strip(),
            "technique": self.prompting_technique
        }
    
    def detect_batch(self, texts: List[str], model: str = None) -> List[Dict]:
        """
        Detect sarcasm in multiple texts
        
        Args:
            texts: List of texts to analyze
            model: Model name
        
        Returns:
            List of detection results
        """
        results = []
        for text in texts:
            result = self.detect(text, model)
            results.append(result)
        
        return results
