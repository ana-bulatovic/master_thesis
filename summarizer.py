from typing import Dict, List, Optional
from ollama_client import OllamaClient

class TextSummarizer:
    """Generate abstractive summaries using LLM"""
    
    def __init__(self, client: OllamaClient, prompting_technique: str = "few-shot", 
                 use_sentiment_info: bool = False):
        self.client = client
        self.prompting_technique = prompting_technique
        self.use_sentiment_info = use_sentiment_info
    
    def _get_zero_shot_prompt(self, text: str, sentiment: Optional[str] = None) -> str:
        """Generate zero-shot summarization prompt"""
        prompt = f"""Summarize the following review in ONE short sentence.

ONLY use information explicitly mentioned in the review.
Do not add new details.
Do not invent product features.

Review:
{text}

Summary:
"""
        
        if self.use_sentiment_info and sentiment:
            prompt = f"""Summarize the following review in ONE short sentence.

ONLY use information explicitly mentioned in the review.
Do not add new details.
Do not invent product features.
The review expresses a {sentiment} sentiment.

Review:
{text}

Summary:
"""
        
        return prompt
    
    def _get_few_shot_prompt(self, text: str, sentiment: Optional[str] = None) -> str:
        """Generate few-shot summarization prompt with examples"""
        examples = [
            (
                "This smartphone has excellent camera quality and long battery life, but it's quite expensive.",
                "High-end smartphone with great camera and battery, though pricey."
            ),
            (
                "The laptop is very slow and keeps freezing. I regret buying it.",
                "Sluggish laptop with freezing issues - a disappointing purchase."
            ),
        ]
        
        prompt = "Summarize the following review in ONE short sentence.\n\nONLY use information explicitly mentioned in the review.\nDo not add new details.\nDo not invent product features.\n\nExamples:\n"
        
        for example_text, summary in examples:
            prompt += f"Review: {example_text}\nSummary: {summary}\n\n"
        
        prompt += f"Now summarize this review:\n\nReview: {text}\n\nSummary:"
        
        if self.use_sentiment_info and sentiment:
            prompt = f"Summarize the following review in ONE short sentence.\n\nONLY use information explicitly mentioned in the review.\nDo not add new details.\nDo not invent product features.\nThe review expresses a {sentiment} sentiment.\n\nExamples:\n"
            for example_text, summary in examples:
                prompt += f"Review: {example_text}\nSummary: {summary}\n\n"
            prompt += f"Now summarize this review:\n\nReview: {text}\n\nSummary:"
        
        return prompt
    
    def _get_chain_of_thought_prompt(self, text: str, sentiment: Optional[str] = None) -> str:
        """Generate chain-of-thought summarization prompt"""
        prompt = f"""Summarize the following review in ONE short sentence.

ONLY use information explicitly mentioned in the review.
Do not add new details.
Do not invent product features.

Review: {text}

Step 1: Identify the main product/service being reviewed.
Step 2: List key points (positive, negative, or neutral).
Step 3: Generate a concise summary combining key points.

Summary:"""
        
        if self.use_sentiment_info and sentiment:
            prompt = f"""Summarize the following review in ONE short sentence.

ONLY use information explicitly mentioned in the review.
Do not add new details.
Do not invent product features.
The review expresses a {sentiment} sentiment.

Review: {text}

Step 1: Identify the main product/service being reviewed.
Step 2: List key points (positive, negative, or neutral).
Step 3: Generate a concise summary combining key points.

Summary:"""
        
        return prompt
    
    def summarize(self, text: str, sentiment: Optional[str] = None, model: str = None) -> Dict[str, any]:
        """
        Summarize text
        
        Args:
            text: Text to summarize
            sentiment: Optional sentiment classification for sentiment-aware summarization
            model: Model name
        
        Returns:
            Dictionary with summary
        """
        # Select prompt based on technique
        if self.prompting_technique == "zero-shot":
            prompt = self._get_zero_shot_prompt(text, sentiment)
        elif self.prompting_technique == "chain-of-thought":
            prompt = self._get_chain_of_thought_prompt(text, sentiment)
        else:  # few-shot (default)
            prompt = self._get_few_shot_prompt(text, sentiment)
        
        response = self.client.generate(prompt, model)
        
        return {
            "original_text": text,
            "summary": response.strip(),
            "sentiment": sentiment,
            "sentiment_aware": self.use_sentiment_info and sentiment is not None,
            "technique": self.prompting_technique
        }
    
    def summarize_batch(self, texts: List[str], sentiments: List[Optional[str]] = None, 
                       model: str = None) -> List[Dict]:
        """
        Summarize multiple texts
        
        Args:
            texts: List of texts to summarize
            sentiments: Optional list of sentiments (same length as texts)
            model: Model name
        
        Returns:
            List of summaries
        """
        if sentiments is None:
            sentiments = [None] * len(texts)
        
        results = []
        for text, sentiment in zip(texts, sentiments):
            result = self.summarize(text, sentiment, model)
            results.append(result)
        
        return results
