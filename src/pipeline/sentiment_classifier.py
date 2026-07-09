from typing import Dict, List

from llm.ollama_client import OllamaClient


class SentimentClassifier:
    """Classify sentiment of text using LLM"""

    def __init__(self, client: OllamaClient, prompting_technique: str = "few-shot"):
        self.client = client
        self.prompting_technique = prompting_technique
        self.sentiment_labels = ["positive", "negative", "neutral"]

    def _get_zero_shot_prompt(self, text: str) -> str:
        return f"""Analyze the sentiment of the following review.

Review: {text}

Classify the sentiment as one of: positive, negative, neutral.
Answer with only the sentiment label."""

    def _get_few_shot_prompt(self, text: str) -> str:
        examples = [
            ("This product is amazing and works perfectly!", "positive"),
            ("I'm very happy with this purchase, excellent quality.", "positive"),
            ("Terrible product, stopped working immediately.", "negative"),
            ("Waste of money, very disappointed.", "negative"),
            ("The product is average, nothing special.", "neutral"),
            ("It's okay, does what it's supposed to.", "neutral"),
        ]

        prompt = "Classify the sentiment of reviews as: positive, negative, or neutral.\n\nExamples:\n"
        for example_text, sentiment in examples:
            prompt += f"Review: {example_text}\nSentiment: {sentiment}\n\n"

        prompt += f"Now classify this review:\n\nReview: {text}\n\nSentiment:"
        return prompt

    def _get_chain_of_thought_prompt(self, text: str) -> str:
        return f"""Analyze the sentiment of this review step by step.

Review: {text}

Step 1: Identify positive words/phrases.
Step 2: Identify negative words/phrases.
Step 3: Determine overall tone.
Step 4: Classify as positive, negative, or neutral.

Final sentiment:"""

    def classify(self, text: str, model: str = None) -> Dict[str, object]:
        if self.prompting_technique == "zero-shot":
            prompt = self._get_zero_shot_prompt(text)
        elif self.prompting_technique == "chain-of-thought":
            prompt = self._get_chain_of_thought_prompt(text)
        else:
            prompt = self._get_few_shot_prompt(text)

        response = self.client.generate(prompt, model)
        response_lower = response.lower().strip()
        sentiment = "neutral"

        for label in self.sentiment_labels:
            if label in response_lower:
                sentiment = label
                break

        return {
            "text": text,
            "sentiment": sentiment,
            "response": response.strip(),
            "technique": self.prompting_technique,
        }

    def classify_batch(self, texts: List[str], model: str = None) -> List[Dict]:
        return [self.classify(text, model) for text in texts]
