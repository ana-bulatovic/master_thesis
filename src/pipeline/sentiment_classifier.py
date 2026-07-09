from typing import Dict, List, Optional

from llm.ollama_client import OllamaClient


class SentimentClassifier:
    """Classify sentiment of text using LLM"""

    def __init__(self, client: OllamaClient, prompting_technique: str = "few-shot"):
        self.client = client
        self.prompting_technique = prompting_technique
        self.sentiment_labels = ["positive", "negative", "neutral"]

    def _build_sarcasm_hint(self, sarcasm: Optional[bool]) -> str:
        if sarcasm is None:
            return ""

        if sarcasm:
            return (
                "The review is sarcastic. The literal wording may not reflect the true sentiment. "
                "Classify the intended/true sentiment.\n"
            )

        return "The review is not sarcastic.\n"

    def _get_zero_shot_prompt(self, text: str, sarcasm: Optional[bool] = None) -> str:
        sarcasm_hint = self._build_sarcasm_hint(sarcasm)

        return f"""Analyze the sentiment of the following review.

{sarcasm_hint}Review: {text}

Classify the sentiment as one of: positive, negative, neutral.
Answer with only the sentiment label."""

    def _get_few_shot_prompt(self, text: str, sarcasm: Optional[bool] = None) -> str:
        examples = [
            ("This product is amazing and works perfectly!", "positive"),
            ("I'm very happy with this purchase, excellent quality.", "positive"),
            ("Terrible product, stopped working immediately.", "negative"),
            ("Waste of money, very disappointed.", "negative"),
            ("The product is average, nothing special.", "neutral"),
            ("It's okay, does what it's supposed to.", "neutral"),
            (
                "Oh wonderful, another premium product that broke in two days. Just perfect.",
                "negative",
            ),
        ]

        sarcasm_hint = self._build_sarcasm_hint(sarcasm)

        prompt = (
            "Classify the sentiment of reviews as: positive, negative, or neutral.\n"
            "For sarcastic reviews, classify the intended sentiment, not the literal words.\n\n"
            f"{sarcasm_hint}Examples:\n"
        )
        for example_text, sentiment in examples:
            prompt += f"Review: {example_text}\nSentiment: {sentiment}\n\n"

        prompt += f"Now classify this review:\n\nReview: {text}\n\nSentiment:"
        return prompt

    def _get_chain_of_thought_prompt(self, text: str, sarcasm: Optional[bool] = None) -> str:
        sarcasm_hint = self._build_sarcasm_hint(sarcasm)

        return f"""Analyze the sentiment of this review step by step.

{sarcasm_hint}Review: {text}

Step 1: Check whether the review might be sarcastic.
Step 2: Identify positive and negative signals (literal and intended).
Step 3: Determine the true overall tone.
Step 4: Classify as positive, negative, or neutral.

Final sentiment:"""

    def classify(
        self,
        text: str,
        model: str = None,
        sarcasm: Optional[bool] = None,
        sarcasm_aware: bool = False,
    ) -> Dict[str, object]:
        sarcasm_hint = sarcasm if sarcasm_aware else None

        if self.prompting_technique == "zero-shot":
            prompt = self._get_zero_shot_prompt(text, sarcasm_hint)
        elif self.prompting_technique == "chain-of-thought":
            prompt = self._get_chain_of_thought_prompt(text, sarcasm_hint)
        else:
            prompt = self._get_few_shot_prompt(text, sarcasm_hint)

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
            "sarcasm": sarcasm if sarcasm_aware else None,
            "sarcasm_aware": sarcasm_aware and sarcasm is not None,
            "response": response.strip(),
            "technique": self.prompting_technique,
        }

    def classify_batch(self, texts: List[str], model: str = None) -> List[Dict]:
        return [self.classify(text, model) for text in texts]
