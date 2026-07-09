from typing import Dict, List, Optional

from llm.ollama_client import OllamaClient


class SentimentClassifier:
    """Classify sentiment of text using LLM"""

    def __init__(self, client: OllamaClient, prompting_technique: str = "few-shot"):
        self.client = client
        self.prompting_technique = prompting_technique
        self.sentiment_labels = ["positive", "negative", "neutral"]

    def _format_confidence(self, confidence: Optional[float]) -> str:
        if confidence is None:
            return ""
        return f" (confidence: {confidence:.0%})"

    def _build_sarcasm_hint(
        self,
        sarcasm: Optional[bool],
        sarcasm_aware: bool = False,
        sarcasm_confidence: Optional[float] = None,
    ) -> str:
        if not sarcasm_aware or sarcasm is None:
            return ""

        confidence_text = self._format_confidence(sarcasm_confidence)

        if sarcasm:
            return (
                f"Sarcasm detector result: SARCASTIC{confidence_text}. "
                "The literal wording may not reflect the true sentiment. "
                "Classify the intended/true sentiment.\n"
            )

        return (
            f"Sarcasm detector result: NON-SARCASTIC{confidence_text}. "
            "Classify the literal sentiment of the review.\n"
        )

    def _get_zero_shot_prompt(
        self,
        text: str,
        sarcasm: Optional[bool] = None,
        sarcasm_aware: bool = False,
        sarcasm_confidence: Optional[float] = None,
    ) -> str:
        sarcasm_hint = self._build_sarcasm_hint(sarcasm, sarcasm_aware, sarcasm_confidence)

        return f"""Analyze the sentiment of the following review.

{sarcasm_hint}Review: {text}

Classify the sentiment as one of: positive, negative, neutral.
Answer with only the sentiment label."""

    def _get_few_shot_prompt(
        self,
        text: str,
        sarcasm: Optional[bool] = None,
        sarcasm_aware: bool = False,
        sarcasm_confidence: Optional[float] = None,
    ) -> str:
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

        sarcasm_hint = self._build_sarcasm_hint(sarcasm, sarcasm_aware, sarcasm_confidence)

        prompt = "Classify the sentiment of reviews as: positive, negative, or neutral.\n"
        if sarcasm_hint:
            prompt += f"\n{sarcasm_hint}"
        prompt += "Examples:\n"
        for example_text, sentiment in examples:
            prompt += f"Review: {example_text}\nSentiment: {sentiment}\n\n"

        prompt += f"Now classify this review:\n\nReview: {text}\n\nSentiment:"
        return prompt

    def _get_chain_of_thought_prompt(
        self,
        text: str,
        sarcasm: Optional[bool] = None,
        sarcasm_aware: bool = False,
        sarcasm_confidence: Optional[float] = None,
    ) -> str:
        sarcasm_hint = self._build_sarcasm_hint(sarcasm, sarcasm_aware, sarcasm_confidence)

        if sarcasm_aware and sarcasm is not None:
            step1 = "Step 1: Use the sarcasm detector result above."
            if sarcasm:
                step2 = "Step 2: Identify positive and negative signals, focusing on intended meaning."
            else:
                step2 = "Step 2: Identify positive and negative signals in the literal wording."
        else:
            step1 = "Step 1: Check whether the review might be sarcastic."
            step2 = "Step 2: Identify positive and negative signals (literal and intended)."

        return f"""Analyze the sentiment of this review step by step.

{sarcasm_hint}Review: {text}

{step1}
{step2}
Step 3: Determine the overall tone.
Step 4: Classify as positive, negative, or neutral.

Final sentiment:"""

    def classify(
        self,
        text: str,
        model: str = None,
        sarcasm: Optional[bool] = None,
        sarcasm_aware: bool = False,
        sarcasm_confidence: Optional[float] = None,
    ) -> Dict[str, object]:
        sarcasm_value = sarcasm if sarcasm_aware else None
        confidence_value = sarcasm_confidence if sarcasm_aware else None

        if self.prompting_technique == "zero-shot":
            prompt = self._get_zero_shot_prompt(
                text, sarcasm_value, sarcasm_aware, confidence_value
            )
        elif self.prompting_technique == "chain-of-thought":
            prompt = self._get_chain_of_thought_prompt(
                text, sarcasm_value, sarcasm_aware, confidence_value
            )
        else:
            prompt = self._get_few_shot_prompt(
                text, sarcasm_value, sarcasm_aware, confidence_value
            )

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
            "sarcasm_confidence": sarcasm_confidence if sarcasm_aware else None,
            "sarcasm_aware": sarcasm_aware and sarcasm is not None,
            "response": response.strip(),
            "technique": self.prompting_technique,
        }

    def classify_batch(self, texts: List[str], model: str = None) -> List[Dict]:
        return [self.classify(text, model) for text in texts]
