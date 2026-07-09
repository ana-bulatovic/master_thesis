from typing import Dict, List, Optional
import re

from llm.ollama_client import OllamaClient


class SentimentClassifier:
    """Classify sentiment of text using LLM"""

    SARCASM_EXAMPLES = [
        (
            "Oh wonderful, another premium product that broke in two days. Just perfect.",
            "negative",
        ),
        (
            "Best purchase ever! It stopped working after one hour.",
            "negative",
        ),
        (
            "Amazing quality, exactly what I expected from garbage.",
            "negative",
        ),
        (
            "Love it! Totally worth every penny... NOT.",
            "negative",
        ),
    ]

    STANDARD_EXAMPLES = [
        ("This product is amazing and works perfectly!", "positive"),
        ("I'm very happy with this purchase, excellent quality.", "positive"),
        ("Terrible product, stopped working immediately.", "negative"),
        ("Waste of money, very disappointed.", "negative"),
        ("The product is average, nothing special.", "neutral"),
        ("It's okay, does what it's supposed to.", "neutral"),
    ]

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
                f"Sarcasm detector result: SARCASTIC{confidence_text}.\n"
                "IMPORTANT: This review IS sarcastic.\n"
                "Praise words such as amazing, perfect, great, wonderful, or best "
                "are used ironically and usually express disappointment or criticism.\n"
                "Ignore the literal positive wording. Classify the TRUE intended sentiment.\n"
            )

        return (
            f"Sarcasm detector result: NON-SARCASTIC{confidence_text}. "
            "Classify the literal sentiment of the review.\n"
        )

    def _parse_sentiment(self, response: str, sarcasm: Optional[bool] = None) -> str:
        text = response.lower().strip()
        if not text:
            return "neutral"

        if text in self.sentiment_labels:
            return text

        for pattern in (
            r"final sentiment:\s*(positive|negative|neutral)",
            r"sentiment:\s*(positive|negative|neutral)",
            r"true sentiment:\s*(positive|negative|neutral)",
            r"intended sentiment:\s*(positive|negative|neutral)",
        ):
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        found = re.findall(r"\b(positive|negative|neutral)\b", text)
        if found:
            if sarcasm:
                if "negative" in found:
                    return "negative"
                if "neutral" in found:
                    return "neutral"
            return found[-1]

        return "neutral"

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
        sarcasm_hint = self._build_sarcasm_hint(sarcasm, sarcasm_aware, sarcasm_confidence)
        examples = (
            self.SARCASM_EXAMPLES + self.STANDARD_EXAMPLES[:2]
            if sarcasm_aware and sarcasm
            else self.STANDARD_EXAMPLES
        )

        prompt = "Classify the sentiment of reviews as: positive, negative, or neutral.\n"
        if sarcasm_hint:
            prompt += f"\n{sarcasm_hint}"
        prompt += "Examples:\n"
        for example_text, sentiment in examples:
            prompt += f"Review: {example_text}\nSentiment: {sentiment}\n\n"

        prompt += (
            f"Now classify this review.\n"
            f"Answer with only one word: positive, negative, or neutral.\n\n"
            f"Review: {text}\n\nSentiment:"
        )
        return prompt

    def _get_chain_of_thought_prompt(
        self,
        text: str,
        sarcasm: Optional[bool] = None,
        sarcasm_aware: bool = False,
        sarcasm_confidence: Optional[float] = None,
    ) -> str:
        sarcasm_hint = self._build_sarcasm_hint(sarcasm, sarcasm_aware, sarcasm_confidence)

        if sarcasm_aware and sarcasm:
            step1 = "Step 1: The sarcasm detector says this review is sarcastic. Trust that result."
            step2 = (
                "Step 2: Find ironic praise or criticism. "
                "Positive words may actually mean negative sentiment."
            )
            step3 = "Step 3: Decide the TRUE intended sentiment, not the literal wording."
        elif sarcasm_aware and sarcasm is False:
            step1 = "Step 1: The sarcasm detector says this review is not sarcastic."
            step2 = "Step 2: Identify positive and negative signals in the literal wording."
            step3 = "Step 3: Decide the overall literal tone."
        else:
            step1 = "Step 1: Check whether the review might be sarcastic."
            step2 = "Step 2: Identify positive and negative signals (literal and intended)."
            step3 = "Step 3: Determine the true overall tone."

        return f"""Analyze the sentiment of this review step by step.

{sarcasm_hint}Review: {text}

{step1}
{step2}
{step3}
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
        use_chain_of_thought = (
            self.prompting_technique == "chain-of-thought"
            or (sarcasm_aware and sarcasm is True)
        )

        if self.prompting_technique == "zero-shot" and not (sarcasm_aware and sarcasm):
            prompt = self._get_zero_shot_prompt(
                text, sarcasm_value, sarcasm_aware, confidence_value
            )
        elif use_chain_of_thought:
            prompt = self._get_chain_of_thought_prompt(
                text, sarcasm_value, sarcasm_aware, confidence_value
            )
        else:
            prompt = self._get_few_shot_prompt(
                text, sarcasm_value, sarcasm_aware, confidence_value
            )

        response = self.client.generate(prompt, model)
        sentiment = self._parse_sentiment(response, sarcasm=sarcasm_value)

        return {
            "text": text,
            "sentiment": sentiment,
            "sarcasm": sarcasm if sarcasm_aware else None,
            "sarcasm_confidence": sarcasm_confidence if sarcasm_aware else None,
            "sarcasm_aware": sarcasm_aware and sarcasm is not None,
            "response": response.strip(),
            "technique": (
                "chain-of-thought"
                if use_chain_of_thought and self.prompting_technique != "chain-of-thought"
                else self.prompting_technique
            ),
        }

    def classify_batch(self, texts: List[str], model: str = None) -> List[Dict]:
        return [self.classify(text, model) for text in texts]
