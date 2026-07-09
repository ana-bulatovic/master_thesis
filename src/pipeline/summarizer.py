from typing import Dict, List, Optional

from llm.ollama_client import OllamaClient


class TextSummarizer:
    """Generate abstractive summaries using LLM"""

    def __init__(
        self,
        client: OllamaClient,
        prompting_technique: str = "few-shot",
        use_sentiment_info: bool = False,
        use_sarcasm_info: bool = False,
    ):
        self.client = client
        self.prompting_technique = prompting_technique
        self.use_sentiment_info = use_sentiment_info
        self.use_sarcasm_info = use_sarcasm_info

    def _build_context_hints(
        self, sentiment: Optional[str] = None, sarcasm: Optional[bool] = None
    ) -> str:
        hints = []

        if self.use_sarcasm_info and sarcasm is not None:
            label = "sarcastic" if sarcasm else "non-sarcastic"
            hints.append(f"The review is {label}.")

        if self.use_sentiment_info and sentiment:
            hints.append(f"The review expresses a {sentiment} sentiment.")

        if not hints:
            return ""

        return "\n".join(hints) + "\n"

    def _get_zero_shot_prompt(
        self,
        text: str,
        sentiment: Optional[str] = None,
        sarcasm: Optional[bool] = None,
    ) -> str:
        context_hint = self._build_context_hints(sentiment, sarcasm)

        return f"""Summarize the following review in ONE short sentence.

ONLY use information explicitly mentioned in the review.
Do not add new details.
Do not invent product features.
{context_hint}
Review:
{text}

Summary:
"""

    def _get_few_shot_prompt(
        self,
        text: str,
        sentiment: Optional[str] = None,
        sarcasm: Optional[bool] = None,
    ) -> str:
        examples = [
            (
                "This smartphone has excellent camera quality and long battery life, but it's quite expensive.",
                "High-end smartphone with great camera and battery, though pricey.",
            ),
            (
                "The laptop is very slow and keeps freezing. I regret buying it.",
                "Sluggish laptop with freezing issues - a disappointing purchase.",
            ),
        ]

        context_hint = self._build_context_hints(sentiment, sarcasm)

        prompt = (
            "Summarize the following review in ONE short sentence.\n\n"
            "ONLY use information explicitly mentioned in the review.\n"
            "Do not add new details.\n"
            "Do not invent product features.\n"
            f"{context_hint}\nExamples:\n"
        )

        for example_text, summary in examples:
            prompt += f"Review: {example_text}\nSummary: {summary}\n\n"

        prompt += f"Now summarize this review:\n\nReview: {text}\n\nSummary:"
        return prompt

    def _get_chain_of_thought_prompt(
        self,
        text: str,
        sentiment: Optional[str] = None,
        sarcasm: Optional[bool] = None,
    ) -> str:
        context_hint = self._build_context_hints(sentiment, sarcasm)

        return f"""Summarize the following review in ONE short sentence.

ONLY use information explicitly mentioned in the review.
Do not add new details.
Do not invent product features.
{context_hint}
Review: {text}

Step 1: Identify the main product/service being reviewed.
Step 2: List key points (positive, negative, or neutral).
Step 3: Generate a concise summary combining key points.

Summary:"""

    def summarize(
        self,
        text: str,
        sentiment: Optional[str] = None,
        sarcasm: Optional[bool] = None,
        model: str = None,
    ) -> Dict[str, object]:
        if self.prompting_technique == "zero-shot":
            prompt = self._get_zero_shot_prompt(text, sentiment, sarcasm)
        elif self.prompting_technique == "chain-of-thought":
            prompt = self._get_chain_of_thought_prompt(text, sentiment, sarcasm)
        else:
            prompt = self._get_few_shot_prompt(text, sentiment, sarcasm)

        response = self.client.generate(prompt, model)

        return {
            "original_text": text,
            "summary": response.strip(),
            "sentiment": sentiment,
            "sarcasm": sarcasm,
            "sentiment_aware": self.use_sentiment_info and sentiment is not None,
            "sarcasm_aware": self.use_sarcasm_info and sarcasm is not None,
            "technique": self.prompting_technique,
        }

    def summarize_batch(
        self,
        texts: List[str],
        sentiments: List[Optional[str]] = None,
        sarcasms: List[Optional[bool]] = None,
        model: str = None,
    ) -> List[Dict]:
        if sentiments is None:
            sentiments = [None] * len(texts)
        if sarcasms is None:
            sarcasms = [None] * len(texts)

        return [
            self.summarize(text, sentiment, sarcasm, model)
            for text, sentiment, sarcasm in zip(texts, sentiments, sarcasms)
        ]
