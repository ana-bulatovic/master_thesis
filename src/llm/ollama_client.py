import requests
from typing import List, Dict, Optional

from config import ModelConfig


class OllamaClient:
    """Client for interacting with Ollama LLM API"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.base_url = config.base_url
        self.model_name = config.name

    def check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except Exception as exc:
            print(f"Error listing models: {exc}")
            return []

    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        model = model or self.model_name

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.config.temperature,
                    "top_k": self.config.top_k,
                    "top_p": self.config.top_p,
                    "num_predict": self.config.num_predict,
                },
                timeout=300,
            )

            if response.status_code == 200:
                return response.json()["response"]
            raise Exception(f"API error: {response.text}")
        except Exception as exc:
            print(f"Error generating text: {exc}")
            return ""

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        model = model or self.model_name

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "temperature": self.config.temperature,
                },
                timeout=300,
            )

            if response.status_code == 200:
                return response.json()["message"]["content"]
            raise Exception(f"API error: {response.text}")
        except Exception as exc:
            print(f"Error in chat: {exc}")
            return ""
