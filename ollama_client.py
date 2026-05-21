import requests
import json
from typing import List, Dict, Optional
from config import ModelConfig

class OllamaClient:
    """Client for interacting with Ollama LLM API"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.base_url = config.base_url
        self.model_name = config.name
    
    def check_connection(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        """List available models on Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Generate text using Ollama
        
        Args:
            prompt: Input prompt
            model: Model name (uses default if not specified)
        
        Returns:
            Generated text
        """
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
                }
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                raise Exception(f"API error: {response.text}")
        
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""
    
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        """
        Chat interface (if supported by Ollama)
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name
        
        Returns:
            Chat response
        """
        model = model or self.model_name
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "temperature": self.config.temperature,
                }
            )
            
            if response.status_code == 200:
                return response.json()['message']['content']
            else:
                raise Exception(f"API error: {response.text}")
        
        except Exception as e:
            print(f"Error in chat: {e}")
            return ""
