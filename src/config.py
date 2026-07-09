import os
import json
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from paths import PROJECT_ROOT


@dataclass
class ModelConfig:
    """Configuration for LLM models"""

    name: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    num_predict: int = 128


@dataclass
class PipelineConfig:
    """Configuration for the NLP pipeline"""

    sarcasm_model: str = "llama3.1"
    sentiment_model: str = "llama3.1"
    summarization_model: str = "llama3.1"
    prompting_technique: str = "few-shot"
    use_sarcasm_detection: bool = True
    use_sentiment_for_summarization: bool = True
    use_sarcasm_for_summarization: bool = True
    data_dir: str = "data/processed"
    output_dir: str = "results"
    batch_size: int = 8
    use_rouge: bool = True
    use_bertscore: bool = True
    use_traditional_metrics: bool = True


class Config:
    """Main configuration class"""

    def __init__(self, config_path: Optional[str] = None):
        self.model_config = ModelConfig(name="llama3.1")
        self.pipeline_config = PipelineConfig()

        default_path = PROJECT_ROOT / "config" / "config.yaml"
        resolved_path = Path(config_path) if config_path else default_path

        if resolved_path.exists():
            self.load_from_file(str(resolved_path))

    def load_from_file(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle) or {}

        if "model" in config_dict:
            self.model_config = ModelConfig(**config_dict["model"])

        if "pipeline" in config_dict:
            self.pipeline_config = PipelineConfig(**config_dict["pipeline"])

    def to_dict(self) -> Dict:
        return {
            "model": self.model_config.__dict__,
            "pipeline": self.pipeline_config.__dict__,
        }

    def save_to_file(self, config_path: str):
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.dump(self.to_dict(), handle, default_flow_style=False, allow_unicode=True)
