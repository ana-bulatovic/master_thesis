import os
import json
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    # Models to use
    sarcasm_model: str = "llama2"
    sentiment_model: str = "llama2"
    summarization_model: str = "llama2"
    
    # Prompting techniques: zero-shot, few-shot, chain-of-thought
    prompting_technique: str = "few-shot"
    
    # Pipeline variants
    use_sarcasm_detection: bool = True
    use_sentiment_for_summarization: bool = True
    
    # Data paths
    data_dir: str = "./data"
    output_dir: str = "./results"
    
    # Batch processing
    batch_size: int = 8
    
    # Evaluation metrics
    use_rouge: bool = True
    use_bertscore: bool = True
    use_traditional_metrics: bool = True

class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.model_config = ModelConfig(name="llama2")
        self.pipeline_config = PipelineConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
        if 'model' in config_dict:
            model_cfg = config_dict['model']
            self.model_config = ModelConfig(**model_cfg)
        
        if 'pipeline' in config_dict:
            pipeline_cfg = config_dict['pipeline']
            self.pipeline_config = PipelineConfig(**pipeline_cfg)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'model': self.model_config.__dict__,
            'pipeline': self.pipeline_config.__dict__
        }
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file"""
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
