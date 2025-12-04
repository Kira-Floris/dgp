from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for translation models."""
    model_name: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    
    def __post_init__(self):
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")

