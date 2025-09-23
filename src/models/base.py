from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseLLM(ABC):
    """Base class for all LLM implementations."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize the LLM with configuration."""
        self.model_config = model_config
        self.model_name = model_config.get("name", "unnamed_model")
        
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on input prompt."""
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name})"
