"""Model registry for pre-configured models."""

from typing import List, Dict, Any


class ModelRegistry:
    """Registry of pre-configured models."""
    
    def __init__(self):
        """Initialize the model registry."""
        self._models = self._load_predefined_models()
    
    def _load_predefined_models(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined model configurations."""
        return {
            # OpenAI Models
            "gpt-4": {
                "name": "gpt-4",
                "type": "openai",
                "model_id": "gpt-4",
                "max_tokens": 4096,
                "description": "OpenAI GPT-4 model"
            },
            "gpt-4-turbo": {
                "name": "gpt-4-turbo", 
                "type": "openai",
                "model_id": "gpt-4-turbo-preview",
                "max_tokens": 4096,
                "description": "OpenAI GPT-4 Turbo model"
            },
            "gpt-3.5-turbo": {
                "name": "gpt-3.5-turbo",
                "type": "openai", 
                "model_id": "gpt-3.5-turbo",
                "max_tokens": 4096,
                "description": "OpenAI GPT-3.5 Turbo model"
            },
            
            # LLaMA Models
            "llama-3-8b": {
                "name": "llama-3-8b",
                "type": "huggingface",
                "path": "meta-llama/Llama-3-8b-instruct-hf",
                "max_length": 4096,
                "description": "Meta LLaMA 3 8B Instruct model"
            },
            "llama-3-70b": {
                "name": "llama-3-70b",
                "type": "vllm",
                "path": "meta-llama/Llama-3-70b-instruct-hf",
                "max_model_len": 4096,
                "tensor_parallel_size": 4,
                "description": "Meta LLaMA 3 70B Instruct model (VLLM)"
            },
            
            # DeepSeek Models
            "deepseek-coder-6.7b": {
                "name": "deepseek-coder-6.7b",
                "type": "huggingface",
                "path": "deepseek-ai/deepseek-coder-6.7b-instruct",
                "max_length": 4096,
                "description": "DeepSeek Coder 6.7B Instruct model"
            },
            "deepseek-coder-33b": {
                "name": "deepseek-coder-33b",
                "type": "vllm",
                "path": "deepseek-ai/deepseek-coder-33b-instruct",
                "max_model_len": 4096,
                "tensor_parallel_size": 2,
                "description": "DeepSeek Coder 33B Instruct model (VLLM)"
            },
            
            # Qwen Models
            "qwen-72b": {
                "name": "qwen-72b",
                "type": "vllm",
                "path": "Qwen/Qwen-72B-Chat",
                "max_model_len": 4096,
                "tensor_parallel_size": 4,
                "description": "Qwen 72B Chat model (VLLM)"
            },
            
            # Code-specific models
            "codellama-34b": {
                "name": "codellama-34b",
                "type": "vllm",
                "path": "codellama/CodeLlama-34b-Instruct-hf",
                "max_model_len": 4096,
                "tensor_parallel_size": 2,
                "description": "Code Llama 34B Instruct model"
            },
            "starcoder2-15b": {
                "name": "starcoder2-15b",
                "type": "huggingface",
                "path": "bigcode/starcoder2-15b",
                "max_length": 4096,
                "description": "StarCoder2 15B model"
            }
        }
    
    def list_available_models(self) -> List[str]:
        """
        List all available pre-configured models.
        
        Returns:
            List of model names.
        """
        return list(self._models.keys())
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            Model configuration dictionary.
            
        Raises:
            KeyError: If model is not found.
        """
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' not found in registry. Available models: {list(self._models.keys())}")
        
        return self._models[model_name].copy()
    
    def get_models_by_type(self, model_type: str) -> List[str]:
        """
        Get all models of a specific type.
        
        Args:
            model_type: Type of model (e.g., "openai", "huggingface", "vllm").
            
        Returns:
            List of model names of the specified type.
        """
        return [
            name for name, config in self._models.items()
            if config.get("type") == model_type
        ]
    
    def register_model(self, name: str, config: Dict[str, Any]):
        """
        Register a new model configuration.
        
        Args:
            name: Model name.
            config: Model configuration dictionary.
        """
        required_fields = ["name", "type"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Model config must include '{field}' field")
        
        self._models[name] = config.copy()
    
    def get_model_info(self, model_name: str) -> str:
        """
        Get human-readable information about a model.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            Formatted string with model information.
        """
        if model_name not in self._models:
            return f"Model '{model_name}' not found"
        
        config = self._models[model_name]
        info = f"Model: {config['name']}\n"
        info += f"Type: {config['type']}\n"
        
        if "description" in config:
            info += f"Description: {config['description']}\n"
        
        if "path" in config:
            info += f"Path: {config['path']}\n"
        elif "model_id" in config:
            info += f"Model ID: {config['model_id']}\n"
        
        if "max_tokens" in config:
            info += f"Max Tokens: {config['max_tokens']}\n"
        elif "max_length" in config:
            info += f"Max Length: {config['max_length']}\n"
        elif "max_model_len" in config:
            info += f"Max Model Length: {config['max_model_len']}\n"
        
        if "tensor_parallel_size" in config:
            info += f"Tensor Parallel Size: {config['tensor_parallel_size']}\n"
        
        return info
