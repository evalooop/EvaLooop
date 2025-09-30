"""Configuration management for EvaLoop framework."""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class EvaluationConfig:
    """Configuration for evaluation experiments."""
    
    # Model configuration
    model: str = ""
    
    # Dataset configuration
    dataset: str = "mbpp_plus"
    languages: List[str] = field(default_factory=lambda: ["python"])
    
    # Evaluation parameters
    max_cycles: int = 10
    temperature: float = 0.0
    top_p: float = 1.0
    batch_size: int = 1
    
    # Output configuration
    output_dir: str = "results/"
    experiment_name: Optional[str] = None
    
    # Hardware configuration
    gpu_ids: Optional[List[int]] = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    trust_remote_code: bool = True
    
    # Generation configuration
    max_new_tokens: int = 1024
    seed: Optional[int] = None
    use_beam_search: bool = False
    
    # Batch configuration for VLLM
    max_batch_size: int = 128
    auto_batch_size: bool = True
    preferred_batch_size: int = 64
    
    # Chat model configuration
    is_chat_model: bool = True
    system_prompt: str = "You are a helpful programming assistant."
    
    # API configuration
    openai_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.model:
            raise ValueError("A model must be specified")
        
        if self.max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")
        
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        
        if not (0.1 <= self.gpu_memory_utilization <= 1.0):
            raise ValueError("gpu_memory_utilization must be between 0.1 and 1.0")
        
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = self._generate_experiment_name()
        
        # Set OpenAI API key from environment if not provided
        if self.openai_api_key is None:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    def _generate_experiment_name(self) -> str:
        """Generate a default experiment name based on configuration."""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self._sanitize_model_name(self.model)
        
        return f"evaloop_{model_name}_{self.dataset}_{timestamp}"
    
    def _sanitize_model_name(self, model_name: str) -> str:
        """Sanitize model name for use in file names."""
        # Extract just the model name part from paths like "meta-llama/Llama-3-70b-instruct-hf"
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        
        # Replace problematic characters
        return model_name.replace("-", "_").replace(".", "_").lower()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model,
            "dataset": self.dataset,
            "languages": self.languages,
            "max_cycles": self.max_cycles,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "batch_size": self.batch_size,
            "output_dir": self.output_dir,
            "experiment_name": self.experiment_name,
            "gpu_ids": self.gpu_ids,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "trust_remote_code": self.trust_remote_code,
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Generate model configuration for the single model."""
        return self._create_model_config(self.model)
    
    def _create_model_config(self, model: str) -> Dict[str, Any]:
        """Create configuration for a specific model."""
        # Determine model type based on model name/path
        if any(provider in model.lower() for provider in ["gpt", "chatgpt", "openai"]):
            model_type = "openai"
        elif "claude" in model.lower():
            model_type = "anthropic"
        elif "/" in model or self._is_huggingface_model(model):
            # Check if model is supported by VLLM
            if self._is_vllm_supported(model):
                model_type = "vllm"
            else:
                model_type = "huggingface"
        else:
            # Assume it's a Hugging Face model name
            model_type = "huggingface"
        
        base_config = {
            "name": self._sanitize_model_name(model),
            "type": model_type,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        
        if model_type == "openai":
            base_config.update({
                "model_id": model,
                "api_key": self.openai_api_key,
                "max_tokens": self.max_model_len,
            })
        elif model_type == "anthropic":
            base_config.update({
                "model_id": model,
                "api_key": os.environ.get("ANTHROPIC_API_KEY"),
                "max_tokens": self.max_model_len,
            })
        elif model_type == "vllm":
            base_config.update({
                "path": model,
                "device": "cuda",
                "gpu_ids": self.gpu_ids or [0],
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "trust_remote_code": self.trust_remote_code,
                "max_new_tokens": self.max_new_tokens,
                "seed": self.seed,
                "use_beam_search": self.use_beam_search,
                "is_chat_model": self.is_chat_model,
                "system_prompt": self.system_prompt,
                # Batch configuration
                "batch_config": {
                    "max_batch_size": self.max_batch_size,
                    "auto_batch_size": self.auto_batch_size,
                    "preferred_batch_size": self.preferred_batch_size,
                },
                # Generation configuration
                "generation_config": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_new_tokens": self.max_new_tokens,
                    "seed": self.seed,
                    "use_beam_search": self.use_beam_search,
                }
            })
        else:  # huggingface
            base_config.update({
                "path": model,
                "device": "cuda" if self.gpu_ids else "auto",
                "max_length": self.max_model_len,
                "trust_remote_code": self.trust_remote_code,
                "max_new_tokens": self.max_new_tokens,
                "seed": self.seed,
                "use_beam_search": self.use_beam_search,
                "is_chat_model": self.is_chat_model,
                "system_prompt": self.system_prompt,
                # Generation configuration
                "generation_config": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_new_tokens": self.max_new_tokens,
                    "seed": self.seed,
                    "use_beam_search": self.use_beam_search,
                }
            })
        
        return base_config
    
    def _is_huggingface_model(self, model: str) -> bool:
        """Check if a model is a HuggingFace model."""
        # Common HuggingFace model patterns
        hf_patterns = [
            "/",  # Path-like format (org/model)
            "meta-llama", "microsoft", "google", "facebook", "huggingface",
            "qwen", "deepseek", "codellama", "mistral", "llama",
            "bigcode", "starcoder", "wizardcoder", "phind"
        ]
        
        model_lower = model.lower()
        return any(pattern in model_lower for pattern in hf_patterns)
    
    def _is_vllm_supported(self, model: str) -> bool:
        """
        Determine if a model is supported by VLLM.
        
        Uses a three-tier strategy:
        1. Check config.json for architecture information (most accurate)
        2. Heuristic check based on model name
        3. Conservative fallback: return False if uncertain
        """
        # First, try to load and check config.json
        config = self._try_load_model_config(model)
        if config:
            is_supported = self._check_architecture_support(config)
            if is_supported is not None:  # If we can make a definitive determination
                return is_supported
        
        # Fall back to heuristic check based on model name
        return self._check_vllm_support_by_name(model)
    
    def _try_load_model_config(self, model: str) -> Optional[Dict[str, Any]]:
        """Try to load model configuration file."""
        import json
        
        try:
            # Check if it's a local path
            if os.path.exists(model):
                config_path = os.path.join(model, 'config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            
            # Try to load from HuggingFace Hub
            else:
                try:
                    from huggingface_hub import hf_hub_download
                    config_path = hf_hub_download(
                        repo_id=model, 
                        filename='config.json',
                        cache_dir=None
                    )
                    with open(config_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except ImportError:
                    pass  # huggingface_hub not installed
                except Exception:
                    pass  # Download failed
        except Exception:
            pass
        
        return None
    
    def _check_architecture_support(self, config: Dict[str, Any]) -> Optional[bool]:
        """
        Check if architecture is supported by VLLM based on config file.
        
        Returns None if unable to determine.
        """
        # VLLM supported architectures (as of early 2025)
        supported_architectures = {
            # LLaMA family
            'LlamaForCausalLM', 'LLaMAForCausalLM',
            # Mistral/Mixtral
            'MistralForCausalLM', 'MixtralForCausalLM',
            # Qwen family
            'QWenLMHeadModel', 'Qwen2ForCausalLM', 'Qwen2MoeForCausalLM',
            # GPT family
            'GPT2LMHeadModel', 'GPTJForCausalLM', 'GPTNeoXForCausalLM', 
            'GPTBigCodeForCausalLM', 'GPTNeoForCausalLM',
            # DeepSeek
            'DeepseekForCausalLM', 'DeepseekV2ForCausalLM',
            # Other mainstream architectures
            'BloomForCausalLM', 'OPTForCausalLM', 'FalconForCausalLM',
            'BaichuanForCausalLM', 'Baichuan2ForCausalLM',
            'ChatGLMModel', 'ChatGLMForConditionalGeneration',
            'InternLMForCausalLM', 'InternLM2ForCausalLM',
            'MPTForCausalLM', 
            'PhiForCausalLM', 'Phi3ForCausalLM',
            'GemmaForCausalLM', 'Gemma2ForCausalLM',
            'StableLmForCausalLM', 'StableLmEpochForCausalLM',
            'PersimmonForCausalLM', 'AquilaForCausalLM', 'AquilaModel',
            'XverseForCausalLM', 'DeciLMForCausalLM',
            'YiForCausalLM', 'Starcoder2ForCausalLM',
            'CohereForCausalLM',  # Command-R
            'DbrxForCausalLM',
        }
        
        # Explicitly unsupported architectures (encoder-decoder models, etc.)
        unsupported_architectures = {
            'T5ForConditionalGeneration',
            'MT5ForConditionalGeneration', 
            'BartForConditionalGeneration',
            'MBartForConditionalGeneration',
            'MarianMTModel',
            'PegasusForConditionalGeneration',
        }
        
        architectures = config.get('architectures', [])
        model_type = config.get('model_type', '').lower()
        
        # Check architecture
        if architectures:
            arch = architectures[0]
            
            # Explicitly supported
            if arch in supported_architectures:
                return True
            
            # Explicitly unsupported
            if arch in unsupported_architectures:
                return False
            
            # Check if it's a CausalLM (usually supported)
            if 'ForCausalLM' in arch or 'LMHeadModel' in arch:
                return True
        
        # Check model_type
        supported_model_types = {
            'llama', 'mistral', 'mixtral', 'qwen', 'qwen2', 
            'gpt2', 'gpt_neox', 'gptj', 'gpt_bigcode',
            'bloom', 'opt', 'falcon', 'baichuan', 
            'chatglm', 'internlm', 'internlm2',
            'mpt', 'phi', 'phi3', 'gemma', 'gemma2',
            'stablelm', 'persimmon', 'aquila', 'xverse', 
            'deci', 'yi', 'deepseek', 'deepseek_v2',
            'starcoder2', 'cohere', 'command-r', 'dbrx'
        }
        
        if model_type in supported_model_types:
            return True
        
        # Unable to determine
        return None
    
    def _check_vllm_support_by_name(self, model: str) -> bool:
        """Heuristic check based on model name."""
        model_lower = model.lower()
        
        # Known supported model families
        supported_families = [
            'llama', 'llama-2', 'llama-3', 'codellama', 'vicuna', 'alpaca',
            'qwen', 'qwen1.5', 'qwen2', 'qwen2.5', 'codeqwen',
            'deepseek', 'deepseek-coder', 'deepseek-v2',
            'mistral', 'mixtral', 'codestral',
            'baichuan', 'chatglm', 'internlm', 'yi', 'gemma',
            'starcoder', 'starcoder2', 'bigcode',
            'phi', 'phi-3', 'wizardcoder', 'wizardlm',
            'command-r', 'cohere', 'dbrx',
        ]
        
        # Check if model matches any supported family
        for family in supported_families:
            if family in model_lower:
                return True
        
        # Explicitly unsupported patterns
        unsupported_patterns = [
            't5-', 'flan-t5', 'mt5',  # T5 family
            'bart-', 'mbart-',  # BART family
            'pegasus-',  # Pegasus
            'whisper-',  # Whisper (speech model)
            'clip-',  # CLIP (vision model)
        ]
        
        for pattern in unsupported_patterns:
            if pattern in model_lower:
                return False
        
        # Conservative approach: return False if uncertain
        return False
    
    def _should_use_vllm(self, model: str) -> bool:
        """Determine if a model should use VLLM based on size and configuration (deprecated)."""
        # This method is now replaced by _is_vllm_supported but kept for backward compatibility
        return self._is_vllm_supported(model)
    
    def get_task_configs(self, task_type: str = "code_generation_summarization") -> List[Dict[str, Any]]:
        """Generate task configurations based on task type."""
        if task_type == "code_generation_summarization":
            return [
                {
                    "type": "code_generation",
                    "language": "python",
                    "prompt_template": "Generate Python code for the following task: \n{description}\nReturn only the code, without explanations or comments.",
                },
                {
                    "type": "code_summarization", 
                    "prompt_template": "Use one sentence to summarize the following code and start with 'write a python function to':\n\n```\n{code}\n```\n\n```\nwrite a python function to\n```",
                },
            ]
        elif task_type == "code_translation":
            # Code translation is not currently supported but kept as interface
            raise NotImplementedError("Code translation is not currently supported")
        else:
            raise ValueError(f"Unsupported task type: {task_type}. Currently only 'code_generation_summarization' is supported.")