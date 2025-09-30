import os
import torch
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import random
import numpy as np

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    raise ImportError("vLLM is required for this implementation. Install with: pip install vllm")

from .base import BaseLLM


@dataclass
class GenerationStrategy:
    """Configuration for different generation strategies."""
    num_return_sequences: int = 1
    max_length: int = 512
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    num_beams: int = 1  # vLLM doesn't support beam search in the same way
    use_beam_search: bool = False
    seed: Optional[int] = None
    
    # vLLM specific parameters
    top_k: int = -1  # Disabled by default
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None


@dataclass 
class BatchConfig:
    """Configuration for batch processing optimization."""
    max_batch_size: int = 32  # Maximum batch size for processing
    auto_batch_size: bool = True  # Automatically determine optimal batch size
    batch_timeout: float = 0.1  # Timeout for batching in seconds
    preferred_batch_size: int = 8  # Preferred batch size when auto-batching


class VllmLLM(BaseLLM):
    """vLLM implementation for accelerated inference with batch processing."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize vLLM model with configuration.
        
        Args:
            model_config: Dictionary containing model configuration parameters.
        """
        super().__init__(model_config)
        self.model_path = model_config["path"]
        
        # Basic configuration
        self.max_length = model_config.get("max_length", 2048)
        self.gen_config = model_config.get('generation_config', {})
        self.is_chat_model = model_config.get('is_chat_model', False)
        self.system_prompt = model_config.get('system_prompt', "You are a helpful assistant.")
        self.seed = model_config.get('seed', None)
        self.use_temperature = model_config.get('use_temperature', False)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_id = model_config.get("gpu_id", None)
        self.tensor_parallel_size = model_config.get("tensor_parallel_size", 1)
        
        # Batch processing configuration
        batch_config_dict = model_config.get("batch_config", {})
        self.batch_config = BatchConfig(**batch_config_dict)
        
        # vLLM specific configuration
        self.trust_remote_code = model_config.get("trust_remote_code", True)
        self.max_model_len = model_config.get("max_model_len", None)
        self.swap_space = model_config.get("swap_space", 4)  # GB
        self.gpu_memory_utilization = model_config.get("gpu_memory_utilization", 0.9)
        
        # Initialize model
        self.model = None
        self.load()
        
    def load(self) -> None:
        """Load the vLLM model with optimized configuration."""
        if not HAS_VLLM:
            raise RuntimeError("vLLM is not available. Please install it with: pip install vllm")
            
        print(f"Loading model {self.model_path} with vLLM...")
        
        # Debug CUDA availability
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
        
        # Handle GPU selection first, before other configurations
        if self.gpu_id is not None:
            # Convert gpu_id to proper format for CUDA_VISIBLE_DEVICES
            if isinstance(self.gpu_id, list):
                gpu_ids = ",".join(map(str, self.gpu_id))
                print(f"Setting CUDA_VISIBLE_DEVICES to: {gpu_ids}")
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
                available_gpus = len(self.gpu_id)
            else:
                print(f"Setting CUDA_VISIBLE_DEVICES to: {self.gpu_id}")
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
                available_gpus = 1
        else:
            available_gpus = torch.cuda.device_count()
        
        print(f"Available GPUs for this model: {available_gpus}")
        
        # Prepare vLLM initialization arguments
        model_kwargs = {
            "model": self.model_path,
            "trust_remote_code": self.trust_remote_code,
            "swap_space": self.swap_space,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
        
        # Set tensor parallel size for multi-GPU setups
        if available_gpus > 1 and self.tensor_parallel_size > 1:
            # Ensure tensor_parallel_size doesn't exceed available GPUs
            effective_tp_size = min(self.tensor_parallel_size, available_gpus)
            model_kwargs["tensor_parallel_size"] = effective_tp_size
            print(f"Using tensor parallelism with {effective_tp_size} GPUs")
        elif available_gpus == 1:
            # Single GPU, no tensor parallelism
            model_kwargs["tensor_parallel_size"] = 1
            print(f"Using single GPU")
        
        # Set maximum model length if specified
        if self.max_model_len is not None:
            model_kwargs["max_model_len"] = self.max_model_len
            print(f"Setting max_model_len to: {self.max_model_len}")
        
        # Handle CPU inference fallback
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU inference")
            # Note: vLLM primarily supports GPU inference
            # CPU support may be limited
        else:
            print(f"CUDA available with {torch.cuda.device_count()} total GPUs")
        
        print(f"vLLM model kwargs: {model_kwargs}")
        
        try:
            self.model = LLM(**model_kwargs)
            print("vLLM model loaded successfully")
            
            # Determine optimal batch size if auto-batching is enabled
            if self.batch_config.auto_batch_size:
                self._determine_optimal_batch_size()
                
        except Exception as e:
            print(f"Error loading model with vLLM: {e}")
            raise
    
    def _determine_optimal_batch_size(self) -> None:
        """Automatically determine optimal batch size based on available memory."""
        if not torch.cuda.is_available():
            self.batch_config.max_batch_size = 1
            return
            
        try:
            # Get available GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = total_memory * self.gpu_memory_utilization
            
            # Rough estimation: larger models need smaller batch sizes
            # This is a heuristic and may need tuning for specific models
            if available_memory > 40e9:  # > 40GB
                optimal_batch = min(64, self.batch_config.max_batch_size)
            elif available_memory > 20e9:  # > 20GB  
                optimal_batch = min(32, self.batch_config.max_batch_size)
            elif available_memory > 10e9:  # > 10GB
                optimal_batch = min(16, self.batch_config.max_batch_size)
            else:
                optimal_batch = min(8, self.batch_config.max_batch_size)
            
            self.batch_config.preferred_batch_size = optimal_batch
            print(f"Auto-determined optimal batch size: {optimal_batch}")
            
        except Exception as e:
            print(f"Could not determine optimal batch size: {e}")
            print(f"Using default batch size: {self.batch_config.preferred_batch_size}")
    
    def _set_seed(self, seed: Optional[int] = None) -> None:
        """Set random seeds for reproducibility.
        
        Args:
            seed: Random seed value. Uses instance seed if None.
        """
        if seed is None:
            seed = self.seed
            
        if seed is None:
            return
            
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _create_sampling_params(self, strategy: GenerationStrategy) -> SamplingParams:
        """Create vLLM SamplingParams from GenerationStrategy.
        
        Args:
            strategy: Generation strategy configuration.
            
        Returns:
            SamplingParams object for vLLM.
        """
        params = {
            "max_tokens": strategy.max_new_tokens,
            "n": strategy.num_return_sequences,
            "seed": strategy.seed,
        }
        
        # Handle different sampling strategies
        use_temperature = getattr(self, 'use_temperature', False) or strategy.temperature != 1.0
        
        if strategy.num_return_sequences == 1 and not use_temperature:
            # Greedy decoding
            params.update({
                "temperature": 0.0,
                "top_p": 1.0,
            })
            print("Using greedy decoding")
        else:
            # Sampling with temperature
            params.update({
                "temperature": strategy.temperature,
                "top_p": strategy.top_p,
                "top_k": strategy.top_k,
                "frequency_penalty": strategy.frequency_penalty,
                "presence_penalty": strategy.presence_penalty,
                "repetition_penalty": strategy.repetition_penalty,
            })
            print(f"Using temperature sampling (T={strategy.temperature})")
        
        # Add stop tokens if specified
        if strategy.stop:
            params["stop"] = strategy.stop
            
        return SamplingParams(**params)
    
    def _prepare_chat_prompt(self, prompt: str) -> str:
        """Prepare prompt for chat models using system prompt.
        
        Args:
            prompt: User input prompt.
            
        Returns:
            Formatted prompt string.
        """
        if self.is_chat_model:
            # Simple chat format - can be extended for specific model formats
            return f"System: {self.system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        return prompt
    
    def generate(self, prompt: str, **kwargs) -> Union[str, List[str]]:
        """Generate completion(s) for a given prompt.
        
        Args:
            prompt: Input prompt text.
            **kwargs: Generation parameters to override defaults.
            
        Returns:
            Single string if num_return_sequences=1, otherwise list of strings.
        """
        # Create generation strategy with defaults and overrides
        strategy = GenerationStrategy()
        if self.gen_config:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(self.gen_config)
            strategy = GenerationStrategy(**strategy_dict)
        
        # Apply any kwargs overrides
        for key, value in kwargs.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)
        
        # Set random seed
        self._set_seed(strategy.seed)
        
        # Prepare prompt for chat models
        formatted_prompt = self._prepare_chat_prompt(prompt)
        
        # Create sampling parameters
        sampling_params = self._create_sampling_params(strategy)
        
        try:
            # Generate using vLLM
            outputs = self.model.generate([formatted_prompt], sampling_params)
            
            # Extract generated text
            results = []
            for output in outputs:
                for completion in output.outputs:
                    results.append(completion.text.strip())
            
            print("Generation completed")
            
            # Return format based on num_return_sequences
            if strategy.num_return_sequences == 1:
                return results[0] if results else ""
            else:
                return results
                
        except Exception as e:
            print(f"Error during generation: {e}")
            raise
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate completions for multiple prompts with optimized batching.
        
        Args:
            prompts: List of input prompt texts.
            **kwargs: Generation parameters to override defaults.
            
        Returns:
            List of generated completions.
        """
        if not prompts:
            return []
        
        # Create generation strategy
        strategy = GenerationStrategy()
        if self.gen_config:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(self.gen_config)
            strategy = GenerationStrategy(**strategy_dict)
        
        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)
        
        # Set random seed
        self._set_seed(strategy.seed)
        
        # Prepare all prompts
        formatted_prompts = [self._prepare_chat_prompt(prompt) for prompt in prompts]
        
        # Create sampling parameters
        sampling_params = self._create_sampling_params(strategy)
        
        # Determine batch size
        if len(prompts) <= self.batch_config.preferred_batch_size:
            # Single batch processing
            return self._process_batch(formatted_prompts, sampling_params)
        else:
            # Multi-batch processing
            return self._process_multi_batch(formatted_prompts, sampling_params)
    
    def _process_batch(self, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
        """Process a single batch of prompts.
        
        Args:
            prompts: List of formatted prompts.
            sampling_params: vLLM sampling parameters.
            
        Returns:
            List of generated completions.
        """
        try:
            outputs = self.model.generate(prompts, sampling_params)
            
            results = []
            for output in outputs:
                # Handle multiple completions per prompt if n > 1
                completions = [comp.text.strip() for comp in output.outputs]
                if len(completions) == 1:
                    results.append(completions[0])
                else:
                    results.append(completions)
            
            return results
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            raise
    
    def _process_multi_batch(self, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
        """Process multiple batches of prompts for memory efficiency.
        
        Args:
            prompts: List of formatted prompts.
            sampling_params: vLLM sampling parameters.
            
        Returns:
            List of generated completions.
        """
        results = []
        batch_size = self.batch_config.preferred_batch_size
        
        print(f"Processing {len(prompts)} prompts in batches of {batch_size}")
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = self._process_batch(batch_prompts, sampling_params)
            results.extend(batch_results)
            
            print(f"Processed batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
        
        return results
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics.
        
        Returns:
            Dictionary containing memory usage information.
        """
        stats = {"model_type": "vLLM"}
        
        if torch.cuda.is_available():
            stats.update({
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "device_count": torch.cuda.device_count(),
            })
        else:
            stats["device"] = "CPU"
        
        # Add batch configuration info
        stats["batch_config"] = {
            "max_batch_size": self.batch_config.max_batch_size,
            "preferred_batch_size": self.batch_config.preferred_batch_size,
            "auto_batch_size": self.batch_config.auto_batch_size,
        }
        
        if self.tensor_parallel_size > 1:
            stats["tensor_parallel_size"] = self.tensor_parallel_size
            
        return stats
    
    def __del__(self):
        """Cleanup resources when object is destroyed."""
        if hasattr(self, 'model') and self.model is not None:
            # vLLM handles cleanup automatically
            pass