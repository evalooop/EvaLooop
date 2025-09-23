import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from typing import Dict, Any, Optional, List, Union, Literal
from dataclasses import dataclass
import random
import numpy as np

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from .base import BaseLLM

@dataclass
class QuantizationConfig:
    """Configuration for BitsAndBytes quantization"""
    quantization: Optional[str] = None  # '4bit', '8bit', or None
    bnb_4bit_quant_type: str = "nf4"  # For 4-bit: "nf4" or "fp4"
    bnb_4bit_use_double_quant: bool = True  # Use double quantization
    bnb_4bit_compute_dtype: Union[torch.dtype, str] = torch.float32  # Computation dtype for 4-bit
    
    def __post_init__(self):
        """Convert string dtype to torch.dtype if needed and validate"""
        if isinstance(self.bnb_4bit_compute_dtype, str):
            # Handle string representations of torch dtypes
            dtype_mapping = {
                "torch.float16": torch.float16,
                "torch.float32": torch.float32,
                "torch.bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            if self.bnb_4bit_compute_dtype in dtype_mapping:
                self.bnb_4bit_compute_dtype = dtype_mapping[self.bnb_4bit_compute_dtype]
            else:
                raise ValueError(f"Unsupported dtype string: {self.bnb_4bit_compute_dtype}. "
                               f"Supported: {list(dtype_mapping.keys())}")
        
        # Validate quantization type
        if self.quantization is not None and self.quantization not in ['4bit', '8bit']:
            raise ValueError(f"Unsupported quantization type: {self.quantization}. "
                           f"Supported: ['4bit', '8bit', None]")

class NoiseConfig:
    """Configuration for adding noise to model parameters"""
    def __init__(self, noise_type: str = "gaussian", noise_level: float = 1e-3):
        self.noise_type = noise_type
        self.noise_level = noise_level
        self._validate()
    
    def _validate(self):
        valid_noise_types = ["uniform", "gaussian"]
        if self.noise_type not in valid_noise_types:
            raise ValueError(f"Noise type must be one of {valid_noise_types}")
        if self.noise_level < 0:
            raise ValueError("Noise level must be positive")

@dataclass
class GenerationStrategy:
    """Configuration for different generation strategies"""
    num_return_sequences: int = 1
    max_length: int = 512
    max_new_tokens: int = 512  # Added for compatibility with newer models
    temperature: float = 0.7
    top_p: float = 0.95
    num_beams: int = 10
    use_beam_search: bool = False
    seed: Optional[int] = None  # Add seed parameter

class HuggingFaceLLM(BaseLLM):
    """Implementation for Hugging Face based models like LLaMA, DeepSeek, Qwen, etc."""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model_path = model_config["path"]

        # Quantization configuration - updated to align with noise.py
        self.quant_config = None
        if "quantization" in model_config:
            self.quant_config = QuantizationConfig(**model_config["quantization"])

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_map = model_config.get("device_map", "auto")
        
        # Legacy GPU ID support for backward compatibility
        self.gpu_id = model_config.get("gpu_id", None)
        if self.gpu_id is not None:
            if torch.cuda.is_available():
                self.device = f"cuda:{self.gpu_id}"
                # Set environment variable to restrict visible devices
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            else:
                print(f"Warning: CUDA not available, falling back to CPU despite gpu_id={self.gpu_id}")
                self.device = "cpu"
        
        self.max_length = model_config.get("max_length", 2048)
        self.gen_config = model_config.get('generation_config')
        self.is_chat_model = model_config.get('is_chat_model', False)
        self.system_prompt = model_config.get('system_prompt', "You are a helpful assistant.")
        self.seed = model_config.get('seed', None)
        self.use_temperature = model_config.get('use_temperature', False)
        
        # Device for inputs (important for quantized models)
        self.input_device = None
        
        self.load()
        
    def load(self) -> None:
        """Load model and tokenizer"""
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Prepare model loading arguments
        model_kwargs = {
            "trust_remote_code": True
        }
        
        # Handle quantization following noise.py pattern
        if self.quant_config is not None and self.quant_config.quantization is not None:
            print(f"Loading model with BitsAndBytes quantization ({self.quant_config.quantization})")
            
            if self.quant_config.quantization == '8bit':
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                model_kwargs["torch_dtype"] = torch.float16
            elif self.quant_config.quantization == '4bit':
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.quant_config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.quant_config.bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype=self.quant_config.bnb_4bit_compute_dtype,
                )
                model_kwargs["torch_dtype"] = torch.float32 if self.quant_config.quantization == '4bit' else torch.float16
            else:
                raise ValueError(f"Unsupported quantization type: {self.quant_config.quantization}")
            
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = self.device_map
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            print(f"Model loaded with {self.quant_config.quantization} quantization")
            
            # Set input device for quantized models (following noise.py pattern)
            if self.quant_config.quantization == '4bit':
                if hasattr(self.model, 'hf_device_map'):
                    # If model uses a device map, get first device
                    first_device_key = next(iter(self.model.hf_device_map.values()))
                    self.input_device = torch.device(first_device_key)
                else:
                    # Otherwise use first parameter's device
                    self.input_device = next(self.model.parameters()).device
            else:
                self.input_device = self.device
            
        else:
            # Load original model without quantization (following noise.py pattern)
            print("Loading original model...")
            model_kwargs["torch_dtype"] = torch.float16 if "cuda" in str(self.device) else torch.float32
            
            # Handle device mapping
            if self.gpu_id is not None:
                # When using specific GPU, don't use device_map="auto"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs
                ).to(self.device)
                print(f"Model loaded on specific GPU: {self.device}")
            else:
                # Use device_map for automatic mapping
                model_kwargs["device_map"] = self.device_map if "cuda" in str(self.device) else None
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
                print(f"Model loaded with device mapping: {self.device_map}")
            
            self.input_device = self.device
        
        # Handle missing pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Clear GPU cache after loading to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _set_seed(self, seed: Optional[int] = None) -> None:
        """Set random seeds for reproducibility"""
        # Use instance seed if no seed is provided
        if seed is None:
            seed = self.seed
        
        # Skip if no seed is specified
        if seed is None:
            return
            
        # Set seeds for all random number generators
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # For older PyTorch versions, enforce deterministic operations
        if hasattr(torch, 'set_deterministic'):
            torch.set_deterministic(True)
        else:
            # Alternative for older PyTorch versions
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def get_memory_stats(self) -> dict:
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        stats = {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3
        }
        
        if self.quant_config is not None and self.quant_config.quantization is not None:
            stats["quantization"] = {
                "type": self.quant_config.quantization,
                "bnb_4bit_quant_type": self.quant_config.bnb_4bit_quant_type if self.quant_config.quantization == '4bit' else "N/A",
                "bnb_4bit_compute_dtype": str(self.quant_config.bnb_4bit_compute_dtype) if self.quant_config.quantization == '4bit' else "N/A",
                "bnb_4bit_use_double_quant": self.quant_config.bnb_4bit_use_double_quant if self.quant_config.quantization == '4bit' else "N/A"
            }
            
        if self.gpu_id is not None:
            stats["gpu_id"] = self.gpu_id
            
        return stats

    def _get_generation_config(self, strategy: GenerationStrategy) -> GenerationConfig:
        """Create generation configuration based on strategy"""
        config_dict = {
            'max_length': strategy.max_length,
            'max_new_tokens': strategy.max_new_tokens,
            'num_return_sequences': strategy.num_return_sequences,
        }
        
        # Add seed to generation config if provided
        if strategy.seed is not None:
            config_dict['seed'] = strategy.seed
        
        # Check if temperature is explicitly specified to override default behavior
        use_temperature = getattr(self, 'use_temperature', False)
        
        if strategy.num_return_sequences == 1 and not use_temperature:
            # Use greedy decoding for single sequence (unless temperature is explicitly enabled)
            print("Using greedy decoding")
            config_dict.update({
                'do_sample': False,
            })
        elif strategy.use_beam_search:
            # Use beam search for multiple sequences
            print("Using beam search")
            config_dict.update({
                'num_beams': strategy.num_beams,
                'do_sample': False,
            })
        else:
            # Use temperature sampling
            print("Using temperature sampling")
            config_dict.update({
                'temperature': strategy.temperature,
                'top_p': strategy.top_p,
                'do_sample': True,
            })
        
        return GenerationConfig(**config_dict)
    
    def _extract_completion(self, full_text: str, prompt: str) -> str:
        """Extract only the completion part from the generated text"""
        output = full_text[len(prompt):].strip()
        gen_solution = output

        if gen_solution is not None:
            return gen_solution
        else:
            return output
            
    def generate(self, prompt: str, **kwargs) -> Union[str, List[str]]:
        """
        Generate completion(s) for a given prompt.
        
        Args:
            prompt: Input prompt text
            **kwargs: Generation parameters to override defaults
        
        Returns:
            Single string if num_return_sequences=1, otherwise list of strings
        """
        # Start with default strategy and update with any provided kwargs
        strategy = GenerationStrategy()
        if self.gen_config:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(self.gen_config)
            strategy = GenerationStrategy(**strategy_dict)
            
        # Override strategy with any kwargs
        for key, value in kwargs.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)
        
        # Set random seed for reproducibility
        self._set_seed(strategy.seed)

        # Handle chat models differently (like Qwen)
        if self.is_chat_model:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Use the chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = self.tokenizer([text], return_tensors="pt")
            else:
                # Fallback for models without chat template
                inputs = self.tokenizer(prompt, return_tensors="pt")
        else:
            # Regular completion models
            inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Handle device placement for inputs (following noise.py pattern)
        if self.quant_config is not None and self.quant_config.quantization == '4bit':
            inputs = {k: v.to(self.input_device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        generation_config = self._get_generation_config(strategy)

        with torch.no_grad():
            # print(f"generating with prompt:\n===\n{prompt}\n===")
            
            if self.is_chat_model:
                # For chat models, we track the new tokens separately
                input_length = inputs['input_ids'].shape[1]  # Use dict access for consistency
                outputs = self.model.generate(
                    inputs['input_ids'],
                    generation_config=generation_config,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                )
                # Only keep the newly generated tokens
                generated_ids = [output_ids[input_length:] for output_ids in outputs]
                decoded_outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            else:
                # Traditional approach for completion models
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                )
                # Decode and extract completions
                decoded_outputs = [
                    self._extract_completion(
                        self.tokenizer.decode(output, skip_special_tokens=True),
                        prompt
                    )
                    for output in outputs
                ]
                
        print("done generating")

        # Return single string if num_return_sequences=1, otherwise list
        return decoded_outputs[0] if strategy.num_return_sequences == 1 else decoded_outputs
        
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate completions for multiple prompts"""
        # Get seed from kwargs if provided
        seed = kwargs.get('seed', self.seed)
        
        results = []
        for i, prompt in enumerate(prompts):
            # For batch generation with seed, increment seed for each prompt
            # to get different but reproducible outputs for each prompt
            if seed is not None:
                current_seed = seed + i
                current_kwargs = {**kwargs, 'seed': current_seed}
            else:
                current_kwargs = kwargs
                
            result = self.generate(prompt, **current_kwargs)
            results.append(result)
        return results
    
    def add_noise(self, noise_config: NoiseConfig):
        """Add noise to model parameters (following noise.py pattern)"""
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    if noise_config.noise_type == 'gaussian':
                        noise = torch.randn_like(param) * noise_config.noise_level
                    else:  # uniform
                        noise = (torch.rand_like(param) * 2 - 1) * noise_config.noise_level
                    param.add_(noise)