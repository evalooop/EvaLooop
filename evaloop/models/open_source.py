"""Hugging Face based model implementation."""

import logging
import os
import random
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from .base import BaseLLM


logger = logging.getLogger(__name__)


class HuggingFaceModelError(Exception):
    """Base exception for HuggingFace model related errors."""
    pass


class HuggingFaceModelLoadError(HuggingFaceModelError):
    """Raised when model loading fails."""
    pass


class HuggingFaceGenerationError(HuggingFaceModelError):
    """Raised when text generation fails."""
    pass


@dataclass
class GenerationParameters:
    """Generation parameters for Hugging Face models."""

    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    num_return_sequences: int = 1
    num_beams: int = 1
    use_beam_search: bool = False
    seed: Optional[int] = None


class HuggingFaceLLM(BaseLLM):
    """Hugging Face transformers based language model.
    
    This class provides a wrapper around Hugging Face transformers for text
    generation, with support for multi-GPU inference, device mapping, and
    various generation strategies.
    
    Attributes:
        model_path: Path to the Hugging Face model.
        trust_remote_code: Whether to trust remote code for model loading.
        max_length: Maximum sequence length for the model.
        is_chat_model: Whether the model is a chat/instruction-tuned model.
        system_prompt: System prompt for chat models.
        device_map: Device mapping configuration.
        gpu_ids: List of GPU IDs to use for inference.
        device: Primary device for model inference.
        tokenizer: Hugging Face tokenizer instance.
        model: Hugging Face model instance.
        input_device: Device where input tensors should be placed.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize Hugging Face model with configuration.
        
        Args:
            model_config: Configuration dictionary produced by the CLI layer.
                Expected keys: path, trust_remote_code, max_length, is_chat_model,
                system_prompt, device_map, gpu_ids, device, generation_config.
                
        Raises:
            HuggingFaceModelLoadError: If model loading fails.
            ValueError: If required configuration parameters are missing.
        """
        super().__init__(model_config)
        self.logger = logging.getLogger(f"{__name__}.{self.model_name}")

        # Validate required configuration
        if "path" not in model_config:
            raise ValueError("Model path is required for HuggingFace models")
            
        self.model_path = model_config["path"]
        self.trust_remote_code = model_config.get("trust_remote_code", True)
        self.max_length = model_config.get("max_length", 2048)
        self.is_chat_model = model_config.get("is_chat_model", False)
        self.system_prompt = model_config.get(
            "system_prompt", "You are a helpful assistant."
        )
        self.device_map = model_config.get("device_map")
        
        # Handle GPU IDs configuration for compatibility with CLI
        self.gpu_ids = model_config.get("gpu_ids")
        if self.gpu_ids and torch.cuda.is_available():
            # For multi-GPU setup, use device_map="auto" to distribute across all GPUs
            if len(self.gpu_ids) > 1:
                self.device = torch.device("cuda")  # Use default CUDA device
                self.logger.info(f"Multi-GPU setup detected: {self.gpu_ids}")
            else:
                # Single GPU setup
                self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
        else:
            self.device = self._resolve_device(model_config.get("device", "auto"))
        self.default_generation = self._build_default_generation(model_config)

        self.tokenizer: AutoTokenizer
        self.model: AutoModelForCausalLM
        self.input_device: torch.device

        self._load()

    def _resolve_device(
        self, device_config: Union[str, torch.device]
    ) -> torch.device:
        """Resolve device configuration into a torch.device."""
        if device_config == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        try:
            device = torch.device(device_config)
        except (TypeError, RuntimeError):
            self.logger.warning(
                "Invalid device '%s'. Falling back to CPU.", device_config
            )
            return torch.device("cpu")

        if device.type.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning(
                "CUDA requested but not available. Falling back to CPU."
            )
            return torch.device("cpu")

        return device

    def _build_default_generation(
        self, config: Dict[str, Any]
    ) -> GenerationParameters:
        """Build default generation parameters from configuration."""
        gen_config = config.get("generation_config", {})

        return GenerationParameters(
            max_new_tokens=gen_config.get(
                "max_new_tokens", config.get("max_new_tokens", 512)
            ),
            temperature=gen_config.get(
                "temperature", config.get("temperature", 0.0)
            ),
            top_p=gen_config.get("top_p", config.get("top_p", 1.0)),
            num_return_sequences=gen_config.get("num_return_sequences", 1),
            num_beams=gen_config.get("num_beams", 1),
            use_beam_search=gen_config.get(
                "use_beam_search", config.get("use_beam_search", False)
            ),
            seed=gen_config.get("seed", config.get("seed")),
        )

    def _load(self) -> None:
        """Load tokenizer and model.
        
        Raises:
            HuggingFaceModelLoadError: If tokenizer or model loading fails.
        """
        self.logger.info("Loading Hugging Face model from %s", self.model_path)

        # Set CUDA_VISIBLE_DEVICES if specific GPUs are requested
        if self.gpu_ids and torch.cuda.is_available():
            gpu_ids_str = ",".join(map(str, self.gpu_ids))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
            self.logger.info("Set CUDA_VISIBLE_DEVICES to: %s", gpu_ids_str)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
            )
        except Exception as e:
            raise HuggingFaceModelLoadError(f"Failed to load tokenizer: {e}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": (
                torch.float16 if self.device.type == "cuda" else torch.float32
            ),
        }

        # Handle device mapping based on configuration
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
        elif self.gpu_ids and len(self.gpu_ids) > 0:
            if len(self.gpu_ids) > 1:
                # Multi-GPU setup: use automatic device mapping
                model_kwargs["device_map"] = "auto"
                self.logger.info("Using automatic device mapping across GPUs: %s", self.gpu_ids)
            else:
                # Single GPU setup
                model_kwargs["device_map"] = f"cuda:{self.gpu_ids[0]}"
                self.logger.info("Using single GPU: %s", self.gpu_ids[0])

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs,
            )
        except Exception as e:
            raise HuggingFaceModelLoadError(f"Failed to load model: {e}")

        # Move model to device if no device_map was used
        if "device_map" not in model_kwargs:
            self.model.to(self.device)

        self.model.eval()

        if "device_map" in model_kwargs or self.device_map is not None:
            self.input_device = next(self.model.parameters()).device
        else:
            self.input_device = self.device

        self.logger.info("Model loaded on device %s", self.input_device.type)

    def _set_seed(self, seed: Optional[int]) -> None:
        """Set random seeds for reproducibility."""
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_generation_parameters(
        self, overrides: Dict[str, Any]
    ) -> GenerationParameters:
        """Create generation parameters with overrides applied."""
        params = replace(self.default_generation)
        allowed_fields = set(GenerationParameters.__dataclass_fields__.keys())

        for key, value in overrides.items():
            if key in allowed_fields and value is not None:
                setattr(params, key, value)

        return params

    def _build_generation_config(
        self, params: GenerationParameters
    ) -> GenerationConfig:
        """Construct transformers.GenerationConfig from parameters."""
        config_kwargs: Dict[str, Any] = {
            "max_length": self.max_length,
            "max_new_tokens": params.max_new_tokens,
            "num_return_sequences": params.num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if params.seed is not None:
            config_kwargs["seed"] = params.seed

        if params.use_beam_search or params.num_beams > 1:
            config_kwargs.update(
                {
                    "num_beams": max(
                        params.num_beams, params.num_return_sequences
                    ),
                    "do_sample": False,
                }
            )
        elif params.temperature == 0.0 or params.top_p >= 1.0:
            config_kwargs["do_sample"] = False
        else:
            config_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": params.temperature,
                    "top_p": params.top_p,
                }
            )

        return GenerationConfig(**config_kwargs)

    def _prepare_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Tokenize prompt and move tensors to the correct device."""
        if (
            self.is_chat_model
            and hasattr(self.tokenizer, "apply_chat_template")
        ):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            tokenized = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
        else:
            tokenized = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )

        return {
            key: value.to(self.input_device) for key, value in tokenized.items()
        }

    def _decode_outputs(
        self,
        outputs: torch.Tensor,
        prompt: str,
        input_length: int,
    ) -> List[str]:
        """Decode generated token ids into text outputs."""
        if self.is_chat_model:
            generated_ids = outputs[:, input_length:]
            return self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )

        decoded_sequences = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

        completions: List[str] = []
        for text in decoded_sequences:
            if text.startswith(prompt):
                completions.append(text[len(prompt) :].strip())
            else:
                completions.append(text.strip())

        return completions

    def generate(self, prompt: str, **kwargs: Any) -> Union[str, List[str]]:
        """Generate text for a single prompt.
        
        Args:
            prompt: Input prompt for text generation.
            **kwargs: Additional generation parameters to override defaults.
                Supported: max_new_tokens, temperature, top_p, num_return_sequences,
                num_beams, use_beam_search, seed.
                
        Returns:
            Generated text response(s) from the model. Returns a single string
            if num_return_sequences=1, otherwise returns a list of strings.
            
        Raises:
            HuggingFaceGenerationError: If text generation fails.
        """
        params = self._create_generation_parameters(kwargs)
        self._set_seed(params.seed)

        inputs = self._prepare_inputs(prompt)
        input_length = inputs["input_ids"].shape[1]
        generation_config = self._build_generation_config(params)

        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        except Exception as e:
            raise HuggingFaceGenerationError(f"Failed to generate text: {e}")

        completions = self._decode_outputs(outputs, prompt, input_length)

        if params.num_return_sequences == 1:
            return completions[0] if completions else ""

        return completions

    def batch_generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """Generate text for a list of prompts sequentially.
        
        Args:
            prompts: List of input prompts for text generation.
            **kwargs: Additional generation parameters to override defaults.
                Same parameters as generate() method.
                
        Returns:
            List of generated text responses, one for each input prompt.
            
        Raises:
            HuggingFaceGenerationError: If text generation fails for any prompt.
        """
        if not prompts:
            return []

        params = self._create_generation_parameters(kwargs)
        results: List[str] = []

        for index, prompt in enumerate(prompts):
            prompt_kwargs = dict(kwargs)

            if params.seed is not None:
                prompt_kwargs["seed"] = params.seed + index

            output = self.generate(prompt, **prompt_kwargs)

            if isinstance(output, list):
                results.append(output[0] if output else "")
            else:
                results.append(output)

        return results

