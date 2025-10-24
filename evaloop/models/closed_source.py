"""OpenAI API-based model implementation."""

import logging
import os
import time
from typing import Any, Dict, Optional

import openai
from openai import OpenAI

from .base import BaseLLM


logger = logging.getLogger(__name__)


class OpenAIAPIError(Exception):
    """Base exception for OpenAI API related errors."""
    pass


class OpenAIAPIRateLimitError(OpenAIAPIError):
    """Raised when OpenAI API rate limit is exceeded."""
    pass


class OpenAIAPIModelError(OpenAIAPIError):
    """Raised when OpenAI model is unavailable or overloaded."""
    pass


class OpenAILLM(BaseLLM):
    """Implementation for OpenAI models like ChatGPT.
    
    This class provides a wrapper around the OpenAI API for text generation,
    with built-in retry logic, error handling, and support for different
    model configurations.
    
    Attributes:
        api_key: OpenAI API key for authentication.
        model_id: Model identifier (e.g., "gpt-4", "gpt-3.5-turbo").
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature for generation.
        top_p: Nucleus sampling parameter.
        max_retries: Maximum number of retry attempts for failed requests.
        retry_delay: Delay between retry attempts in seconds.
        client: OpenAI client instance.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize OpenAI model with configuration.
        
        Args:
            model_config: Configuration dictionary containing model parameters.
                Expected keys: api_key, model_id, max_tokens, temperature, top_p.
                
        Raises:
            ValueError: If required configuration parameters are missing.
            OpenAIAPIError: If client initialization fails.
        """
        super().__init__(model_config)
        
        # Validate required configuration
        self.api_key = model_config.get("api_key")
        if not self.api_key:
            raise ValueError("API key is required for OpenAI models")
            
        self.model_id = model_config.get("model_id", "gpt-3.5-turbo")
        self.max_tokens = model_config.get("max_tokens", 4096)
        
        # Set default to greedy decoding
        self.temperature = model_config.get("temperature", 0.0)
        self.top_p = model_config.get("top_p", 1.0)
        
        # Retry configuration
        self.max_retries = model_config.get("max_retries", 10)
        self.retry_delay = model_config.get("retry_delay", 3)
        
        # Create client
        try:
            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            raise OpenAIAPIError(f"Failed to initialize OpenAI client: {e}")
    
    def _uses_system_role(self, model_id: str) -> bool:
        """Determine if a model supports system role messages.
        
        Args:
            model_id: The model identifier.
            
        Returns:
            True if model supports system role, False otherwise.
        """
        # Models that don't support system role
        no_system_role_models = ["o1-", "o3-", "o4-"]
        
        # Check if model ID contains any of the prefixes for models that don't support system role
        for prefix in no_system_role_models:
            if prefix in model_id.lower():
                return False
        
        # Default to using system role for other models (GPT-3.5, GPT-4, etc.)
        return True
        
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using the OpenAI API with retry logic.
        
        Args:
            prompt: Input prompt for text generation.
            **kwargs: Additional generation parameters to override defaults.
                Supported: max_tokens, temperature, top_p, model_id.
                
        Returns:
            Generated text response from the model.
            
        Raises:
            OpenAIAPIRateLimitError: When API rate limit is exceeded.
            OpenAIAPIModelError: When model is unavailable or overloaded.
            OpenAIAPIError: For other API-related errors.
        """
        # Default to greedy decoding but allow overrides
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        model_id = kwargs.get("model_id", self.model_id)  # Allow model ID override
        
        # Filter out parameters that are explicitly handled
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ["max_tokens", "temperature", "top_p", "model_id"]}
        
        # Use module-level logger
        
        retry_count = 0
        last_error = None
        
        # Determine if we should use system role for this model
        use_system_role = self._uses_system_role(model_id)
        if not use_system_role:
            logger.info(
                "Model %s doesn't support parameter max_tokens. "
                "Using max_completion_tokens instead.", model_id
            )
        
        while retry_count <= self.max_retries:  # Use <= to ensure we try max_retries+1 times
            try:

                # Prepare messages based on whether system role is supported
                if use_system_role:
                    response = self.client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": "You are a helpful AI assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        **filtered_kwargs
                    )
                else:
                    # For models that don't support system role (o1, o3), use user role only
                    if 'o1-mini' not in model_id:
                        response = self.client.chat.completions.create(
                            model=model_id,
                            messages=[
                                {"role": "system", "content": "You are a helpful AI assistant."},
                                {"role": "user", "content": prompt}
                            ],
                            max_completion_tokens=max_tokens,
                        )
                    else:
                        logger.info(
                            "Running o1-mini without system role, "
                            "this is required for o1-mini"
                        )
                        response = self.client.chat.completions.create(
                            model=model_id,
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            max_completion_tokens=max_tokens,
                        )

                # Check if we received an empty response
                content = response.choices[0].message.content
                if not content or content.strip() == "":
                    if retry_count < self.max_retries:
                        retry_count += 1
                        logger.warning(
                            f"Received empty response, retrying ({retry_count}/{self.max_retries}) with {self.retry_delay} seconds delay."
                        )
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        return "Error: Still received empty response after multiple attempts."
                
                # If response was successfully generated, return content
                return content
                
            except openai.RateLimitError as e:
                # Handle rate limit errors
                if retry_count < self.max_retries:
                    retry_count += 1
                    logger.warning(
                        "Rate limit error, retrying (%d/%d) with %d seconds delay. "
                        "Error: %s", retry_count, self.max_retries, 
                        self.retry_delay, str(e)
                    )
                    time.sleep(self.retry_delay)
                    last_error = OpenAIAPIRateLimitError(str(e))
                else:
                    logger.error("OpenAI API rate limit error: %s", str(e))
                    raise OpenAIAPIRateLimitError(f"Rate limit exceeded: {e}")
            
            except openai.BadRequestError as e:
                error_msg = str(e).lower()
                
                # Check if the error is related to unsupported system role
                if ("unsupported value" in error_msg and 
                    "role" in error_msg and 
                    "system" in error_msg and 
                    use_system_role):
                    
                    # Switch to not using system role and retry immediately
                    use_system_role = False
                    logger.info(f"Model {model_id} doesn't support system role. Retrying with user role only.")
                    continue
                
                # Other bad request errors
                logger.error("OpenAI API bad request error: %s", str(e))
                raise OpenAIAPIError(f"Bad request: {e}")
                    
            except (openai.APITimeoutError, openai.APIError) as e:
                # Handle API timeout and general API errors
                error_str = str(e).lower()
                
                # Check if error is related to model generation
                model_error_indicators = [
                    "model", "overloaded", "unavailable", "capacity", 
                    "please retry", "try again", "502", "503", "504"
                ]
                
                if any(indicator in error_str for indicator in model_error_indicators):
                    if retry_count < self.max_retries:
                        retry_count += 1
                        logger.warning(
                            "Model generation error, retrying (%d/%d) with %d seconds delay. "
                            "Error: %s", retry_count, self.max_retries, 
                            self.retry_delay, str(e)
                        )
                        time.sleep(self.retry_delay)
                        last_error = OpenAIAPIModelError(str(e))
                    else:
                        logger.error(
                            "Model generation error still occurred after multiple attempts: %s", 
                            str(e)
                        )
                        raise OpenAIAPIModelError(f"Model unavailable after retries: {e}")
                else:
                    # Non-retryable API errors
                    logger.error("OpenAI API error (non-retryable): %s", str(e))
                    raise OpenAIAPIError(f"API error: {e}")
                    
            except Exception as e:
                # Handle other unforeseen errors
                logger.error("Unexpected error occurred while generating response: %s", str(e))
                raise OpenAIAPIError(f"Unexpected error: {e}")
        
        # If all retry attempts have been exhausted
        logger.error("All retry attempts exhausted, last error: %s", str(last_error))
        raise OpenAIAPIError(f"All retries exhausted: {last_error}")