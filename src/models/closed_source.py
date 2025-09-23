import openai
from openai import OpenAI
from typing import Dict, Any, Optional
import logging
import os
import time

from .base import BaseLLM


# class OpenAILLM(BaseLLM):
#     """Implementation for OpenAI models like ChatGPT."""
    
#     def __init__(self, model_config: Dict[str, Any]):
#         super().__init__(model_config)
#         self.api_key = model_config.get("api_key")
#         self.model_id = model_config.get("model_id", "gpt-3.5-turbo")
#         self.max_tokens = model_config.get("max_tokens", 4096)
        
#         # Set default to greedy decoding
#         self.temperature = model_config.get("temperature", 0.0)  # 0.0 = greedy decoding
#         self.top_p = model_config.get("top_p", 1.0)  # 1.0 = no nucleus sampling
        
#         # Create client in init to avoid recreating for each request
#         self.client = OpenAI(api_key=self.api_key)
        
#     def generate(self, prompt: str, **kwargs) -> str:
#         """Generate text using the OpenAI API."""
#         # Default to greedy decoding but allow overrides
#         max_tokens = kwargs.get("max_tokens", self.max_tokens)
#         temperature = kwargs.get("temperature", self.temperature)
#         top_p = kwargs.get("top_p", self.top_p)
        
#         # Filter out parameters that are explicitly handled
#         filtered_kwargs = {k: v for k, v in kwargs.items() 
#                          if k not in ["max_tokens", "temperature", "top_p"]}
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model_id,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful AI assistant."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 max_tokens=max_tokens,
#                 temperature=temperature,
#                 top_p=top_p,
#                 **filtered_kwargs
#             )
            
#             return response.choices[0].message.content
#         except Exception as e:
#             # Log error and return error information
#             import logging
#             logger = logging.getLogger(__name__)
#             logger.error(f"OpenAI API error: {str(e)}")
#             return f"Error generating response: {str(e)}"
class OpenAILLM(BaseLLM):
    """Implementation for OpenAI models like ChatGPT."""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.api_key = model_config.get("api_key")
        self.model_id = model_config.get("model_id", "gpt-3.5-turbo")
        self.max_tokens = model_config.get("max_tokens", 4096)
        
        # Set default to greedy decoding
        self.temperature = model_config.get("temperature", 0.0)  # 0.0 = greedy decoding
        self.top_p = model_config.get("top_p", 1.0)  # 1.0 = no nucleus sampling
        
        # Hardcoded retry parameters
        self.max_retries = 10  # Maximum number of retry attempts
        self.retry_delay = 3  # Retry delay in seconds
        
        # Create client in init to avoid recreating for each request
        self.client = OpenAI(api_key=self.api_key)
    
    def _uses_system_role(self, model_id: str) -> bool:
        """
        Determine if a model supports system role messages.
        
        Args:
            model_id: The model identifier
            
        Returns:
            bool: True if model supports system role, False otherwise
        """
        # Models that don't support system role
        no_system_role_models = ["o1-", "o3-", "o4-"]
        
        # Check if model ID contains any of the prefixes for models that don't support system role
        for prefix in no_system_role_models:
            if prefix in model_id.lower():
                return False
        
        # Default to using system role for other models (GPT-3.5, GPT-4, etc.)
        return True
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the OpenAI API with retry logic."""
        # Default to greedy decoding but allow overrides
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        model_id = kwargs.get("model_id", self.model_id)  # Allow model ID override
        
        # Filter out parameters that are explicitly handled
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ["max_tokens", "temperature", "top_p", "model_id"]}
        
        # Initialize logger
        logger = logging.getLogger(__name__)
        
        retry_count = 0
        last_error = None
        
        # Determine if we should use system role for this model
        use_system_role = self._uses_system_role(model_id)
        if not use_system_role:
            logger.info(f"Model {model_id} doesn't support parameter max_tokens. Using max_completion_tokens instead of.")
        
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
                        print("run o1-mini without system role, this is required for o1-mini")
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
                        f"Rate limit error, retrying ({retry_count}/{self.max_retries}) with {self.retry_delay} seconds delay. Error: {str(e)}"
                    )
                    time.sleep(self.retry_delay)
                    last_error = e
                else:
                    logger.error(f"OpenAI API rate limit error: {str(e)}")
                    return f"Error generating response: {str(e)}"
            
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
                logger.error(f"OpenAI API bad request error: {str(e)}")
                return f"Error generating response: {str(e)}"
                    
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
                            f"Model generation error, retrying ({retry_count}/{self.max_retries}) with {self.retry_delay} seconds delay. Error: {str(e)}"
                        )
                        time.sleep(self.retry_delay)
                        last_error = e
                    else:
                        logger.error(f"Model generation error still occurred after multiple attempts: {str(e)}")
                        return f"Error generating response after maximum retries: {str(e)}"
                else:
                    # Non-retryable API errors
                    logger.error(f"OpenAI API error (non-retryable): {str(e)}")
                    return f"Error generating response: {str(e)}"
                    
            except Exception as e:
                # Handle other unforeseen errors
                logger.error(f"Unexpected error occurred while generating response: {str(e)}")
                return f"Error generating response: {str(e)}"
        
        # If all retry attempts have been exhausted
        logger.error(f"All retry attempts exhausted, last error: {str(last_error)}")
        return f"Error generating response after maximum retries: {str(last_error)}"