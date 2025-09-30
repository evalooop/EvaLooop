"""Factory for creating task instances."""

import logging
from typing import Dict, Any

from .code_generation import CodeGenerationTask
from .code_summarization import CodeSummarizationTask
from .code_translation import CodeTranslationTask


class TaskFactory:
    """Factory class for creating task instances."""
    
    def __init__(self):
        """Initialize the task factory."""
        self.logger = logging.getLogger(__name__)
    
    def create_task(self, task_config: Dict[str, Any]):
        """
        Create a task instance based on configuration.
        
        Args:
            task_config: Configuration dictionary for the task.
            
        Returns:
            Task instance.
            
        Raises:
            ValueError: If task type is not supported.
        """
        task_type = task_config.get("type", "").lower()
        
        self.logger.debug(f"Creating task with config: {task_config}")
        
        if task_type == "code_generation":
            return CodeGenerationTask(task_config)
        elif task_type == "code_summarization":
            return CodeSummarizationTask(task_config)
        elif task_type == "code_translation":
            return CodeTranslationTask(task_config)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def list_supported_types(self) -> list:
        """
        List all supported task types.
        
        Returns:
            List of supported task type strings.
        """
        return [
            "code_generation",
            "code_summarization", 
            "code_translation"
        ]
