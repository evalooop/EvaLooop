"""Task types for evaluation loops.

Importing this package registers the built-in tasks. To add a new task type,
create a module here with a BaseTask subclass decorated with
@register_task("your_type") and import it below.
"""

from .base import BaseTask
from .factory import TaskFactory, register_task
from .code_generation import CodeGenerationTask
from .code_summarization import CodeSummarizationTask
from .code_translation import CodeTranslationTask

__all__ = [
    "BaseTask",
    "TaskFactory",
    "register_task",
    "CodeGenerationTask",
    "CodeSummarizationTask",
    "CodeTranslationTask",
]
