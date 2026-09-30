"""Registry-based factory for creating task instances.

New task types register themselves with the @register_task decorator:

    from evaloop.tasks import BaseTask, register_task

    @register_task("my_task")
    class MyTask(BaseTask):
        ...

and are then created via ``TaskFactory().create_task({"type": "my_task", ...})``.
"""

import logging
from typing import Any, Callable, Dict, List, Type

from .base import BaseTask

_TASK_REGISTRY: Dict[str, Type[BaseTask]] = {}


def register_task(task_type: str) -> Callable[[Type[BaseTask]], Type[BaseTask]]:
    """Class decorator registering a BaseTask subclass under a task type name.

    Args:
        task_type: The config "type" string (case-insensitive).

    Raises:
        TypeError: If the decorated class is not a BaseTask subclass.
        ValueError: If the task type is already registered.
    """
    def decorator(cls: Type[BaseTask]) -> Type[BaseTask]:
        key = task_type.lower()
        if not (isinstance(cls, type) and issubclass(cls, BaseTask)):
            raise TypeError(f"{cls!r} must subclass BaseTask to be registered")
        if key in _TASK_REGISTRY:
            raise ValueError(f"Task type '{key}' is already registered")
        _TASK_REGISTRY[key] = cls
        return cls
    return decorator


class TaskFactory:
    """Factory class for creating task instances from the registry."""

    def __init__(self):
        """Initialize the task factory."""
        self.logger = logging.getLogger(__name__)

    def create_task(self, task_config: Dict[str, Any]) -> BaseTask:
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

        task_cls = _TASK_REGISTRY.get(task_type)
        if task_cls is None:
            raise ValueError(
                f"Unsupported task type: '{task_type}'. "
                f"Available types: {', '.join(sorted(_TASK_REGISTRY))}"
            )
        return task_cls(task_config)

    def list_supported_types(self) -> List[str]:
        """
        List all supported task types.

        Returns:
            Sorted list of registered task type strings.
        """
        return sorted(_TASK_REGISTRY)
