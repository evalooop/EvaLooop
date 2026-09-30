"""Abstract base class for evaluation loop tasks."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTask(ABC):
    """One step in an evaluation loop.

    The loop runner (evaloop.evaluation.cycle.BatchEvaluationCycle) interacts
    with tasks only through this interface: create_prompt -> model ->
    extract_output -> postprocess. Subclasses must never be special-cased by
    name anywhere outside their own module.
    """

    #: True if this task's output is runnable code that the evaluator should
    #: functionally test (used to pick which task's output is run against the
    #: dataset's test cases).
    produces_testable_code: bool = False

    def __init__(self, task_config: Dict[str, Any]):
        """Initialize task with configuration."""
        self.task_config = task_config

    @abstractmethod
    def create_prompt(self, input_text: str) -> str:
        """Build the model prompt for this task from the previous step's output."""

    @abstractmethod
    def extract_output(self, raw_response: str) -> str:
        """Extract this task's output from the raw model response."""

    def postprocess(self, output: str, context: Dict[str, Any]) -> str:
        """Hook run after extract_output. Default: identity.

        Context keys provided by the loop runner:
          - "cycle_input": the text the first task of the current cycle
            received (e.g. the docstring-style prompt at cycle start)
          - "task_id": dataset task id
          - "cycle_number": current cycle number
        """
        return output
