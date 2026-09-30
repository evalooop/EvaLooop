"""Shared fixtures for the EvaLoop test suite.

The whole suite runs with only pytest + tqdm installed: no torch, no vllm,
no evalplus, and no API calls.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from evaloop.models.base import BaseLLM
from evaloop.tasks import factory as task_factory

# Mirrors the MBPP+ prompt layout used by the real dataset loader: the
# second-to-last line is an assert statement (this is what the
# summarization postprocess re-attaches each cycle).
MBPP_STYLE_PROMPT = (
    '"""\n'
    "Write a function to find the shared elements from the given two lists.\n"
    "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n"
    '"""\n'
)

ASSERT_LINE = "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))"

# The real task configs produced by EvaluationConfig.get_task_configs()
GENERATION_CONFIG = {
    "type": "code_generation",
    "language": "python",
    "prompt_template": (
        "Generate Python code for the following task: \n{description}\n"
        "Return only the code, without explanations or comments."
    ),
}

SUMMARIZATION_CONFIG = {
    "type": "code_summarization",
    "prompt_template": (
        "Use one sentence to summarize the following code and start with "
        "'write a python function to':\n\n```\n{code}\n```\n\n"
        "```\nwrite a python function to\n```"
    ),
}


class FakeLLM(BaseLLM):
    """Offline stand-in for a judge model that returns canned responses."""

    def __init__(self, responses: Optional[List[str]] = None, default: str = "0.5"):
        super().__init__({"name": "fake-judge"})
        self.responses = list(responses or [])
        self.default = default
        self.prompts: List[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else self.default


def make_prompt_result(task_id: str, sustained: int, max_cycles: int) -> Dict[str, Any]:
    """Builds a results entry whose code passes ``sustained`` loops and then fails."""
    cycles = []
    for index in range(min(sustained + 1, max_cycles)):
        loop = index + 1
        cycles.append({
            "cycle": loop,
            "task_id": task_id,
            "task_1_output": f"code-{task_id}-{loop}",
            "task_2_output": f"summary-{task_id}-{loop}",
            "success": index < sustained,
        })
    return {
        "task_id": task_id,
        "initial_prompt": f"prompt-{task_id}",
        "cycles": cycles,
        "successful_cycles": sustained,
        "max_cycles_reached": sustained == max_cycles,
    }


@pytest.fixture
def generation_config():
    return dict(GENERATION_CONFIG)


@pytest.fixture
def summarization_config():
    return dict(SUMMARIZATION_CONFIG)


@pytest.fixture
def mbpp_prompt():
    return MBPP_STYLE_PROMPT


@pytest.fixture
def clean_registry():
    """Snapshot the task registry and restore it after the test."""
    snapshot = dict(task_factory._TASK_REGISTRY)
    yield task_factory._TASK_REGISTRY
    task_factory._TASK_REGISTRY.clear()
    task_factory._TASK_REGISTRY.update(snapshot)


@pytest.fixture
def results_file(tmp_path: Path) -> Path:
    """Writes a results file with M = 3 and sustained loops [0, 1, 2, 2, 3]."""
    prompt_results = [
        make_prompt_result(f"Mbpp/{index}", sustained, 3)
        for index, sustained in enumerate([0, 1, 2, 2, 3])
    ]
    path = tmp_path / "exp_results.json"
    path.write_text(json.dumps([{"model": "toy-model", "max_cycles": 3, "prompt_results": prompt_results}]))
    return path


@pytest.fixture(autouse=True)
def _no_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeps tests offline by hiding any real OpenAI credentials."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
