"""Shared fixtures for the EvaLoop test suite.

The whole suite runs with only pytest + tqdm installed: no torch, no vllm,
no evalplus, and no API calls.
"""

import pytest

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
