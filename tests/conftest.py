"""Shared fixtures for the EvaLoop test suite."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from evaloop.models.base import BaseLLM


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
