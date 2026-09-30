"""End-to-end tests for BatchEvaluationCycle with mocked models (no API calls)."""

import pytest

from evaloop.evaluation.cycle import BatchEvaluationCycle
from evaloop.tasks import BaseTask, TaskFactory

from conftest import (
    ASSERT_LINE,
    GENERATION_CONFIG,
    MBPP_STYLE_PROMPT,
    SUMMARIZATION_CONFIG,
)

MOCK_CODE = "def similar_elements(a, b):\n    return tuple(set(a) & set(b))"
MOCK_SUMMARY = "write a python function to find the shared elements of two lists"


class MockModel:
    """Sequential-only model: generation prompts get code, others a summary."""

    def generate(self, prompt: str) -> str:
        if prompt.startswith("Generate Python code"):
            return f"```python\n{MOCK_CODE}\n```"
        return MOCK_SUMMARY


class MockBatchModel(MockModel):
    """Model exposing batch_generate, driving the batch code path."""

    def batch_generate(self, prompts):
        return [self.generate(p) for p in prompts]


class RecordingModel(MockModel):
    """Records every prompt it is asked to generate for."""

    def __init__(self):
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return super().generate(prompt)


class FailingTask(BaseTask):
    def create_prompt(self, input_text):
        raise RuntimeError("boom")

    def extract_output(self, raw_response):
        return raw_response


@pytest.fixture
def tasks():
    factory = TaskFactory()
    return [
        factory.create_task(GENERATION_CONFIG),
        factory.create_task(SUMMARIZATION_CONFIG),
    ]


def _prompt_inputs(n):
    return [{"task_id": f"Mbpp/{i}", "prompt": MBPP_STYLE_PROMPT} for i in range(n)]


class TestSequentialPath:
    def test_single_cycle_outputs(self, tasks):
        cycle = BatchEvaluationCycle(MockModel(), tasks)
        result = cycle.run_single_cycle(_prompt_inputs(1), cycle_number=1)

        assert result["cycle_number"] == 1
        (res,) = result["cycle_results"]
        assert res["task_id"] == "Mbpp/0"
        assert res["task_1_output"] == MOCK_CODE
        # The summary is wrapped as a docstring and retains the original assert line
        assert res["task_2_output"] == f'"""\n{MOCK_SUMMARY}\n{ASSERT_LINE}\n"""\n'
        assert not any(key.startswith("_") for key in res)

    def test_task_error_recorded_and_loop_breaks(self):
        cycle = BatchEvaluationCycle(MockModel(), [FailingTask({})])
        result = cycle.run_single_cycle(_prompt_inputs(1), cycle_number=1)

        (res,) = result["cycle_results"]
        assert "boom" in res["task_1_error"]
        assert "task_1_output" not in res

    def test_output_chains_between_tasks(self, tasks):
        model = RecordingModel()
        cycle = BatchEvaluationCycle(model, tasks)
        result = cycle.run_single_cycle(_prompt_inputs(1), cycle_number=1)

        (res,) = result["cycle_results"]
        # The summarization prompt must be built from the generation output
        assert MOCK_CODE in model.prompts[1]
        assert res["task_1_raw_output"] == f"```python\n{MOCK_CODE}\n```"


class TestBatchPath:
    def test_batch_matches_sequential(self, tasks):
        # The batch path only engages with more than one prompt
        sequential = BatchEvaluationCycle(MockModel(), tasks)
        batch = BatchEvaluationCycle(MockBatchModel(), tasks)
        assert not sequential.supports_batch
        assert batch.supports_batch

        seq_result = sequential.run_single_cycle(_prompt_inputs(2), cycle_number=1)
        batch_result = batch.run_single_cycle(_prompt_inputs(2), cycle_number=1)

        for seq_res, batch_res in zip(
            seq_result["cycle_results"], batch_result["cycle_results"]
        ):
            assert batch_res["task_1_output"] == seq_res["task_1_output"]
            assert batch_res["task_2_output"] == seq_res["task_2_output"]

    def test_batch_failure_marks_all_prompts(self, tasks):
        class BrokenBatchModel(MockBatchModel):
            def batch_generate(self, prompts):
                raise RuntimeError("batch boom")

        cycle = BatchEvaluationCycle(BrokenBatchModel(), tasks)
        result = cycle.run_single_cycle(_prompt_inputs(2), cycle_number=1)

        for res in result["cycle_results"]:
            assert "batch boom" in res["task_1_error"]
            assert not any(key.startswith("_") for key in res)
