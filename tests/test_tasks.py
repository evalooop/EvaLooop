"""Unit tests for the task classes and the BaseTask interface."""

from evaloop.tasks import (
    BaseTask,
    CodeGenerationTask,
    CodeSummarizationTask,
    CodeTranslationTask,
)

from conftest import ASSERT_LINE, MBPP_STYLE_PROMPT


class TestCodeGenerationTask:
    def test_create_prompt_uses_template(self, generation_config):
        task = CodeGenerationTask(generation_config)
        prompt = task.create_prompt("reverse a string")
        assert prompt.startswith("Generate Python code for the following task:")
        assert "reverse a string" in prompt

    def test_create_prompt_default_template(self):
        task = CodeGenerationTask({"type": "code_generation"})
        prompt = task.create_prompt("reverse a string")
        assert "reverse a string" in prompt

    def test_extract_output_fenced_block(self, generation_config):
        task = CodeGenerationTask(generation_config)
        response = "Here you go:\n```python\ndef f(x):\n    return x\n```\nEnjoy!"
        assert task.extract_output(response) == "def f(x):\n    return x"

    def test_extract_output_no_fence_returns_stripped_response(self, generation_config):
        task = CodeGenerationTask(generation_config)
        assert task.extract_output("  def f(x): return x  \n") == "def f(x): return x"

    def test_is_testable(self, generation_config):
        assert CodeGenerationTask(generation_config).produces_testable_code is True


class TestCodeSummarizationTask:
    def test_create_prompt_embeds_code(self, summarization_config):
        task = CodeSummarizationTask(summarization_config)
        prompt = task.create_prompt("def f(x): return x")
        assert "def f(x): return x" in prompt
        assert "write a python function to" in prompt

    def test_extract_output_strips_response(self, summarization_config):
        task = CodeSummarizationTask(summarization_config)
        assert task.extract_output("  a summary  \n") == "a summary"

    def test_postprocess_retains_assert_line(self, summarization_config):
        task = CodeSummarizationTask(summarization_config)
        result = task.postprocess(
            "write a python function to find shared elements",
            {"cycle_input": MBPP_STYLE_PROMPT, "task_id": "Mbpp/2", "cycle_number": 1},
        )
        assert result == (
            '"""\nwrite a python function to find shared elements\n'
            + ASSERT_LINE
            + '\n"""\n'
        )

    def test_postprocess_handles_short_cycle_input(self, summarization_config):
        task = CodeSummarizationTask(summarization_config)
        for cycle_input in ("", "one line"):
            result = task.postprocess("summary", {"cycle_input": cycle_input})
            assert result == '"""\nsummary\n\n"""\n'

    def test_postprocess_handles_missing_cycle_input(self, summarization_config):
        task = CodeSummarizationTask(summarization_config)
        assert task.postprocess("summary", {}) == '"""\nsummary\n\n"""\n'

    def test_is_not_testable(self, summarization_config):
        assert CodeSummarizationTask(summarization_config).produces_testable_code is False


class TestCodeTranslationTask:
    def test_create_prompt_mentions_languages(self):
        task = CodeTranslationTask({"type": "code_translation"})
        prompt = task.create_prompt("def f(x): return x")
        assert "python" in prompt
        assert "java" in prompt
        assert "def f(x): return x" in prompt

    def test_extract_output_fenced_block(self):
        task = CodeTranslationTask({"type": "code_translation"})
        response = "```java\nint f(int x) { return x; }\n```"
        assert task.extract_output(response) == "int f(int x) { return x; }"

    def test_is_testable(self):
        assert CodeTranslationTask({"type": "code_translation"}).produces_testable_code is True


class TestBaseTask:
    def test_default_postprocess_is_identity(self):
        class NoOpTask(BaseTask):
            def create_prompt(self, input_text):
                return input_text

            def extract_output(self, raw_response):
                return raw_response

        task = NoOpTask({})
        assert task.postprocess("anything", {"cycle_input": "ignored"}) == "anything"
        assert task.produces_testable_code is False

    def test_cannot_instantiate_abstract(self):
        import pytest

        with pytest.raises(TypeError):
            BaseTask({})
