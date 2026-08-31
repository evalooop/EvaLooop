"""Unit tests for the task registry and factory."""

import pytest

from evaloop.tasks import (
    BaseTask,
    CodeGenerationTask,
    CodeSummarizationTask,
    CodeTranslationTask,
    TaskFactory,
    register_task,
)


class TestBuiltinRegistrations:
    def test_builtin_types_registered(self):
        types = TaskFactory().list_supported_types()
        assert {"code_generation", "code_summarization", "code_translation"} <= set(types)

    @pytest.mark.parametrize(
        "task_type,expected_cls",
        [
            ("code_generation", CodeGenerationTask),
            ("code_summarization", CodeSummarizationTask),
            ("code_translation", CodeTranslationTask),
        ],
    )
    def test_create_task_returns_right_class(self, task_type, expected_cls):
        task = TaskFactory().create_task({"type": task_type})
        assert isinstance(task, expected_cls)

    def test_create_task_is_case_insensitive(self):
        task = TaskFactory().create_task({"type": "Code_Generation"})
        assert isinstance(task, CodeGenerationTask)

    def test_unknown_type_raises_with_available_types(self):
        with pytest.raises(ValueError) as excinfo:
            TaskFactory().create_task({"type": "nonexistent"})
        assert "nonexistent" in str(excinfo.value)
        assert "code_generation" in str(excinfo.value)

    def test_missing_type_key_raises(self):
        with pytest.raises(ValueError):
            TaskFactory().create_task({})


class TestRegisterTask:
    def test_register_and_create_new_task(self, clean_registry):
        @register_task("my_custom_task")
        class MyCustomTask(BaseTask):
            def create_prompt(self, input_text):
                return f"custom: {input_text}"

            def extract_output(self, raw_response):
                return raw_response

        factory = TaskFactory()
        assert "my_custom_task" in factory.list_supported_types()
        task = factory.create_task({"type": "my_custom_task"})
        assert isinstance(task, MyCustomTask)
        assert task.create_prompt("x") == "custom: x"

    def test_registration_key_is_lowercased(self, clean_registry):
        @register_task("MiXeD_CaSe")
        class MixedCaseTask(BaseTask):
            def create_prompt(self, input_text):
                return input_text

            def extract_output(self, raw_response):
                return raw_response

        assert "mixed_case" in TaskFactory().list_supported_types()

    def test_duplicate_registration_raises(self, clean_registry):
        with pytest.raises(ValueError):
            @register_task("code_generation")
            class DuplicateTask(BaseTask):
                def create_prompt(self, input_text):
                    return input_text

                def extract_output(self, raw_response):
                    return raw_response

    def test_non_basetask_class_raises(self, clean_registry):
        with pytest.raises(TypeError):
            @register_task("not_a_task")
            class NotATask:
                pass
