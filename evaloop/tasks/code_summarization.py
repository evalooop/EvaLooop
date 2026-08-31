from typing import Dict, Any

from .base import BaseTask
from .factory import register_task


@register_task("code_summarization")
class CodeSummarizationTask(BaseTask):
    """Task for summarizing code to natural language descriptions."""

    def create_prompt(self, input_text: str) -> str:
        """Create a prompt for code summarization."""
        prompt_template = self.task_config.get("prompt_template",
            "Summarize what the following code does in a detailed description:\n\n"
            "```\n{code}\n```\n\n"
            "Provide a comprehensive explanation of the code's functionality, inputs, outputs, and any notable algorithms or techniques used."
        )
        return prompt_template.format(code=input_text)

    def extract_output(self, raw_response: str) -> str:
        """Extract summary from LLM response."""
        # For summarization, we typically want the entire response
        return raw_response.strip()

    def postprocess(self, output: str, context: Dict[str, Any]) -> str:
        """Wrap the summary as a docstring, re-attaching the cycle's assert line.

        Deliberate design decision (see README, "How the loop works"): the
        summary that seeds the next generation step keeps the final assert
        line from the prompt the cycle started with, so the function
        name/signature survives across loops and failures measure semantic
        drift rather than naming drift. The [-2] index matches the MBPP+
        prompt layout, where the last line is the closing quotes and the
        second-to-last is an assert statement.
        """
        try:
            assert_code = context.get("cycle_input", "").splitlines()[-2]
        except IndexError:
            assert_code = ""
        return f"\"\"\"\n{output}\n{assert_code}\n\"\"\"\n"
