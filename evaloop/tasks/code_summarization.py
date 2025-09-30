from typing import Dict, Any

class CodeSummarizationTask:
    """Task for summarizing code to natural language descriptions."""
    
    def __init__(self, task_config: Dict[str, Any]):
        """Initialize task with configuration."""
        self.task_config = task_config
        
    def create_prompt(self, code: str) -> str:
        """Create a prompt for code summarization."""
        prompt_template = self.task_config.get("prompt_template", 
            "Summarize what the following code does in a detailed description:\n\n"
            "```\n{code}\n```\n\n"
            "Provide a comprehensive explanation of the code's functionality, inputs, outputs, and any notable algorithms or techniques used."
        )
        return prompt_template.format(code=code)
    
    def extract_summary(self, llm_response: str) -> str:
        """Extract summary from LLM response."""
        # For summarization, we typically want the entire response
        return llm_response.strip()
