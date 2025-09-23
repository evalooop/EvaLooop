from typing import Dict, Any

class CodeTranslationTask:
    """Task for translating code between programming languages."""
    
    def __init__(self, task_config: Dict[str, Any]):
        """Initialize task with configuration."""
        self.task_config = task_config
        self.source_language = task_config.get("source_language", "python")
        self.target_language = task_config.get("target_language", "java")
        
    def create_prompt(self, code: str) -> str:
        """Create a prompt for code translation."""
        prompt_template = self.task_config.get("prompt_template", 
            f"Translate the following {self.source_language} code to {self.target_language}:\n\n"
            "```\n{code}\n```\n\n"
            f"Return only the {self.target_language} code, without explanations or comments."
        )
        return prompt_template.format(code=code)
    
    def extract_code(self, llm_response: str) -> str:
        """Extract translated code from LLM response."""
        # Similar extraction to code generation
        import re
        pattern = r"```(?:\w+)?\s*([\s\S]*?)```"
        matches = re.findall(pattern, llm_response)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks found, return the entire response
        return llm_response.strip()
