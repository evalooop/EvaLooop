from typing import Dict, Any

class CodeGenerationTask:
    """Task for generating code from natural language descriptions."""
    
    def __init__(self, task_config: Dict[str, Any]):
        """Initialize task with configuration."""
        self.task_config = task_config
        self.language = task_config.get("language", "python")
        
    def create_prompt(self, nl_description: str) -> str:
        """Create a prompt for code generation."""
        prompt_template = self.task_config.get("prompt_template", 
            f"Generate {self.language} code for the following task: {{description}}\n\n"
            f"Return only the code, without explanations or comments.\n\n"
        )
        return prompt_template.format(description=nl_description)
    
    def extract_code(self, llm_response: str) -> str:
        """Extract code from LLM response."""
        # Simple extraction: look for code between triple backticks
        # This could be enhanced with more robust parsing
        import re
        pattern = r"```(?:\w+)?\s*([\s\S]*?)```"
        matches = re.findall(pattern, llm_response)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks found, return the entire response
        return llm_response.strip()
