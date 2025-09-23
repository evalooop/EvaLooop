#!/usr/bin/env python3
"""
Setup script for LLM Robustness Evaluation project.
This script creates the directory structure and populates files with actual content.
"""

import os
import sys

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directory structure
DIRECTORIES = [
    "config",
    "src",
    "src/models",
    "src/tasks",
    "src/evaluation",
    "src/utils",
    "data",
    "data/prompts",
    "data/test_cases",
    "data/test_cases/python",
    "data/test_cases/java",
    "results",
    "results/plots",
    "scripts",
    "experiments",
    "logs"
]

# File content dictionary - mapping file paths to their content
FILE_CONTENTS = {
    # Base model interface
    "src/models/base.py": """from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseLLM(ABC):
    \"\"\"Base class for all LLM implementations.\"\"\"
    
    def __init__(self, model_config: Dict[str, Any]):
        \"\"\"Initialize the LLM with configuration.\"\"\"
        self.model_config = model_config
        self.model_name = model_config.get("name", "unnamed_model")
        
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        \"\"\"Generate text based on input prompt.\"\"\"
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name})"
""",
    
    # Open source models implementation
    "src/models/open_source.py": """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional

from .base import BaseLLM

class HuggingFaceLLM(BaseLLM):
    \"\"\"Implementation for Hugging Face based models like LLaMA, DeepSeek, etc.\"\"\"
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.device = model_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = model_config.get("max_length", 2048)
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config["path"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["path"],
            trust_remote_code=True
        )
        
    async def generate(self, prompt: str, **kwargs) -> str:
        \"\"\"Generate text using the model.\"\"\"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Override with kwargs if provided
        max_length = kwargs.get("max_length", self.max_length)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                **{k: v for k, v in kwargs.items() if k not in ["max_length", "temperature", "top_p"]}
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response[len(prompt):].strip()
        
        return response
""",
    
    # Closed source models implementation
    "src/models/closed_source.py": """import openai
from typing import Dict, Any, Optional

from .base import BaseLLM

class OpenAILLM(BaseLLM):
    \"\"\"Implementation for OpenAI models like ChatGPT.\"\"\"
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        openai.api_key = model_config.get("api_key")
        self.model_id = model_config.get("model_id", "gpt-3.5-turbo")
        self.max_tokens = model_config.get("max_tokens", 2048)
        
    async def generate(self, prompt: str, **kwargs) -> str:
        \"\"\"Generate text using the OpenAI API.\"\"\"
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", 0.7)
        
        response = await openai.ChatCompletion.acreate(
            model=self.model_id,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature"]}
        )
        
        return response.choices[0].message.content
""",
    
    # Code generation task
    "src/tasks/code_generation.py": """from typing import Dict, Any

class CodeGenerationTask:
    \"\"\"Task for generating code from natural language descriptions.\"\"\"
    
    def __init__(self, task_config: Dict[str, Any]):
        \"\"\"Initialize task with configuration.\"\"\"
        self.task_config = task_config
        self.language = task_config.get("language", "python")
        
    def create_prompt(self, nl_description: str) -> str:
        \"\"\"Create a prompt for code generation.\"\"\"
        prompt_template = self.task_config.get("prompt_template", 
            f"Generate {self.language} code for the following task: {{description}}\\n\\n"
            f"Return only the code, without explanations or comments.\\n\\n"
        )
        return prompt_template.format(description=nl_description)
    
    def extract_code(self, llm_response: str) -> str:
        \"\"\"Extract code from LLM response.\"\"\"
        # Simple extraction: look for code between triple backticks
        # This could be enhanced with more robust parsing
        import re
        pattern = r"```(?:\\w+)?\\s*([\\s\\S]*?)```"
        matches = re.findall(pattern, llm_response)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks found, return the entire response
        return llm_response.strip()
""",
    
    # Code summarization task
    "src/tasks/code_summarization.py": """from typing import Dict, Any

class CodeSummarizationTask:
    \"\"\"Task for summarizing code to natural language descriptions.\"\"\"
    
    def __init__(self, task_config: Dict[str, Any]):
        \"\"\"Initialize task with configuration.\"\"\"
        self.task_config = task_config
        
    def create_prompt(self, code: str) -> str:
        \"\"\"Create a prompt for code summarization.\"\"\"
        prompt_template = self.task_config.get("prompt_template", 
            "Summarize what the following code does in a detailed description:\\n\\n"
            "```\\n{code}\\n```\\n\\n"
            "Provide a comprehensive explanation of the code's functionality, inputs, outputs, and any notable algorithms or techniques used."
        )
        return prompt_template.format(code=code)
    
    def extract_summary(self, llm_response: str) -> str:
        \"\"\"Extract summary from LLM response.\"\"\"
        # For summarization, we typically want the entire response
        return llm_response.strip()
""",
    
    # Code translation task
    "src/tasks/code_translation.py": """from typing import Dict, Any

class CodeTranslationTask:
    \"\"\"Task for translating code between programming languages.\"\"\"
    
    def __init__(self, task_config: Dict[str, Any]):
        \"\"\"Initialize task with configuration.\"\"\"
        self.task_config = task_config
        self.source_language = task_config.get("source_language", "python")
        self.target_language = task_config.get("target_language", "java")
        
    def create_prompt(self, code: str) -> str:
        \"\"\"Create a prompt for code translation.\"\"\"
        prompt_template = self.task_config.get("prompt_template", 
            f"Translate the following {self.source_language} code to {self.target_language}:\\n\\n"
            "```\\n{code}\\n```\\n\\n"
            f"Return only the {self.target_language} code, without explanations or comments."
        )
        return prompt_template.format(code=code)
    
    def extract_code(self, llm_response: str) -> str:
        \"\"\"Extract translated code from LLM response.\"\"\"
        # Similar extraction to code generation
        import re
        pattern = r"```(?:\\w+)?\\s*([\\s\\S]*?)```"
        matches = re.findall(pattern, llm_response)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks found, return the entire response
        return llm_response.strip()
""",
    
    # Code testing functionality
    "src/evaluation/testing.py": """import os
import subprocess
import tempfile
import logging
from typing import Dict, Any, Optional

class CodeTester:
    \"\"\"Tests code functionality by execution with test cases.\"\"\"
    
    def __init__(self, test_cases_dir: str):
        \"\"\"Initialize with directory containing test cases.\"\"\"
        self.test_cases_dir = test_cases_dir
        self.logger = logging.getLogger(__name__)
        
    def test_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        \"\"\"
        Test the given code by executing it.
        
        Args:
            code: The code to test
            language: The programming language of the code
            
        Returns:
            Dict with test results including success/failure and output
        \"\"\"
        if language.lower() == "python":
            return self._test_python_code(code)
        elif language.lower() == "java":
            return self._test_java_code(code)
        else:
            return {"success": False, "error": f"Unsupported language: {language}"}
    
    def _test_python_code(self, code: str) -> Dict[str, Any]:
        \"\"\"Test Python code by executing it with the test cases.\"\"\"
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        
        try:
            # Run the code with a timeout
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Check if execution was successful
            success = result.returncode == 0
            
            test_result = {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
            # Remove the temporary file
            os.unlink(tmp_path)
            
            return test_result
            
        except subprocess.TimeoutExpired:
            # Remove the temporary file
            os.unlink(tmp_path)
            return {"success": False, "error": "Execution timed out"}
            
        except Exception as e:
            # Remove the temporary file
            os.unlink(tmp_path)
            return {"success": False, "error": str(e)}
    
    def _test_java_code(self, code: str) -> Dict[str, Any]:
        \"\"\"Test Java code by compiling and executing it with the test cases.\"\"\"
        # Extract class name for the file name
        import re
        class_match = re.search(r"public\\s+class\\s+(\\w+)", code)
        if not class_match:
            return {"success": False, "error": "Could not find public class name"}
        
        class_name = class_match.group(1)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            java_file = os.path.join(tmp_dir, f"{class_name}.java")
            
            # Write the code to a temporary file
            with open(java_file, "w") as f:
                f.write(code)
            
            try:
                # Compile the code
                compile_result = subprocess.run(
                    ["javac", java_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if compile_result.returncode != 0:
                    return {
                        "success": False,
                        "stage": "compilation",
                        "stdout": compile_result.stdout,
                        "stderr": compile_result.stderr,
                        "returncode": compile_result.returncode
                    }
                
                # Run the compiled code
                run_result = subprocess.run(
                    ["java", "-cp", tmp_dir, class_name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                # Check if execution was successful
                success = run_result.returncode == 0
                
                return {
                    "success": success,
                    "stage": "execution",
                    "stdout": run_result.stdout,
                    "stderr": run_result.stderr,
                    "returncode": run_result.returncode
                }
                
            except subprocess.TimeoutExpired as e:
                stage = "compilation" if "compile_result" not in locals() else "execution"
                return {"success": False, "stage": stage, "error": "Execution timed out"}
                
            except Exception as e:
                return {"success": False, "error": str(e)}
""",
    
    # Evaluation cycle manager
    "src/evaluation/cycle.py": """from typing import Dict, Any, List, Tuple, Optional
import asyncio
import logging

from ..models.base import BaseLLM

class EvaluationCycle:
    \"\"\"Manages evaluation cycles for LLM testing.\"\"\"
    
    def __init__(self, model: BaseLLM, tasks: List[Any], tester):
        \"\"\"Initialize with model, tasks, and testing component.\"\"\"
        self.model = model
        self.tasks = tasks
        self.tester = tester
        self.cycle_results = []
        self.logger = logging.getLogger(__name__)
        
    async def run_cycle(self, initial_input: str, max_cycles: int = 10) -> Dict[str, Any]:
        \"\"\"
        Run the evaluation cycle until failure or max_cycles is reached.
        
        Args:
            initial_input: Starting input (natural language prompt)
            max_cycles: Maximum number of cycles to run
            
        Returns:
            Dict with results including successful cycles and outputs
        \"\"\"
        current_input = initial_input
        current_cycle = 0
        task_outputs = []
        
        self.logger.info(f"Starting evaluation cycle with model: {self.model}")
        
        while current_cycle < max_cycles:
            current_cycle += 1
            self.logger.info(f"Running cycle {current_cycle}/{max_cycles}")
            
            cycle_results = {}
            cycle_success = True
            
            # Store the initial input for this cycle
            cycle_results["input"] = current_input
            
            # Run each task in sequence
            for i, task in enumerate(self.tasks):
                task_name = task.__class__.__name__
                self.logger.info(f"Running task: {task_name}")
                
                # Create the prompt for this task
                prompt = task.create_prompt(current_input)
                
                # Get the output from the model
                try:
                    raw_output = await self.model.generate(prompt)
                    
                    # Extract the relevant part (code or summary)
                    if hasattr(task, "extract_code"):
                        output = task.extract_code(raw_output)
                    elif hasattr(task, "extract_summary"):
                        output = task.extract_summary(raw_output)
                    else:
                        output = raw_output
                        
                    cycle_results[f"task_{i+1}_output"] = output
                    
                    # Test the output if it's code
                    if i == 0 or task.__class__.__name__ == "CodeTranslationTask":
                        test_results = self.tester.test_code(output, 
                                                           language=getattr(task, "language", None) or 
                                                                     getattr(task, "target_language", "python"))
                        
                        cycle_results[f"task_{i+1}_test_results"] = test_results
                        
                        if not test_results["success"]:
                            self.logger.info(f"Test failed at cycle {current_cycle}, task {i+1}")
                            cycle_success = False
                            break
                    
                    # Update the input for the next task or cycle
                    current_input = output
                    
                except Exception as e:
                    self.logger.error(f"Error in task {task_name}: {str(e)}")
                    cycle_results[f"task_{i+1}_error"] = str(e)
                    cycle_success = False
                    break
            
            # Store this cycle's results
            cycle_results["success"] = cycle_success
            self.cycle_results.append(cycle_results)
            
            # Break the loop if this cycle failed
            if not cycle_success:
                break
                
        return {
            "model": str(self.model),
            "successful_cycles": current_cycle if cycle_success else current_cycle - 1,
            "max_cycles_reached": current_cycle >= max_cycles and cycle_success,
            "cycle_results": self.cycle_results
        }
""",
    
    # Experiment runner
    "scripts/run_experiment.py": """import os
import sys
import yaml
import argparse
import asyncio
import logging
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.base import BaseLLM
from src.models.open_source import HuggingFaceLLM
from src.models.closed_source import OpenAILLM
from src.tasks.code_generation import CodeGenerationTask
from src.tasks.code_summarization import CodeSummarizationTask
from src.tasks.code_translation import CodeTranslationTask
from src.evaluation.cycle import EvaluationCycle
from src.evaluation.testing import CodeTester

def load_config(config_path: str) -> Dict[str, Any]:
    \"\"\"Load configuration from YAML file.\"\"\"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_logging(log_dir: str = "logs", level=logging.INFO):
    \"\"\"Set up logging configuration.\"\"\"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "experiment.log")
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_model(model_config: Dict[str, Any]) -> BaseLLM:
    \"\"\"Create a model instance based on configuration.\"\"\"
    model_type = model_config.get("type", "").lower()
    
    if model_type in ["llama", "deepseek", "huggingface", "open_source"]:
        return HuggingFaceLLM(model_config)
    elif model_type in ["openai", "chatgpt", "closed_source"]:
        return OpenAILLM(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def create_tasks(experiment_config: Dict[str, Any]) -> List[Any]:
    \"\"\"Create task instances based on experiment configuration.\"\"\"
    task_configs = experiment_config.get("tasks", [])
    tasks = []
    
    for task_config in task_configs:
        task_type = task_config.get("type", "").lower()
        
        if task_type == "code_generation":
            tasks.append(CodeGenerationTask(task_config))
        elif task_type == "code_summarization":
            tasks.append(CodeSummarizationTask(task_config))
        elif task_type == "code_translation":
            tasks.append(CodeTranslationTask(task_config))
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    return tasks

async def run_experiment(experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Run an experiment based on the configuration.\"\"\"
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment: {experiment_config.get('name', 'unnamed_experiment')}")
    
    # Create models
    models = [create_model(model_config) for model_config in experiment_config.get("models", [])]
    logger.info(f"Created {len(models)} models for evaluation")
    
    # Create tasks
    tasks = create_tasks(experiment_config)
    logger.info(f"Created {len(tasks)} tasks for evaluation")
    
    # Create code tester
    test_cases_dir = experiment_config.get("test_cases_dir", "data/test_cases")
    tester = CodeTester(test_cases_dir)
    
    # Get evaluation parameters
    max_cycles = experiment_config.get("max_cycles", 10)
    prompts = experiment_config.get("prompts", [])
    
    results = []
    
    # Run evaluation for each model and prompt
    for model in models:
        model_results = {
            "model": str(model),
            "prompt_results": []
        }
        
        for prompt in prompts:
            cycle = EvaluationCycle(model, tasks, tester)
            prompt_result = await cycle.run_cycle(prompt, max_cycles)
            model_results["prompt_results"].append(prompt_result)
        
        # Compute average successful cycles
        if model_results["prompt_results"]:
            avg_cycles = sum(r["successful_cycles"] for r in model_results["prompt_results"]) / len(model_results["prompt_results"])
            model_results["average_successful_cycles"] = avg_cycles
        
        results.append(model_results)
    
    return results

def save_results(results: Dict[str, Any], output_dir: str, experiment_name: str):
    \"\"\"Save experiment results to a file.\"\"\"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{experiment_name}_results.yaml")
    
    with open(output_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    
    logging.getLogger(__name__).info(f"Results saved to {output_file}")

async def main():
    parser = argparse.ArgumentParser(description="Run LLM robustness evaluation experiments")
    parser.add_argument("config", help="Path to experiment configuration file")
    parser.add_argument("--output-dir", default="results", help="Directory for saving results")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=getattr(logging, args.log_level.upper()))
    
    # Load experiment configuration
    experiment_config = load_config(args.config)
    experiment_name = experiment_config.get("name", "experiment")
    
    # Run the experiment
    results = await run_experiment(experiment_config)
    
    # Save results
    save_results(results, args.output_dir, experiment_name)

if __name__ == "__main__":
    asyncio.run(main())
""",
    
    # Results analysis
    "scripts/analyze_results.py": """import os
import sys
import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

def load_results(results_file: str) -> Dict[str, Any]:
    \"\"\"Load results from a YAML file.\"\"\"
    with open(results_file, "r") as f:
        return yaml.safe_load(f)

def convert_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    \"\"\"Convert results to a DataFrame for analysis.\"\"\"
    data = []
    
    for model_result in results:
        model_name = model_result["model"]
        avg_cycles = model_result.get("average_successful_cycles", 0)
        
        # Add overall model data
        data.append({
            "model": model_name,
            "prompt": "Average",
            "successful_cycles": avg_cycles
        })
        
        # Add per-prompt details
        for i, prompt_result in enumerate(model_result.get("prompt_results", [])):
            cycles = prompt_result.get("successful_cycles", 0)
            
            data.append({
                "model": model_name,
                "prompt": f"Prompt {i+1}",
                "successful_cycles": cycles
            })
    
    return pd.DataFrame(data)

def plot_results(df: pd.DataFrame, output_dir: str, experiment_name: str):
    \"\"\"Generate plots for the results.\"\"\"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style
    sns.set(style="whitegrid")
    
    # Plot 1: Average successful cycles by model
    plt.figure(figsize=(10, 6))
    avg_df = df[df["prompt"] == "Average"]
    bars = sns.barplot(x="model", y="successful_cycles", data=avg_df)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars.patches):
        bars.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.1,
            f"{bar.get_height():.2f}",
            ha='center',
            va='bottom'
        )
    
    plt.title(f"Average Successful Cycles by Model - {experiment_name}")
    plt.xlabel("Model")
    plt.ylabel("Average Successful Cycles")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_avg_cycles.png"))
    
    # Plot 2: Successful cycles by model and prompt
    plt.figure(figsize=(12, 8))
    prompt_df = df[df["prompt"] != "Average"]
    sns.barplot(x="model", y="successful_cycles", hue="prompt", data=prompt_df)
    plt.title(f"Successful Cycles by Model and Prompt - {experiment_name}")
    plt.xlabel("Model")
    plt.ylabel("Successful Cycles")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Prompt", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_by_prompt.png"))
    
    # Plot 3: Heatmap of successful cycles by model and prompt
    plt.figure(figsize=(10, 8))
    heatmap_df = prompt_df.pivot(index="prompt", columns="model", values="successful_cycles")
    sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt=".1f")
    plt.title(f"Heatmap of Successful Cycles - {experiment_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_heatmap.png"))

def main():
    parser = argparse.ArgumentParser(description="Analyze LLM robustness evaluation results")
    parser.add_argument("results", help="Path to results YAML file")
    parser.add_argument("--output-dir", default="results/plots", help="Directory for saving plots")
    args = parser.parse_args()
    
    # Extract experiment name from results file
    experiment_name = os.path.basename(args.results).split("_results")[0]
    
    # Load results
    results = load_results(args.results)
    
    # Convert to DataFrame
    df = convert_to_dataframe(results)
    
    # Generate plots
    plot_results(df, args.output_dir, experiment_name)
    
    print(f"Analysis complete. Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
""",
    
    # Models configuration
    "config/models.yaml": """# Open-source models
llama-3-70b:
  name: llama-3-70b
  type: huggingface
  path: meta-llama/Llama-3-70b-instruct-hf
  device: cuda
  max_length: 4096

deepseek-coder:
  name: deepseek-coder
  type: huggingface
  path: deepseek-ai/deepseek-coder-33b-instruct
  device: cuda
  max_length: 4096

# Closed-source models
gpt-4o:
  name: gpt-4o
  type: openai
  model_id: gpt-4o
  api_key: ${OPENAI_API_KEY}
  max_tokens: 4096

gpt-3-5-turbo:
  name: gpt-3-5-turbo
  type: openai
  model_id: gpt-3.5-turbo
  api_key: ${OPENAI_API_KEY}
  max_tokens: 4096
""",
    
    # Tasks configuration
    "config/tasks.yaml": """# Code Generation Tasks
python_generation:
  type: code_generation
  language: python
  prompt_template: |
    Generate Python code for the following task: {description}
    
    Return only the code, without explanations or comments.

java_generation:
  type: code_generation
  language: java
  prompt_template: |
    Generate Java code for the following task: {description}
    
    Return only the code, without explanations or comments.

# Code Summarization Tasks
code_summarization:
  type: code_summarization
  prompt_template: |
    Summarize what the following code does in a detailed description:
    
    ```
    {code}
    ```
    
    Provide a comprehensive explanation of the code's functionality, inputs, outputs, 
    and any notable algorithms or techniques used.

# Code Translation Tasks
python_to_java:
  type: code_translation
  source_language: python
  target_language: java
  prompt_template: |
    Translate the following Python code to Java:
    
    ```
    {code}
    ```
    
    Return only the Java code, without explanations or comments.

java_to_python:
  type: code_translation
  source_language: java
  target_language: python
  prompt_template: |
    Translate the following Java code to Python:
    
    ```
    {code}
    ```
    
    Return only the Python code, without explanations or comments.
""",
    
    # Experiment configuration
    "experiments/generation_summarization.yaml": """name: code_generation_summarization
description: "Evaluate LLM robustness with code generation and summarization cycles"

models:
  - name: llama-3-70b
    type: huggingface
    path: meta-llama/Llama-3-70b-instruct-hf
    device: cuda
    max_length: 4096
  
  - name: deepseek-coder
    type: huggingface
    path: deepseek-ai/deepseek-coder-33b-instruct
    device: cuda
    max_length: 4096
  
  - name: gpt-4o
    type: openai
    model_id: gpt-4o
    api_key: ${OPENAI_API_KEY}
    max_tokens: 4096

tasks:
  - type: code_generation
    language: python
    prompt_template: "Generate Python code for the following task: {description}\\n\\nReturn only the code, without explanations or comments."
  
  - type: code_summarization
    prompt_template: "Summarize what the following code does in a detailed description:\\n\\n```\\n{code}\\n```\\n\\nProvide a comprehensive explanation of the code's functionality, inputs, outputs, and any notable algorithms or techniques used."

max_cycles: 10
test_cases_dir: "data/test_cases/python"

prompts:
  - "Write a function that calculates the Fibonacci sequence up to n terms."
  - "Create a function to check if a string is a palindrome."
  - "Implement a function to find the prime factors of a number."
  - "Write a program to solve the Tower of Hanoi puzzle."
  - "Implement a binary search algorithm."
""",
    
    # Sample test file for Python
    "data/test_cases/python/test_basic.py": """
def test_fibonacci():
    # Test the fibonacci function
    fib_10 = fibonacci(10)
    expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    assert fib_10 == expected, f"Expected {expected}, got {fib_10}"

def test_is_palindrome():
    # Test the palindrome function
    assert is_palindrome("racecar") == True
    assert is_palindrome("hello") == False
    assert is_palindrome("A man a plan a canal Panama") == True

# Run tests if this file is executed directly
if __name__ == "__main__":
    try:
        from solution import *
        print("Testing solution...")
        test_fibonacci()
        test_is_palindrome()
        print("All tests passed!")
    except ImportError:
        print("solution.py not found or missing required functions")
    except Exception as e:
        print(f"Tests failed: {e}")
""",
    
    # Sample test file for Java
    "data/test_cases/java/TestSolution.java": """
public class TestSolution {
    public static void main(String[] args) {
        try {
            // Test fibonacci
            int[] fib10 = Solution.fibonacci(10);
            int[] expected = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34};
            boolean fibMatch = true;
            if (fib10.length != expected.length) {
                fibMatch = false;
            } else {
                for (int i = 0; i < expected.length; i++) {
                    if (fib10[i] != expected[i]) {
                        fibMatch = false;
                        break;
                    }
                }
            }
            assert fibMatch : "Fibonacci test failed";
            
            // Test palindrome
            assert Solution.isPalindrome("racecar") == true : "isPalindrome test failed for 'racecar'";
            assert Solution.isPalindrome("hello") == false : "isPalindrome test failed for 'hello'";
            
            System.out.println("All tests passed!");
        } catch (Error e) {
            System.out.println("Tests failed: " + e.getMessage());
        } catch (Exception e) {
            System.out.println("Error running tests: " + e.getMessage());
        }
    }
}
""",
    
    # README.md
    "README.md": """# LLM Robustness Evaluation

This project evaluates the robustness of different LLMs through iterative dual tasks:
- Code generation and code summarization
- Code translation (Python to Java and Java to Python)

The system measures how many cycles an LLM can execute before generating code that fails testing.

## Project Structure

- `config/`: Configuration files for models and tasks
- `src/`: Source code for the evaluation pipeline
- `data/`: Datasets, prompts, and test cases
- `results/`: Evaluation results
- `scripts/`: Utility scripts for running experiments and analyzing results
- `experiments/`: Experiment configurations

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run an experiment
python scripts/run_experiment.py experiments/generation_summarization.yaml

# Analyze results
python scripts/analyze_results.py results/code_generation_summarization_results.yaml
```

## Evaluation Methodology

The evaluation pipeline works through the following steps:

1. For each model and task combination:
   - Start with a natural language prompt
   - Generate code from the prompt
   - Test the code functionality
   - Use the code as input for the next task (summarization or translation)
   - Continue cycling through tasks until the code fails testing or max cycles is reached

2. The robustness score is the number of successful cycles completed before failure.

## Supported Models

- Open-source: LLaMA series, DeepSeek series
- Closed-source: ChatGPT series (GPT-3.5-Turbo, GPT-4o)

## Tasks

- Code Generation: Create code from natural language descriptions
- Code Summarization: Generate natural language descriptions from code
- Code Translation: Convert code between Python and Java

## Results

Results are stored in YAML format with detailed information about each cycle. Visualization tools are provided to analyze and compare model performance.
""",
    
    # requirements.txt
    "requirements.txt": """torch>=2.0.0
transformers>=4.30.0
openai>=1.0.0
pyyaml>=6.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""
}

# __init__.py file content (for Python package directories)
INIT_DIRS = [
    "src",
    "src/models",
    "src/tasks",
    "src/evaluation",
    "src/utils"
]

def create_directory_structure():
    """Create the directory structure for the project."""
    print("Creating directory structure...")
    
    for directory in DIRECTORIES:
        dir_path = os.path.join(PROJECT_ROOT, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {directory}")

def create_files():
    """Create all files with content."""
    print("\nCreating project files...")
    
    for file_path, content in FILE_CONTENTS.items():
        full_path = os.path.join(PROJECT_ROOT, file_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Skip if file already exists
        if os.path.exists(full_path):
            print(f"File already exists (skipping): {file_path}")
            continue
        
        # Create file with content
        with open(full_path, 'w') as f:
            f.write(content)
        
        print(f"Created file: {file_path}")

def create_init_files():
    """Create __init__.py files in Python package directories."""
    print("\nCreating __init__.py files...")
    
    for directory in INIT_DIRS:
        init_path = os.path.join(PROJECT_ROOT, directory, "__init__.py")
        
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                f.write("# This file marks the directory as a Python package\n")
            
            print(f"Created file: {directory}/__init__.py")

def main():
    """Main function to set up the project."""
    print("Setting up LLM Robustness Evaluation project with all content...\n")
    
    create_directory_structure()
    create_files()
    create_init_files()
    
    print("\nProject setup complete! All files created with content.")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Configure your API keys for OpenAI in environment variables")
    print("3. Run an experiment: python scripts/run_experiment.py experiments/generation_summarization.yaml")

if __name__ == "__main__":
    main()