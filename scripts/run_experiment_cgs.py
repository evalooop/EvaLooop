import os
import sys
import yaml
import json
import argparse
import asyncio
import logging
from typing import Dict, Any, List

from datasets import load_dataset

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.base import BaseLLM
from src.models.open_source import HuggingFaceLLM
from src.models.open_source_vllm import VllmLLM
from src.models.closed_source import OpenAILLM
from src.tasks.code_generation import CodeGenerationTask
from src.tasks.code_summarization import CodeSummarizationTask
from src.tasks.code_translation import CodeTranslationTask
from src.evaluation.cycle import BatchEvaluationCycle
from src.evaluation.py_testing import EvalPlusCodeTester, load_mbpp_dataset
from src.evaluation.code_testing import evaluate_functional_correctness
from scripts.analyze_results import analyze_and_save_results


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable support."""
    with open(config_path, "r") as f:
        config_text = f.read()
    
    # Replace environment variable placeholders
    import re
    pattern = r'\${([^}]*)}'
    def replace_env_var(match):
        var_name = match.group(1)
        return os.environ.get(var_name, '')
    
    config_text = re.sub(pattern, replace_env_var, config_text)
    
    # Load processed YAML
    return yaml.safe_load(config_text)


def setup_logging(log_dir: str = "logs", level=logging.INFO):
    """Set up logging configuration."""
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
    """Create a model instance based on configuration."""
    model_type = model_config.get("type", "").lower()
    
    # Log configs without revealing full API keys
    logger = logging.getLogger(__name__)
    sanitized_config = model_config.copy()
    if 'api_key' in sanitized_config:
        api_key = sanitized_config['api_key']
        if api_key:
            sanitized_config['api_key'] = f"{api_key[:5]}...{api_key[-3:]}" if len(api_key) > 8 else "***"
        else:
            logger.warning(f"API key for model {sanitized_config.get('name', 'unknown')} is empty!")
    logger.debug(f"Creating model with config: {sanitized_config}")
    
    if model_type in ["llama", "deepseek", "huggingface", "open_source"]:
        return HuggingFaceLLM(model_config)
    elif model_type in ["vllm"]:
        return VllmLLM(model_config)
    elif model_type in ["openai", "chatgpt", "closed_source"]:
        return OpenAILLM(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def create_tasks(experiment_config: Dict[str, Any]) -> List[Any]:
    """Create task instances based on experiment configuration."""
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


def create_dataset(source_dataset: dict, target_dataset: list) -> List[Dict[str, Any]]:
    """Take the overlap of task IDs from source and target datasets."""
    prepared_dataset = []

    get_target_task_ids = []
    for task in target_dataset:
        task_id = task["task_id"].split("/")[-1]
        get_target_task_ids.append(task_id)
    
    for task_id, task in source_dataset.items():
        task_id = task_id.split("/")[-1]
        if task_id in get_target_task_ids:
            prepared_dataset.append({
                "task_id": task["task_id"],
                "prompt": task["canonical_solution"],
                "language": 'python',
            })
    
    assert len(prepared_dataset) == 375, "Wrong number of tasks in the final dataset"
    
    return prepared_dataset


def test_python_code(output: str, task_id: str, problem: Dict, tester: EvalPlusCodeTester) -> Dict[str, Any]:
    """
    Test code output using the EvalPlus tester.
    
    Args:
        output: The code to test
        task_id: The MBPP task ID
        problem: The problem details
        tester: The EvalPlus tester instance
        
    Returns:
        Dict with test results
    """
    try:
        test_results = tester.test_code(
            output, 
            task_id, 
            problem["entry_point"]
        )
        return test_results
    except Exception as e:
        return {"success": False, "error": str(e)}


def test_other_code(sample_list: list, problem_file: str = "./data/meta-data/mbrbp_release_v1.2.jsonl") -> tuple:
    """
    Test code output for languages other than Python.
    """
    pass_at_k, results = evaluate_functional_correctness(
        sample_list,
        k=[1],
        n_workers=4,
        timeout=10.0,
        problem_file=problem_file,
    )
    return pass_at_k, results


def update_prompts(results_list: List[Dict[str, Any]], py_problems: Dict[str, Any], tester: EvalPlusCodeTester) -> List[Dict[str, Any]]:
    """
    Update prompts for the next cycle.
    """
    # Extract python and java solutions for each task.
    py_results = []
    ruby_results = []

    for result in results_list:
        task_id = result['task_id']
        ruby_results.append({
            "task_id": task_id.replace("Mbpp", "MBRBP"),
            "solution": result['task_1_output'],
            "language": 'ruby'
        })
        py_results.append({
            "task_id": task_id,
            "prompt": result['task_2_output'],
            "language": 'python'
        })
    
    # Get the test results for the python and java solutions
    _, ruby_eval_results = test_other_code(ruby_results)
    print("Ruby evaluation results:", ruby_eval_results)
    py_eval_results = []
    for result in py_results:
        py_eval_result = test_python_code(
            result['prompt'],
            result['task_id'],
            py_problems[result['task_id']],
            tester
        )
        py_eval_results.append(py_eval_result)
    
    # Get java results for tasks that passed the test
    correct_ruby_solutions = []
    for result in ruby_eval_results:
        if result['passed'] != True:
            continue
        correct_ruby_solutions.append(result['task_id'].split("/")[-1])

    updated_prompts = []
    for i in range(len(py_results)):
        task_id = py_results[i]['task_id'].split("/")[-1]
        if task_id not in correct_ruby_solutions:
            continue
        py_eval_result = py_eval_results[i]
        if py_eval_result['plus_success'] != True:
            continue
        updated_prompts.append({
            "task_id": py_results[i]['task_id'],
            "prompt": py_results[i]['prompt'],
            "language": 'python'
        })
    return updated_prompts


def run_experiment(experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run an experiment based on the configuration."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment: {experiment_config.get('name', 'unnamed_experiment')}")
    
    # Create models
    models = [create_model(model_config) for model_config in experiment_config.get("models", [])]
    logger.info(f"Created {len(models)} models for evaluation")
    
    # Create tasks
    tasks = create_tasks(experiment_config)
    logger.info(f"Created {len(tasks)} tasks for evaluation")

    # Get task type
    task_type = experiment_config.get("tasks", [{}])[0].get("type", "").lower()
    
    # Load MBPP dataset
    py_problems, ground_truth = load_mbpp_dataset()
    logger.info(f"Loaded MBPP dataset with {len(py_problems)} problems")
    
    # Create EvalPlus tester
    tester = EvalPlusCodeTester(py_problems, ground_truth)
    
    # Get evaluation parameters
    max_cycles = experiment_config.get("max_cycles", 10)
    
    # Prepare prompts - use MBPP problems
    if task_type == "code_generation" or task_type == "code_summarization":
        prompts = []
        for task_id, problem in py_problems.items():
            prompts.append({
                "task_id": task_id,
                "prompt": problem["prompt"]
            })
    elif task_type == "code_translation":
        target_language = experiment_config.get("tasks", "")[0]['target_language']
        pl_problems = load_dataset("AmazonScience/mxeval", "mbxp", split=target_language, trust_remote_code=True)
        prompts = create_dataset(py_problems, pl_problems)
        
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    logger.info(f"Prepared {len(prompts)} prompts from MBPP dataset")
    
    all_results = []
    
    # Run evaluation for each model
    for model in models:
        logger.info(f"Evaluating model: {model}")
        
        # Create evaluation cycle instance (without tester)
        cycle_runner = BatchEvaluationCycle(model, tasks)
        
        # Initialize prompt results for each prompt
        if task_type == "code_generation" or task_type == "code_summarization":
            # Initialize model results structure
            model_result = {
                "model": str(model),
                "prompt_results": []
            }
            for prompt in prompts:
                prompt_result = {
                    "task_id": prompt["task_id"],
                    "initial_prompt": prompt["prompt"],
                    "cycles": [],
                    "successful_cycles": 0,
                    "max_cycles_reached": False
                }
                model_result["prompt_results"].append(prompt_result)
        elif task_type == "code_translation":
            # Initialize data structure for storing results
            # Tips: we only store the prompt that passes the test, hence we could calculate the number of successful cycles for a task by counting the number of times it appears in the results
            model_result = {
                "model": str(model),
                "cycle_results": {}
            }
            passed_tasks_per_cycle = {
                "model": str(model),
                "cycle_results": {}
            }
            for i in range(max_cycles):
                model_result["cycle_results"][i+1] = []
                passed_tasks_per_cycle["cycle_results"][i+1] = []
        
        # Run cycles until max_cycles or all prompts fail
        current_prompts = prompts.copy()
        current_cycle = 0
        
        while current_cycle < max_cycles and current_prompts:
            current_cycle += 1
            logger.info(f"Starting cycle {current_cycle}/{max_cycles} for model: {model}")
            
            # Run single cycle for all current prompts (without testing)
            cycle_output = cycle_runner.run_single_cycle(current_prompts, current_cycle)
            
            if task_type == "code_generation" or task_type == "code_summarization":
                next_cycle_prompts = []
                
                # Process each result, add testing, and prepare for next cycle
                for cycle_result in cycle_output["cycle_results"]:
                    task_id = cycle_result["task_id"]
                    problem = py_problems.get(task_id)
                    
                    # Find the corresponding prompt result
                    prompt_result = next(r for r in model_result["prompt_results"] if r["task_id"] == task_id)
                    
                    # Get code generation output and test it
                    code_gen_index = next((i for i, task in enumerate(tasks) if isinstance(task, CodeGenerationTask)), None)
                    
                    if code_gen_index is not None:
                        output_key = f"task_{code_gen_index + 1}_output"
                        if output_key in cycle_result:
                            code_output = cycle_result[output_key]
                            test_results = test_python_code(code_output, task_id, problem, tester)
                            test_results["plus_details"] = "".join(map(str, test_results["plus_details"]))
                            cycle_result["test_results"] = test_results
                            cycle_result["success"] = test_results.get("success", False)
                        else:
                            cycle_result["success"] = False
                    else:
                        # No code generation task found
                        cycle_result["success"] = False
                    
                    # Add this cycle's result to prompt result
                    prompt_result["cycles"].append(cycle_result)
                    
                    # Update successful cycles if this one succeeded
                    if cycle_result.get("success", False):
                        prompt_result["successful_cycles"] = current_cycle
                        
                        # Get the last task output for the next cycle
                        last_task_index = len(tasks)
                        last_output_key = f"task_{last_task_index}_output"
                        
                        if last_output_key in cycle_result:
                            next_cycle_prompts.append({
                                "task_id": task_id,
                                "prompt": cycle_result[last_output_key]
                            })
                
                # Update prompts for next cycle (only successful ones continue)
                current_prompts = next_cycle_prompts
            elif task_type == "code_translation":
                model_result["cycle_results"][current_cycle] = cycle_output["cycle_results"] # Store the entire cycle output
                current_prompts = update_prompts(cycle_output["cycle_results"], py_problems, tester)
                passed_tasks_per_cycle["cycle_results"][current_cycle] = current_prompts # Store the tasks that passed the test in this cycle
            
            # If all prompts have failed, break the loop
            if not current_prompts:
                logger.info(f"All prompts failed or completed after cycle {current_cycle}")
                break
        
        # Mark prompts that reached max cycles
        if task_type == "code_generation" or task_type == "code_summarization":
            for prompt_result in model_result["prompt_results"]:
                prompt_result["max_cycles_reached"] = prompt_result["successful_cycles"] == max_cycles
            
            # Calculate the average successful cycles
            if model_result["prompt_results"]:
                avg_cycles = sum(r["successful_cycles"] for r in model_result["prompt_results"]) / len(model_result["prompt_results"])
                model_result["average_successful_cycles"] = avg_cycles
            
            all_results.append(model_result)
            return all_results, all_results
        else:
            # Analyze and save results: analyze passed_tasks_per_cycle for robustness and save model_result
            return model_result, passed_tasks_per_cycle


def save_results(results: Dict[str, Any], output_dir: str, experiment_name: str):
    """Save experiment results to a file."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{experiment_name}_results.json")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logging.getLogger(__name__).info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM robustness evaluation experiments")
    parser.add_argument("--config", help="Path to experiment configuration file")
    parser.add_argument("--output-dir", default="results", help="Directory for saving results")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=getattr(logging, args.log_level.upper()))
    
    # Load experiment configuration
    experiment_config = load_config(args.config)
    task_type = experiment_config.get("tasks", [{}])[0].get("type", "").lower()
    experiment_name = experiment_config.get("name", "experiment")
    
    # Run the experiment
    _, results = run_experiment(experiment_config)
    
    # Save results
    if task_type == "code_generation" or task_type == "code_summarization":
        save_results(results, args.output_dir, experiment_name)
    elif task_type == "code_translation":
        # Analyze and save results for code translation
        analyze_and_save_results(_, results, args.output_dir, experiment_name)
        logging.getLogger(__name__).info(f"Results saved to {args.output_dir}/{experiment_name}_entire_results.json")


if __name__ == "__main__":
    main()
