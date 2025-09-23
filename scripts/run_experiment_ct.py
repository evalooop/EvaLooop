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
from src.models.closed_source import OpenAILLM
from src.tasks.code_generation import CodeGenerationTask
from src.tasks.code_summarization import CodeSummarizationTask
from src.tasks.code_translation import CodeTranslationTask
from src.evaluation.cycle import BatchEvaluationCycle
from src.evaluation.py_testing import EvalPlusCodeTester, load_mbpp_dataset
from src.evaluation.code_testing import evaluate_functional_correctness


PL_CHAIN = ["javascript", "php", "ruby", "perl", "python"]


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


def run_experiment(experiment_config: Dict[str, Any], output_dir: str, experiment_name: str) -> Dict[str, Any]:
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
    
    if task_type == "code_translation":
        pl_problems = load_dataset("AmazonScience/mxeval", "mbxp", split='ruby', trust_remote_code=True)
        prompts = create_dataset(py_problems, pl_problems)
        
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    logger.info(f"Prepared {len(prompts)} prompts from MBPP dataset")
    
    # Run evaluation for each model
    for model in models:
        logger.info(f"Evaluating model: {model}")
        
        # Create evaluation cycle instance (without tester)
        cycle_runner = BatchEvaluationCycle(model, tasks)
        
        # Initialize data structure for storing results
        # Tips: we only store the prompt that passes the test, hence we could calculate the number of successful cycles for a task by counting the number of times it appears in the results
        model_result = {
            "model": str(model),
            "pl_results": {},
            "pl_passed_ids": {},
        }
        
        # Run single cycle for all current prompts (without testing)
        cycle_output = cycle_runner.run_single_cycle(prompts, 1)
        
        # Analyze and save results for code translation
        analyze_and_save_results(cycle_output['cycle_results'], py_problems=py_problems, model_result=model_result, tester=tester, output_dir=output_dir, experiment_name=experiment_name)
        logging.getLogger(__name__).info(f"Results saved to {output_dir}/{experiment_name}_entire_results.json")
        
        return model_result


def save_results(results: Dict[str, Any], output_dir: str, experiment_name: str):
    """Save experiment results to a file."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{experiment_name}_results.json")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logging.getLogger(__name__).info(f"Results saved to {output_file}")


def analyze_and_save_results(results_list: List[Dict[str, Any]], py_problems: Dict[str, Any], model_result: dict, tester: EvalPlusCodeTester, output_dir: str, experiment_name: str):
    """Analyze and save results for the experiment."""
    # Initialize data structure for storing results
    js_results = []
    php_results = []
    ruby_results = []
    perl_results = []
    py_results = []
    for result in results_list:
        task_id = result['task_id']

        php_results.append({
            "task_id": task_id.replace("Mbpp", "MBPHP"),
            "solution": result['task_1_output'],
            "language": 'php'
        })
        ruby_results.append({
            "task_id": task_id.replace("Mbpp", "MBRBP"),
            "solution": result['task_2_output'],
            "language": 'ruby'
        })
        js_results.append({
            "task_id": task_id.replace("Mbpp", "MBJSP"),
            "solution": result['task_3_output'],
            "language": 'javascript'
        })
        perl_results.append({
            "task_id": task_id.replace("Mbpp", "MBPLP"),
            "solution": result['task_4_output'],
            "language": 'perl'
        })
        py_results.append({
            "task_id": task_id,
            "solution": result['task_5_output'],
            "language": 'python'
        })
    model_result["pl_results"]["javascript"] = js_results
    model_result["pl_results"]["php"] = php_results
    model_result["pl_results"]["ruby"] = ruby_results
    model_result["pl_results"]["perl"] = perl_results
    model_result["pl_results"]["python"] = py_results
    
    # get the results for all languages
    js_pass_at_k, js_eval_results = test_other_code(js_results, problem_file="./data/meta-data/mbjsp_release_v1.2.jsonl")
    php_pass_at_k, php_eval_results = test_other_code(php_results, problem_file="./data/meta-data/mbphp_release_v1.2.jsonl")
    ruby_pass_at_k, ruby_eval_results = test_other_code(ruby_results, problem_file="./data/meta-data/mbrbp_release_v1.2.jsonl")
    perl_pass_at_k, perl_eval_results = test_other_code(perl_results, problem_file="./data/meta-data/mbplp_release_v1.jsonl")
    py_eval_results = []
    for result in py_results:
        py_eval_result = test_python_code(
            result['solution'],
            result['task_id'],
            py_problems[result['task_id']],
            tester
        )
        py_eval_results.append(py_eval_result)
    
    # Extract the task ids that passed the test
    def extract_passed_task_ids(eval_results: List[Dict[str, Any]]) -> List[str]:
        return [result['task_id'].split("/")[-1] for result in eval_results if result['passed'] == True]
    js_correct_solutions = extract_passed_task_ids(js_eval_results)
    php_correct_solutions = extract_passed_task_ids(php_eval_results)
    ruby_correct_solutions = extract_passed_task_ids(ruby_eval_results)
    perl_correct_solutions = extract_passed_task_ids(perl_eval_results)
    py_correct_solutions = []
    for i in range(len(py_results)):
        if py_eval_results[i]['plus_success'] == True:
            py_correct_solutions.append(py_results[i]['task_id'].split("/")[-1])
    model_result["pl_passed_ids"]["javascript"] = (len(js_correct_solutions), js_correct_solutions)
    model_result["pl_passed_ids"]["php"] = (len(php_correct_solutions), php_correct_solutions)
    model_result["pl_passed_ids"]["ruby"] = (len(ruby_correct_solutions), ruby_correct_solutions)
    model_result["pl_passed_ids"]["perl"] = (len(perl_correct_solutions), perl_correct_solutions)
    model_result["pl_passed_ids"]["python"] = (len(py_correct_solutions), py_correct_solutions)

    avg_cycle_metric = ((len(js_correct_solutions) - len(php_correct_solutions)) + (len(php_correct_solutions) - len(ruby_correct_solutions)) * 2 \
                        + (len(ruby_correct_solutions) - len(perl_correct_solutions)) * 3 + (len(perl_correct_solutions) - len(py_correct_solutions)) * 4 \
                        + len(py_correct_solutions) * 5) / 375
    model_result["avg_cycle_metric"] = avg_cycle_metric

    # Save the results to a file
    save_results(model_result, output_dir, experiment_name)

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
    run_experiment(experiment_config, output_dir=args.output_dir, experiment_name=experiment_name)


if __name__ == "__main__":
    main()
