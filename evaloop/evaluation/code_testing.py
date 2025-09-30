# Original Copyright 2021 OpenAI under MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from io import UnsupportedOperation
import itertools
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Union

import numpy as np
import tqdm
from mxeval.data import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl

# Amazon modification
# import check correctness for all languages
from src.evaluation.executor import (
    check_correctness,
    check_correctness_cpp,
    check_correctness_csharp,
    check_correctness_go,
    check_correctness_java,
    check_correctness_javascript,
    check_correctness_kotlin,
    check_correctness_perl,
    check_correctness_php,
    check_correctness_ruby,
    check_correctness_scala,
    check_correctness_swift,
    check_correctness_typescript,
)

check_correctness_function_map = {
        "python": check_correctness,
        "java": check_correctness_java,
        "javascript": check_correctness_javascript,
        "typescript": check_correctness_typescript,
        "kotlin": check_correctness_kotlin,
        "ruby": check_correctness_ruby,
        "php": check_correctness_php,
        "cpp": check_correctness_cpp,
        "csharp": check_correctness_csharp,
        "go": check_correctness_go,
        "perl": check_correctness_perl,
        "scala": check_correctness_scala,
        "swift": check_correctness_swift,
    }

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )

def get_execute_function(lang):
    lang = lang.lower()
    assert lang in check_correctness_function_map, f"Language {lang} is not among the supported languages: {check_correctness_function_map.keys()}"
    return check_correctness_function_map[lang]

def evaluate_functional_correctness(
    samples_list: List[Dict],  # List of dictionaries containing task_id and solution
    k: List[int] = [1, 10, 100],
    n_workers: int = os.cpu_count() - 1,
    timeout: float = 10.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples from a Python list
    instead of loading from a file.
    
    Args:
        samples_list: List of dictionaries, each containing 'task_id' and 'solution'
        k: List of k values for pass@k calculation
        n_workers: Number of parallel workers for execution
        timeout: Maximum execution time per sample in seconds
        problem_file: Path to problem definitions or problem dictionary
        
    Returns:
        Tuple of (pass@k metrics, list of samples with evaluation results)
    """

    if type(problem_file) is not dict:
        problems = read_problems(problem_file)
    else:
        print("Skip reading problems -- using problem_file (dict) as problems")
        problems = problem_file

    # Set random seed for reproducibility
    seed = int(time.time() * 1000000) % 1000000
    np.random.seed(seed=seed)  # microsecond

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Processing samples...")
        for sample in tqdm.tqdm(samples_list):
            task_id = sample["task_id"]
            # Use solution field instead of completion
            solution = sample["solution"]
            language = sample.get("language", "java")  # Default to python if not specified
            
            # Prepare arguments for correctness check
            args = (problems[task_id], solution, timeout, completion_id[task_id])
            check_correctness_function = check_correctness_function_map[language]
            future = executor.submit(check_correctness_function, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(samples_list), "Some problems are not attempted."

        print("Running test suites...")
        # Collect results as they complete
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()  # This executes the test
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k metrics
    total, correct = [], []
    for result in results.values():
        result.sort()  # Sort by completion_id
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    # Calculate pass@k for each requested k value
    ks = k
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()  # Only calculate for k values that are valid for all problems
    }

    # Add results to the original samples and return
    processed_samples = []
    for sample in samples_list:
        task_id = sample["task_id"]
        if results[task_id]:  # Ensure there are results for this task
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            sample["time_elapsed"] = result[1]["time_elapsed"]
        processed_samples.append(sample)

    return pass_at_k, processed_samples


if __name__ == "__main__":
    # Build a mini dataset for testing
    from datasets import load_dataset
    from mxeval.execution import check_correctness
    import json
    from evalplus.data import get_mbpp_plus

    # ruby_problems = load_dataset("AmazonScience/mxeval", "mbxp", split="cpp", trust_remote_code=True)
    # mbpp_plus = list(get_mbpp_plus())
    # mbpp_plus_task_ids = [task_id.split('/')[-1] for task_id in mbpp_plus]

    # testing_samples = []
    # for problem in ruby_problems:
    #     task_id = problem["task_id"].split("/")[-1]
    #     if task_id in mbpp_plus_task_ids:
    #         if problem['prompt'] is not None and problem['canonical_solution'] is not None and problem['test'] is not None:
    #             testing_samples.append({
    #                 "task_id": problem["task_id"],
    #                 "solution": problem["prompt"] + problem['canonical_solution'],
    #                 "language": "cpp",
    #             })
    #     if len(testing_samples) >= 10:
    #         break
    testing_samples = [{
        "task_id": "MBPLP/2",
        "solution": """
sub similar_elements {
   my ($test_tup1, $test_tup2) = @_;
   my %seen;
   $seen{$_}++ for @$test_tup1;
   return [grep { exists $seen{$_} } @$test_tup2];
}
""",
        "language": "perl",
    }]
    print(f"Testing on {len(testing_samples)} samples...")
    pass_at_k, processed_samples = evaluate_functional_correctness(
        testing_samples,
        k=[1],
        n_workers=4,
        timeout=10.0,
        problem_file="./data/meta-data/mbplp_release_v1.jsonl",
    )
    print("Pass@k results:", pass_at_k)
    print("Processed samples:", processed_samples)