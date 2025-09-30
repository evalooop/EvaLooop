import os
import logging
import pickle
import time
from typing import Dict, Any, Tuple

# Import EvalPlus modules

from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
from evalplus.data.utils import CACHE_DIR
from evalplus.eval import PASS, untrusted_check
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec

def load_mbpp_dataset() -> Tuple[Dict, Dict]:
    """
    Load the MBPP dataset and prepare problems and ground truths.
    
    Returns:
        Tuple containing:
        - problems: Dict of MBPP problems
        - ground_truth: Dict of expected outputs for each problem
    """
    problems = get_mbpp_plus()
    dataset_hash = get_mbpp_plus_hash()
    
    # Load or compute ground truth outputs
    ground_truth = get_groundtruth(problems, dataset_hash, MBPP_OUTPUT_NOT_NONE_TASKS)
    
    return problems, ground_truth

def get_groundtruth(problems, hashcode, tasks_only_output_not_none):
    """
    Get ground truth outputs for the problems.
    
    Args:
        problems: Dict of problems
        hashcode: Hash code for the dataset
        tasks_only_output_not_none: List of tasks that should not return None
        
    Returns:
        Dict of expected outputs for each problem
    """
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")
    logger = logging.getLogger(__name__)
    
    if os.path.exists(cache_file):
        logger.info(f"Loading ground-truth from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    logger.info("Computing expected outputs...")
    tbegin = time.time()
    expected_output = {}
    for task_id, problem in problems.items():
        oracle = {}
        oracle["base"], oracle["base_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["base_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )

        oracle["plus"], oracle["plus_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["plus_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )
        expected_output[task_id] = oracle
    logger.info(f"Expected outputs computed in {time.time() - tbegin:.2f}s")

    with open(cache_file, "wb") as f:
        pickle.dump(expected_output, f)

    return expected_output

class EvalPlusCodeTester:
    """Tests code functionality using EvalPlus."""
    
    def __init__(self, problems, ground_truth):
        """
        Initialize with problems and ground truth.
        
        Args:
            problems: Dict of MBPP problems
            ground_truth: Dict of expected outputs for each problem
        """
        self.problems = problems
        self.ground_truth = ground_truth
        self.logger = logging.getLogger(__name__)
    
    def test_code(self, code: str, task_id: str, entry_point: str) -> Dict[str, Any]:
        """
        Test the given code by executing it using EvalPlus.
        
        Args:
            code: The code to test
            task_id: The MBPP task ID
            entry_point: The name of the function to test
            
        Returns:
            Dict with test results including success/failure and output
        """
        try:
            problem = self.problems.get(task_id)
            if not problem:
                return {"success": False, "error": f"Task ID {task_id} not found"}
            
            # Get expected output from ground truth
            expected_output = self.ground_truth.get(task_id)
            if not expected_output:
                return {"success": False, "error": f"Ground truth for {task_id} not found"}
            
            # Test with base test cases
            base_result = untrusted_check(
                "mbpp",
                code,
                problem["base_input"],
                entry_point,
                expected=expected_output["base"],
                atol=problem.get("atol", 1e-6),
                ref_time=expected_output["base_time"],
                fast_check=False,
                min_time_limit=1.0,
                gt_time_limit_factor=2.0,
            )
            
            # Test with plus test cases (additional EvalPlus test cases)
            plus_result = untrusted_check(
                "mbpp",
                code,
                problem["plus_input"],
                entry_point,
                expected=expected_output["plus"],
                atol=problem.get("atol", 1e-6),
                ref_time=expected_output["plus_time"],
                fast_check=False,
                min_time_limit=1.0,
                gt_time_limit_factor=2.0,
            )
            
            base_success = base_result[0] == PASS
            plus_success = plus_result[0] == PASS
            
            return {
                "success": base_success and plus_success,
                "base_success": base_success,
                "plus_success": plus_success,
                "base_details": base_result[1],
                "plus_details": plus_result[1]
            }
            
        except Exception as e:
            self.logger.error(f"Error testing code: {str(e)}")
            return {"success": False, "error": str(e)}
