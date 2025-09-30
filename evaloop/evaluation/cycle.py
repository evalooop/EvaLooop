import logging
from typing import Dict, Any, List
from tqdm import tqdm

class BatchEvaluationCycle:
    """Manages batch evaluation cycles for LLM testing with support for vLLM batch processing."""
    
    def __init__(self, model, tasks: List[Any]):
        """
        Initialize with model, tasks, and MBPP problems.
        
        Args:
            model: The language model to evaluate
            tasks: List of tasks to run in sequence
            problems: Dict of MBPP problems
        """
        self.model = model
        self.tasks = tasks
        # self.problems = problems
        self.logger = logging.getLogger(__name__)
        
        # Check if model supports batch processing (vLLM models)
        self.supports_batch = hasattr(self.model, 'batch_generate') and callable(getattr(self.model, 'batch_generate'))
        if self.supports_batch:
            self.logger.info("Model supports batch processing - using batch mode")
        else:
            self.logger.info("Model does not support batch processing - using sequential mode")
        
    def run_single_cycle(self, prompts: List[Dict], cycle_number: int) -> Dict[str, Any]:
        """
        Run a single evaluation cycle for multiple prompts without testing.
        
        Args:
            prompts: List of prompts with their task_ids
            cycle_number: Current cycle number
            
        Returns:
            Dict with results for this cycle
        """
        self.logger.info(f"Running cycle {cycle_number} with model: {self.model}")
        
        if self.supports_batch and len(prompts) > 1:
            cycle_results = self._process_batch_prompts(prompts, cycle_number)
        else:
            cycle_results = self._process_sequential_prompts(prompts, cycle_number)
                
        # Return results for this cycle
        return {
            "cycle_number": cycle_number,
            "cycle_results": cycle_results
        }
    
    def _process_batch_prompts(self, prompts: List[Dict], cycle_number: int) -> List[Dict[str, Any]]:
        """
        Process multiple prompts using batch processing for efficiency.
        
        Args:
            prompts: List of prompts with their task_ids
            cycle_number: Current cycle number
            
        Returns:
            List of cycle results for each prompt
        """
        # Initialize results structure
        cycle_results = []
        for prompt_input in prompts:
            task_id = prompt_input["task_id"]
            current_input = prompt_input["prompt"]
            
            # Extract last assertion line if available
            try:
                current_input_lines = current_input.splitlines()
                assert_code = current_input_lines[-2]
            except:
                assert_code = ""
            
            cycle_result = {
                "cycle": cycle_number,
                "task_id": task_id,
                "_assert_code": assert_code,  # Store for later use
                "_current_input": current_input  # Store current input for processing
            }
            cycle_results.append(cycle_result)
        
        # Process each task in sequence using batch processing
        for task_idx, task in enumerate(self.tasks):
            task_name = task.__class__.__name__
            self.logger.info(f"Running {task_name} for {len(prompts)} prompts in batch")
            # import ipdb; ipdb.set_trace()
            try:
                # Collect all prompts for this task
                task_prompts = []
                for cycle_result in cycle_results:
                    if "_current_input" in cycle_result:  # Only process if no error occurred
                        prompt = task.create_prompt(cycle_result["_current_input"])
                        task_prompts.append(prompt)
                    else:
                        task_prompts.append("")  # Placeholder for failed prompts
                
                # Filter out empty prompts and keep track of indices
                valid_prompts = []
                valid_indices = []
                for i, prompt in enumerate(task_prompts):
                    if prompt:
                        valid_prompts.append(prompt)
                        valid_indices.append(i)
                
                # Batch generate for all valid prompts
                if valid_prompts:
                    batch_outputs = self.model.batch_generate(valid_prompts)
                    
                    # Map outputs back to cycle results
                    output_idx = 0
                    for i, cycle_result in enumerate(cycle_results):
                        if i in valid_indices and "_current_input" in cycle_result:
                            raw_output = batch_outputs[output_idx]
                            output_idx += 1
                            
                            # Extract the relevant part (code or summary)
                            if task_name == "CodeGenerationTask":
                                output = task.extract_code(raw_output)
                            elif task_name == "CodeSummarizationTask":
                                description = task.extract_summary(raw_output)
                                assert_code = cycle_result.get("_assert_code", "")
                                output = f"\"\"\"\n{description}\n{assert_code}\n\"\"\"\n"
                            else:
                                output = task.extract_code(raw_output)
                            
                            cycle_result[f"task_{task_idx + 1}_output"] = output
                            cycle_result[f"task_{task_idx + 1}_raw_output"] = raw_output
                            
                            # Update current input for next task
                            cycle_result["_current_input"] = output
                        elif "_current_input" in cycle_result:
                            # Mark as error if prompt was empty but input exists
                            cycle_result[f"task_{task_idx + 1}_error"] = "Empty prompt generated"
                            # Remove current input to prevent further processing
                            del cycle_result["_current_input"]
                
            except Exception as e:
                self.logger.error(f"Error in batch processing task {task_name}: {str(e)}")
                # Mark all remaining prompts as failed for this task
                for cycle_result in cycle_results:
                    if "_current_input" in cycle_result:
                        cycle_result[f"task_{task_idx + 1}_error"] = str(e)
                        del cycle_result["_current_input"]
                break
        
        # Clean up temporary fields
        for cycle_result in cycle_results:
            cycle_result.pop("_assert_code", None)
            cycle_result.pop("_current_input", None)
        
        return cycle_results
    
    def _process_sequential_prompts(self, prompts: List[Dict], cycle_number: int) -> List[Dict[str, Any]]:
        """
        Process prompts sequentially (fallback for non-batch models).
        
        Args:
            prompts: List of prompts with their task_ids
            cycle_number: Current cycle number
            
        Returns:
            List of cycle results for each prompt
        """
        cycle_results = []
        
        # For each prompt, process all tasks and collect results
        for i, prompt_input in tqdm(enumerate(prompts)):
            print(f"processing {i}th of {len(prompts)}")
            
            cycle_result = self._process_single_prompt(prompt_input, cycle_number)
            cycle_results.append(cycle_result)
        
        return cycle_results
    
    def _process_single_prompt(self, prompt_input: Dict, cycle_number: int) -> Dict[str, Any]:
        """
        Process a single prompt through all tasks without testing.
        
        Args:
            prompt_input: Dict containing task_id and prompt
            cycle_number: Current cycle number
            
        Returns:
            Dict with results of processing this prompt through all tasks
        """
        task_id = prompt_input["task_id"]
        current_input = prompt_input["prompt"]
        
        cycle_results = {
            "cycle": cycle_number,
            "task_id": task_id
        }
        
        # Extract last assertion line if available
        try:
            current_input_lines = current_input.splitlines()
            assert_code = current_input_lines[-2]
        except:
            assert_code = ""
        
        # Get problem details, but we already checked this when create the dataset
        # problem = self.problems.get(task_id)
        # if not problem:
        #     self.logger.error(f"Problem {task_id} not found in MBPP dataset")
        #     cycle_results["error"] = f"Problem {task_id} not found"
        #     return cycle_results
        
        # Run each task in sequence
        for i, task in enumerate(self.tasks):
            task_name = task.__class__.__name__
            self.logger.info(f"Running {task_name} for task: {task_id}")
            
            try:
                # Create the prompt for this task
                prompt = task.create_prompt(current_input)
                
                # Get the output from the model
                raw_output = self.model.generate(prompt)
                
                # Extract the relevant part (code or summary)
                if task_name == "CodeGenerationTask":
                    output = task.extract_code(raw_output)
                elif task_name == "CodeSummarizationTask":
                    description = task.extract_summary(raw_output)
                    output = f"\"\"\"\n{description}\n{assert_code}\n\"\"\"\n"
                else:
                    output = task.extract_code(raw_output)
                
                cycle_results[f"task_{i+1}_output"] = output
                cycle_results[f"task_{i+1}_raw_output"] = raw_output
                
                # Update the input for the next task
                current_input = output
                
            except Exception as e:
                self.logger.error(f"Error in task {task_name} for {task_id}: {str(e)}")
                cycle_results[f"task_{i+1}_error"] = str(e)
                break
        
        return cycle_results
