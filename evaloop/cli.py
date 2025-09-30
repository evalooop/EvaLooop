#!/usr/bin/env python3
"""
Fire-based CLI interface for EvaLoop evaluation framework.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

import fire

from .core.evaluator import EvaLoopEvaluator
from .core.config import EvaluationConfig
from .utils.logging_utils import setup_logging
from .utils.validation import SystemValidator
from .models.registry import ModelRegistry


class EvaLoopCLI:
    """
    EvaLoop: LLM Robustness Evaluation Framework
    
    EvaLoop evaluates the robustness of Large Language Models through iterative 
    dual-task cycles. The framework measures how many cycles an LLM can execute 
    before generating code that fails functional testing.
    
    Available Commands:
        evaluate        Run code generation and summarization evaluation
        analyze         Analyze evaluation results and generate reports  
        list_models     List all available pre-configured models
        validate_setup  Validate EvaLoop setup and environment
    
    Quick Start Examples:
        # Basic evaluation with OpenAI model
        evaloop evaluate --model "gpt-4"
        
        # Large model with multi-GPU VLLM
        evaloop evaluate --model "Qwen/Qwen2.5-Coder-32B-Instruct" \\
                         --gpu_ids "0,1,2,3" --tensor_parallel_size 4
        
        # Custom experiment settings
        evaloop evaluate --model "deepseek-coder-33b" \\
                         --max_cycles 15 --seed 42 \\
                         --experiment_name "my_experiment"
        
        # Analyze results
        evaloop analyze --results_path "results/experiment_results.json"
    
    For detailed help on any command, use:
        evaloop COMMAND --help
        
    Environment Setup:
        export OPENAI_API_KEY="your-api-key"  # For OpenAI models
        
    More information: https://github.com/your-org/evaloop
    """

    def __init__(self):
        """Initialize the EvaLoop CLI."""
        self.logger = None

    def evaluate(
        self,
        model: str,
        dataset: str = "mbpp_plus",
        max_cycles: int = 10,
        temperature: float = 0.0,
        top_p: float = 1.0,
        output_dir: str = "results/",
        log_level: str = "INFO",
        experiment_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        gpu_ids: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        max_new_tokens: int = 1024,
        trust_remote_code: bool = True,
        batch_size: int = 1,
        seed: Optional[int] = None,
        use_beam_search: bool = False,
        max_batch_size: int = 128,
        auto_batch_size: bool = True,
        preferred_batch_size: int = 64,
        is_chat_model: bool = True,
        system_prompt: str = "You are a helpful programming assistant.",
    ) -> Dict[str, Any]:
        """
        Run code generation and summarization evaluation experiment.
        
        This method performs iterative cycles of code generation and summarization:
        1. Start with a natural language prompt
        2. Generate code from the prompt
        3. Test the generated code for functionality
        4. Summarize the code back to natural language
        5. Repeat until code fails testing or max cycles reached

        Args:
            model: Single model name/path to evaluate.
                   Examples: "gpt-4", "Qwen/Qwen2.5-Coder-32B-Instruct"
            dataset: Dataset to use for evaluation. Supported: "mbpp_plus", "humaneval"
            max_cycles: Maximum number of evaluation cycles to run.
            temperature: Temperature for model generation (0.0 for greedy decoding).
            top_p: Top-p (nucleus) sampling parameter.
            output_dir: Directory to save results.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
            experiment_name: Name for the experiment (auto-generated if None).
            openai_api_key: OpenAI API key (can also be set via OPENAI_API_KEY env var).
            gpu_ids: Comma-separated GPU IDs for VLLM models (e.g., "0,1,2,3").
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: GPU memory utilization ratio for VLLM.
            max_model_len: Maximum model context length.
            max_new_tokens: Maximum number of new tokens to generate.
            trust_remote_code: Whether to trust remote code for model loading.
            batch_size: Batch size for evaluation.
            seed: Random seed for reproducible generation.
            use_beam_search: Whether to use beam search for generation.
            max_batch_size: Maximum batch size for VLLM inference.
            auto_batch_size: Whether to automatically adjust batch size.
            preferred_batch_size: Preferred batch size when auto-batching.
            is_chat_model: Whether the model is a chat/instruction-tuned model.
            system_prompt: System prompt for chat models.

        Returns:
            Dictionary containing evaluation results with successful cycles for each task.
            
        Examples:
            # Basic evaluation with OpenAI model
            evaloop evaluate --model "gpt-4" --dataset "mbpp_plus"
            
            # Large model evaluation with VLLM and multi-GPU setup
            evaloop evaluate --model "Qwen/Qwen2.5-Coder-32B-Instruct" \\
                             --gpu_ids "0,1,2,3" --tensor_parallel_size 4 \\
                             --max_batch_size 128 --preferred_batch_size 64
                             
            # Custom experiment with specific generation settings
            evaloop evaluate --model "deepseek-coder-33b" \\
                             --max_cycles 15 --temperature 0.2 \\
                             --max_new_tokens 2048 --seed 42 \\
                             --system_prompt "You are an expert Python programmer." \\
                             --experiment_name "deepseek_robust_test"
        """
        
        # Setup logging
        self.logger = setup_logging(level=log_level)
        self.logger.info("Starting EvaLoop evaluation experiment")

        # Log model to evaluate
        self.logger.info(f"Model to evaluate: {model.strip()}")

        # Parse GPU IDs if provided
        gpu_id_list = None
        if gpu_ids:
            # Handle both string and tuple inputs from Fire
            if isinstance(gpu_ids, str):
                gpu_id_list = [int(gpu.strip()) for gpu in gpu_ids.split(",")]
            elif isinstance(gpu_ids, (list, tuple)):
                gpu_id_list = [int(gpu) for gpu in gpu_ids]
            else:
                gpu_id_list = [int(gpu_ids)]  # Single GPU ID

        # Setup API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        # Create configuration
        config = EvaluationConfig(
            model=model.strip(),
            dataset=dataset,
            max_cycles=max_cycles,
            temperature=temperature,
            top_p=top_p,
            output_dir=output_dir,
            experiment_name=experiment_name,
            gpu_ids=gpu_id_list,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
            trust_remote_code=trust_remote_code,
            batch_size=batch_size,
            seed=seed,
            use_beam_search=use_beam_search,
            max_batch_size=max_batch_size,
            auto_batch_size=auto_batch_size,
            preferred_batch_size=preferred_batch_size,
            is_chat_model=is_chat_model,
            system_prompt=system_prompt,
            openai_api_key=openai_api_key,
        )
        # Log model type detection results
        model_config = config.get_model_config()
        model_type = model_config["type"]
        self.logger.info(f"Model '{model}' → Using {model_type.upper()} inference engine")
        
        if model_type == "vllm":
            self.logger.info(f"  VLLM config: {model_config['tensor_parallel_size']} GPUs, "
                           f"batch_size={model_config['batch_config']['preferred_batch_size']}")
        elif model_type == "huggingface":
            self.logger.info(f"  HuggingFace config: device={model_config['device']}")
        elif model_type in ["openai", "anthropic"]:
            self.logger.info(f"  API config: model_id={model_config['model_id']}")

        # Create and run evaluator
        evaluator = EvaLoopEvaluator(config)
        results = evaluator.run_code_generation_summarization()

        self.logger.info("Evaluation completed successfully")
        return results


    def analyze(
        self,
        results_path: str,
        metrics: str = "ASL_std,ASL_base",
        generate_plots: bool = True,
        output_dir: Optional[str] = None,
        log_level: str = "INFO",
    ) -> Dict[str, Any]:
        """
        Analyze evaluation results and generate reports.
        
        Computes robustness metrics and generates visualization plots
        from evaluation results. Supports multiple metrics including
        ASL (Average Successful Length) and pass rates.
        
        Args:
            results_path: Path to the evaluation results JSON file.
            metrics: Comma-separated metrics to compute (ASL_std,ASL_base,pass_rate).
            generate_plots: Whether to generate visualization plots.
            output_dir: Directory to save analysis (defaults to results directory).
            log_level: Logging level for analysis output.
        
        Usage:
            evaloop analyze --results_path "results/my_experiment_results.json"
            evaloop analyze --results_path "results/exp.json" --metrics "ASL_std,pass_rate"
        """
        # Setup logging
        self.logger = setup_logging(level=log_level)
        self.logger.info(f"Analyzing results from: {results_path}")

        # Parse metrics list
        metrics_list = [m.strip() for m in metrics.split(",")]

        # Determine output directory
        if output_dir is None:
            output_dir = str(Path(results_path).parent / "analysis")

        # Import analyzer
        from .analysis.analyzer import ResultAnalyzer

        # Create analyzer and run analysis
        analyzer = ResultAnalyzer(results_path, output_dir)
        analysis_results = analyzer.analyze(
            metrics=metrics_list,
            generate_plots=generate_plots,
        )

        self.logger.info(f"Analysis completed. Results saved to: {output_dir}")
        return analysis_results

    def list_models(self) -> List[str]:
        """
        List all available pre-configured models.
        
        Shows all models that can be used with the --models parameter.
        Includes both model names and full HuggingFace paths.
        
        Usage:
            evaloop list_models
        """
        registry = ModelRegistry()
        models = registry.list_available_models()

        print("Available Models:")
        print("================")
        for model in models:
            print(f"- {model}")

        return models

    def validate_setup(
        self,
        check_gpu: bool = True,
        check_api_keys: bool = True,
        log_level: str = "INFO",
    ) -> Dict[str, Any]:
        """
        Validate the EvaLoop setup and environment.
        
        Checks system requirements, dependencies, GPU availability, and API keys.
        Useful for troubleshooting installation and setup issues.
        
        Args:
            check_gpu: Whether to check GPU availability and CUDA setup.
            check_api_keys: Whether to validate API key configuration.
            log_level: Logging level for validation output.
        
        Usage:
            evaloop validate_setup
            evaloop validate_setup --check_gpu=False  # Skip GPU checks
        """
        # Setup logging
        self.logger = setup_logging(level=log_level)
        self.logger.info("Validating EvaLoop setup")


        validator = SystemValidator()
        results = validator.validate_all(
            check_gpu=check_gpu,
            check_api_keys=check_api_keys,
        )

        # Print results
        print("EvaLoop Setup Validation")
        print("========================")
        for check, status in results.items():
            status_symbol = "✓" if status["passed"] else "✗"
            print(f"{status_symbol} {check}: {status['message']}")

        return results


def main():
    """Main entry point for the CLI."""
    import sys
    
    # Add version command support
    if len(sys.argv) > 1 and sys.argv[1] in ['--version', '-v', 'version']:
        from evaloop import __version__
        print(f"EvaLoop {__version__}")
        return
    
    try:
        fire.Fire(EvaLoopCLI)
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nFor help, run: evaloop --help")
        sys.exit(1)


if __name__ == "__main__":
    main()
