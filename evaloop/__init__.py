"""
EvaLoop: LLM Robustness Evaluation Framework

A comprehensive framework for evaluating the robustness of Large Language Models
through iterative dual-task cycles.
"""

__version__ = "0.1.0"
__author__ = "EvaLoop Team"
__email__ = "evaloop@example.com"

# Lazy imports to avoid heavy dependencies at package level
def __getattr__(name):
    if name == "EvaLoopEvaluator":
        from .core.evaluator import EvaLoopEvaluator
        return EvaLoopEvaluator
    elif name == "EvaluationConfig":
        from .core.config import EvaluationConfig
        return EvaluationConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["EvaLoopEvaluator", "EvaluationConfig"]
