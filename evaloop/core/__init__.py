"""Core functionality for EvaLoop framework."""

# Lazy imports to avoid heavy dependencies
def __getattr__(name):
    if name == "EvaluationConfig":
        from .config import EvaluationConfig
        return EvaluationConfig
    elif name == "EvaLoopEvaluator":
        from .evaluator import EvaLoopEvaluator
        return EvaLoopEvaluator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["EvaluationConfig", "EvaLoopEvaluator"]
