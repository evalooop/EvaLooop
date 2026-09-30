"""Analysis module for EvaLoop results."""


# Lazy import: ResultAnalyzer pulls in pandas/matplotlib, which the metric and similarity
# submodules do not need.
def __getattr__(name):
    if name == "ResultAnalyzer":
        from evaloop.analysis.analyzer import ResultAnalyzer
        return ResultAnalyzer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["ResultAnalyzer"]
