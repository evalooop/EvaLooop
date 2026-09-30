"""Robustness metrics for EvaLooop results.

Every function in this module is pure: it takes the per-task number of sustained loops
(``successful_cycles`` in the results file) and, for the semantic ASL, the judge's similarity
scores, and returns a number. The formulas follow the paper and reproduce the public
leaderboard (https://evalooop.github.io/) exactly; see the "Scoring" section of README.md.

Notation used throughout:
    T    number of tasks in the benchmark run.
    M    maximum number of loops the experiment was run with (``max_cycles``).
    n_i  number of tasks that sustained exactly ``i`` loops, i.e. whose code passed the tests
         in loops 1..i and failed in loop i+1 (``i == M`` means the task never failed).
    s_i  semantic similarity weight of bucket ``i`` (see :func:`bucket_similarity`).
"""

from typing import List, Mapping, Optional, Sequence

# Metric names accepted by ResultAnalyzer / `evaloop analyze --metrics`, with their definitions.
METRIC_DESCRIPTIONS = {
    "ASL": "Semantic ASL (headline score): sum_i n_i * i^2 * s_i / (T * M)",
    "ASL_pow": "Quadratic ASL without semantic weighting: sum_i n_i * i^2 / (T * M)",
    "ASL_bias": "Quadratic ASL normalized by pass@1: sum_i n_i * i^2 / (pass@1 * M)",
    "pass@1": "Fraction of tasks whose code passes the tests in the first loop: pass@1 / T",
    "pass_drop": "Absolute pass-rate drop from the first to the last loop: (pass@1 - pass@M) / T",
}
DEFAULT_METRICS = tuple(METRIC_DESCRIPTIONS)


def cycle_histogram(successful_cycles: Sequence[int], max_cycles: int) -> List[int]:
    """Counts the tasks per number of sustained loops.

    Args:
        successful_cycles: Number of sustained loops of every task.
        max_cycles: Maximum number of loops ``M``.

    Returns:
        A list ``n`` of length ``M + 1`` where ``n[i]`` is the number of tasks that sustained
        exactly ``i`` loops.

    Raises:
        ValueError: If ``max_cycles`` is not positive or a value lies outside ``[0, M]``.
    """
    if max_cycles <= 0:
        raise ValueError(f"max_cycles must be positive, got {max_cycles}")
    counts = [0] * (max_cycles + 1)
    for cycles in successful_cycles:
        if not 0 <= cycles <= max_cycles:
            raise ValueError(f"successful_cycles={cycles} lies outside [0, {max_cycles}]")
        counts[cycles] += 1
    return counts


def pass_at_k(successful_cycles: Sequence[int], k: int) -> int:
    """Counts the tasks whose code still passes the tests in loop ``k``.

    ``pass_at_k(cycles, 1)`` is the classic one-shot pass@1 count.

    Args:
        successful_cycles: Number of sustained loops of every task.
        k: Loop index (1-based).

    Returns:
        Number of tasks that sustained at least ``k`` loops.
    """
    return sum(1 for cycles in successful_cycles if cycles >= k)


def pass_drop(successful_cycles: Sequence[int], max_cycles: int) -> float:
    """Computes the absolute drop in pass rate between loop 1 and loop ``M``.

    Args:
        successful_cycles: Number of sustained loops of every task.
        max_cycles: Maximum number of loops ``M``.

    Returns:
        ``(pass@1 - pass@M) / T``, a fraction in ``[0, 1]``.
    """
    num_tasks = _num_tasks(successful_cycles)
    return (pass_at_k(successful_cycles, 1) - pass_at_k(successful_cycles, max_cycles)) / num_tasks


def bucket_similarity(
    scores_by_bucket: Optional[Mapping[int, Sequence[float]]],
    max_cycles: int,
) -> List[float]:
    """Turns boundary similarity scores into the per-bucket weights ``s_i``.

    A task that sustained ``i`` loops went through ``i`` prompt transitions. The first ``i - 1``
    transitions produced code that passed the tests, so their similarity is taken as 1.0; only
    the last transition (the failure boundary) is scored by the judge. With ``m_i`` the mean
    judge score of the tasks in bucket ``i``, the bucket weight is the mean over the ``i``
    transitions::

        s_i = (m_i + (i - 1)) / i

    Buckets without any judge score (``i = 0``, ``i = M`` and empty buckets) get ``s_i = 1.0``.

    Args:
        scores_by_bucket: Judge scores keyed by the number of sustained loops ``i`` of the scored
            task. Only boundary buckets ``1 <= i < M`` may carry scores. ``None`` means no
            semantic weighting (every ``s_i`` is 1.0).
        max_cycles: Maximum number of loops ``M``.

    Returns:
        A list ``s`` of length ``M + 1`` with ``s[i]`` the weight of bucket ``i``.

    Raises:
        ValueError: If scores are given for a bucket outside ``[1, M - 1]`` or outside ``[0, 1]``.
    """
    weights = [1.0] * (max_cycles + 1)
    for bucket, scores in (scores_by_bucket or {}).items():
        if not 1 <= bucket < max_cycles:
            raise ValueError(f"Similarity scores are only defined for buckets 1..{max_cycles - 1}, got {bucket}")
        if not scores:
            continue
        if any(not 0.0 <= score <= 1.0 for score in scores):
            raise ValueError(f"Similarity scores must lie in [0, 1], got {list(scores)} for bucket {bucket}")
        mean_score = sum(scores) / len(scores)
        weights[bucket] = (mean_score + bucket - 1) / bucket
    return weights


def asl_semantic(
    successful_cycles: Sequence[int],
    max_cycles: int,
    scores_by_bucket: Optional[Mapping[int, Sequence[float]]],
) -> float:
    """Computes the semantic ASL, the headline EvaLooop score.

    ``ASL = sum_i n_i * i^2 * s_i / (T * M)``

    Args:
        successful_cycles: Number of sustained loops of every task.
        max_cycles: Maximum number of loops ``M``.
        scores_by_bucket: Judge scores keyed by bucket, see :func:`bucket_similarity`.

    Returns:
        The semantic ASL score in ``[0, M]``.
    """
    num_tasks = _num_tasks(successful_cycles)
    counts = cycle_histogram(successful_cycles, max_cycles)
    weights = bucket_similarity(scores_by_bucket, max_cycles)
    weighted = sum(counts[i] * i * i * weights[i] for i in range(max_cycles + 1))
    return weighted / (num_tasks * max_cycles)


def asl_pow(successful_cycles: Sequence[int], max_cycles: int) -> float:
    """Computes the quadratic ASL without semantic weighting.

    ``ASL_pow = sum_i n_i * i^2 / (T * M)``, i.e. :func:`asl_semantic` with every ``s_i = 1``.

    Args:
        successful_cycles: Number of sustained loops of every task.
        max_cycles: Maximum number of loops ``M``.

    Returns:
        The quadratic ASL score in ``[0, M]``.
    """
    return asl_semantic(successful_cycles, max_cycles, scores_by_bucket=None)


def asl_bias(successful_cycles: Sequence[int], max_cycles: int) -> float:
    """Computes the quadratic ASL normalized by the tasks that pass the first loop.

    ``ASL_bias = sum_i n_i * i^2 / (pass@1 * M)``. It measures how long a model sustains the
    tasks it can solve at all, independent of its one-shot accuracy.

    Args:
        successful_cycles: Number of sustained loops of every task.
        max_cycles: Maximum number of loops ``M``.

    Returns:
        The normalized quadratic ASL in ``[0, M]``; 0.0 when no task passes the first loop.
    """
    passed_first = pass_at_k(successful_cycles, 1)
    if passed_first == 0:
        return 0.0
    return asl_pow(successful_cycles, max_cycles) * _num_tasks(successful_cycles) / passed_first


def _num_tasks(successful_cycles: Sequence[int]) -> int:
    """Returns ``T`` and rejects empty runs, for which no metric is defined."""
    if not successful_cycles:
        raise ValueError("Cannot compute metrics for a run without tasks")
    return len(successful_cycles)
