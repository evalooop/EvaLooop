"""Unit tests for evaloop.analysis.metrics."""

import pytest

from evaloop.analysis import metrics

# M = 3, n = [1, 1, 1, 2], T = 5, pass@1 = 4, pass@3 = 2, sum_i n_i * i^2 = 1 + 4 + 18 = 23.
CYCLES = [0, 1, 2, 3, 3]
MAX_CYCLES = 3


def test_cycle_histogram_counts_tasks_per_bucket():
    assert metrics.cycle_histogram(CYCLES, MAX_CYCLES) == [1, 1, 1, 2]


def test_cycle_histogram_rejects_values_above_max_cycles():
    with pytest.raises(ValueError, match="outside"):
        metrics.cycle_histogram([4], MAX_CYCLES)


def test_pass_at_k_and_pass_drop():
    assert metrics.pass_at_k(CYCLES, 1) == 4
    assert metrics.pass_at_k(CYCLES, MAX_CYCLES) == 2
    assert metrics.pass_drop(CYCLES, MAX_CYCLES) == pytest.approx(2 / 5)


def test_asl_pow_and_bias():
    assert metrics.asl_pow(CYCLES, MAX_CYCLES) == pytest.approx(23 / 15)
    assert metrics.asl_bias(CYCLES, MAX_CYCLES) == pytest.approx(23 / 12)
    assert metrics.asl_bias([0, 0], MAX_CYCLES) == 0.0


def test_bucket_similarity_averages_boundary_with_successful_transitions():
    weights = metrics.bucket_similarity({1: [0.5], 2: [0.8, 0.6]}, MAX_CYCLES)
    # s_1 = 0.5; s_2 = (0.7 + 1) / 2; buckets 0 and M are always 1.0.
    assert weights == pytest.approx([1.0, 0.5, 0.85, 1.0])


def test_bucket_similarity_rejects_scores_without_failure_boundary():
    with pytest.raises(ValueError, match="buckets"):
        metrics.bucket_similarity({MAX_CYCLES: [0.5]}, MAX_CYCLES)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        metrics.bucket_similarity({1: [1.5]}, MAX_CYCLES)


def test_asl_semantic_matches_formula_and_per_task_closed_form():
    scores = {1: [0.5], 2: [0.8]}
    value = metrics.asl_semantic(CYCLES, MAX_CYCLES, scores)
    # sum_i n_i * i^2 * s_i / (T * M) with s = [1, 0.5, 0.9, 1].
    assert value == pytest.approx((1 * 1 * 0.5 + 1 * 4 * 0.9 + 2 * 9 * 1.0) / 15)
    # Equivalent per task: i * (i - 1 + sim) for boundary tasks, i^2 otherwise.
    assert value == pytest.approx((1 * (0 + 0.5) + 2 * (1 + 0.8) + 9 + 9) / 15)


def test_asl_semantic_without_scores_equals_asl_pow():
    assert metrics.asl_semantic(CYCLES, MAX_CYCLES, None) == metrics.asl_pow(CYCLES, MAX_CYCLES)


def test_metrics_reject_empty_runs():
    with pytest.raises(ValueError, match="without tasks"):
        metrics.asl_pow([], MAX_CYCLES)
