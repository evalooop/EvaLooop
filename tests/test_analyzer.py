"""End-to-end tests for evaloop.analysis.analyzer.ResultAnalyzer."""

import json
from pathlib import Path

import pytest

from evaloop.analysis.analyzer import ResultAnalyzer
from evaloop.analysis.similarity import SemanticSimilarityJudge
from tests.conftest import FakeLLM


def test_analyze_computes_all_metrics_with_judge(results_file: Path, tmp_path: Path):
    # Boundary tasks in file order: Mbpp/1 (i=1), Mbpp/2 (i=2), Mbpp/3 (i=2).
    llm = FakeLLM(responses=["0.5", "0.8", "0.6"])
    analyzer = ResultAnalyzer(str(results_file), str(tmp_path / "out"), judge=SemanticSimilarityJudge(llm))
    results = analyzer.analyze(generate_plots=True)

    by_metric = {metric: results[metric]["by_model"]["toy-model"] for metric in
                 ("ASL", "ASL_pow", "ASL_bias", "pass@1", "pass_drop")}
    # n = [1, 1, 2, 1], s = [1, 0.5, 0.85, 1], T = 5, M = 3.
    assert by_metric["ASL"] == pytest.approx((1 * 0.5 + 2 * 4 * 0.85 + 9) / 15)
    assert by_metric["ASL_pow"] == pytest.approx(18 / 15)
    assert by_metric["ASL_bias"] == pytest.approx(18 / 12)
    assert by_metric["pass@1"] == pytest.approx(4 / 5)
    assert by_metric["pass_drop"] == pytest.approx(3 / 5)
    assert results["cycle_histogram"]["toy-model"] == [1, 1, 2, 1]
    assert results["similarity"]["toy-model"]["scored"] == 3

    sidecar = results_file.with_name("exp_results_fake-judge_similarity_scores.json")
    assert json.loads(sidecar.read_text()) == {"Mbpp/1_cycle_1": 0.5, "Mbpp/2_cycle_2": 0.8, "Mbpp/3_cycle_2": 0.6}
    for artifact in ("analysis_results.json", "analysis_summary.txt", "asl_comparison.png"):
        assert (tmp_path / "out" / artifact).exists()


def test_analyze_reuses_sidecar_without_api_key(results_file: Path, tmp_path: Path):
    sidecar = tmp_path / "scores.json"
    sidecar.write_text(json.dumps({"Mbpp/1_cycle_1": 1.0, "Mbpp/2_cycle_2": 1.0, "Mbpp/3_cycle_2": 1.0}))
    analyzer = ResultAnalyzer(str(results_file), str(tmp_path / "out"), similarity_path=str(sidecar))
    results = analyzer.analyze(generate_plots=False)
    assert results["ASL"]["by_model"]["toy-model"] == pytest.approx(results["ASL_pow"]["by_model"]["toy-model"])


def test_analyze_without_asl_needs_no_judge(results_file: Path, tmp_path: Path):
    results = ResultAnalyzer(str(results_file), str(tmp_path / "out")).analyze(
        metrics=["ASL_pow", "pass@1"], generate_plots=False
    )
    assert "similarity" not in results
    assert results["ASL_pow"]["by_model"]["toy-model"] == pytest.approx(18 / 15)


def test_analyze_asl_without_credentials_fails_fast(results_file: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="No API key"):
        ResultAnalyzer(str(results_file), str(tmp_path / "out")).analyze(generate_plots=False)


def test_analyze_rejects_unknown_metrics(results_file: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown metric"):
        ResultAnalyzer(str(results_file), str(tmp_path / "out")).analyze(metrics=["ASL_std"])
