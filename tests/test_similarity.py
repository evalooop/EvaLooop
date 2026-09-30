"""Unit tests for evaloop.analysis.similarity."""

import hashlib
import json
from pathlib import Path

import pytest

from evaloop.analysis import similarity
from evaloop.analysis.similarity import BoundaryPair, JudgeConfig, SemanticSimilarityJudge
from tests.conftest import FakeLLM, make_prompt_result


def test_default_prompt_is_unchanged():
    # The published leaderboard was scored with exactly this prompt; changing it breaks comparability.
    digest = hashlib.sha256(similarity.DEFAULT_JUDGE_PROMPT.encode()).hexdigest()
    assert digest == "95a405af1b020aba7ed26decf1baa63bf46575d5a8dc15d89a40fa5299aed9d0"


@pytest.mark.parametrize(
    "response, expected",
    [
        ("0.85", 0.85),
        (" 1\n", 1.0),
        ("Score: 0.7", 0.7),
        ("1.5", 1.0),
        ("-0.2", 0.0),
        ("Error generating response: Error code: 401", None),
        ("", None),
        (None, None),
        ("I cannot tell", None),
        ("nan", None),
    ],
)
def test_parse_similarity_score(response, expected):
    assert similarity.parse_similarity_score(response) == expected


def test_extract_boundary_pair_first_loop_uses_initial_prompt():
    pair = similarity.extract_boundary_pair(make_prompt_result("T", 1, 3), 3)
    assert pair == BoundaryPair("T", 1, "prompt-T", "code-T-1", "summary-T-1", "code-T-2")
    assert pair.key == "T_cycle_1"


def test_extract_boundary_pair_later_loop_uses_previous_summary():
    pair = similarity.extract_boundary_pair(make_prompt_result("T", 2, 3), 3)
    assert pair == BoundaryPair("T", 2, "summary-T-1", "code-T-2", "summary-T-2", "code-T-3")


@pytest.mark.parametrize("sustained", [0, 3])
def test_extract_boundary_pair_skips_tasks_without_boundary(sustained):
    assert similarity.extract_boundary_pair(make_prompt_result("T", sustained, 3), 3) is None


def test_extract_boundary_pair_skips_unrecorded_failing_loop():
    prompt_result = make_prompt_result("T", 2, 3)
    prompt_result["cycles"] = prompt_result["cycles"][:2]
    assert similarity.extract_boundary_pair(prompt_result, 3) is None


def test_judge_config_validates_prompt_template():
    with pytest.raises(ValueError, match="placeholders"):
        JudgeConfig(prompt_template="Compare {prompt1} and {prompt2}")
    JudgeConfig(prompt_template="{prompt1} {code1} {prompt2} {code2} {{literal}}")


def test_judge_config_requires_api_key_for_openai_endpoint():
    with pytest.raises(ValueError, match="No API key"):
        JudgeConfig().to_model_config()
    local = JudgeConfig(model="local", base_url="http://localhost:8000/v1").to_model_config()
    assert local["api_key"] == "EMPTY"
    assert local["system_prompt"] is None


def test_collect_similarity_scores_caches_and_retries_failures(tmp_path: Path):
    pairs = [similarity.extract_boundary_pair(make_prompt_result(f"T{i}", 1, 3), 3) for i in range(3)]
    cache = tmp_path / "scores.json"
    llm = FakeLLM(responses=["0.9", "Error generating response: timeout", "0.4"])
    judge = SemanticSimilarityJudge(llm)

    scores = similarity.collect_similarity_scores(pairs, cache, lambda: judge)
    assert scores == {"T0_cycle_1": 0.9, "T1_cycle_1": None, "T2_cycle_1": 0.4}
    assert json.loads(cache.read_text()) == scores

    # Only the failed pair is sent to the judge again.
    scores = similarity.collect_similarity_scores(pairs, cache, lambda: judge)
    assert len(llm.prompts) == 4
    assert scores["T1_cycle_1"] == 0.5

    # A complete cache needs no judge at all.
    assert similarity.collect_similarity_scores(pairs, cache, None) == scores


def test_default_similarity_path_matches_archive_naming():
    path = similarity.default_similarity_path(Path("/r/run_results.json"), "gpt-4-turbo-2024-04-09")
    assert path == Path("/r/run_results_gpt-4-turbo-2024-04-09_similarity_scores.json")
    multi = similarity.default_similarity_path(Path("/r/run_results.json"), "org/judge", "OpenAILLM(model_name=x)")
    assert multi.name == "run_results_OpenAILLM_model_name_x_org_judge_similarity_scores.json"
