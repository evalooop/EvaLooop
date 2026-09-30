"""LLM-judged semantic similarity at the failure boundary of EvaLooop tasks.

For a task that sustained ``i`` loops (``1 <= i < M``), the judge compares the last successful
transition with the failing one:

* ``prompt1`` / ``code1``: the prompt that fed loop ``i`` (the original task description when
  ``i == 1``, otherwise the summary written in loop ``i - 1``) and the code generated in loop ``i``.
* ``prompt2`` / ``code2``: the summary written in loop ``i`` (which fed loop ``i + 1``) and the
  failing code generated in loop ``i + 1``.

The judge answers with a score in ``[0, 1]``. Scores are cached in a JSON sidecar next to the
results file (``{"<task_id>_cycle_<i>": score}``, the format of the published archive), so an
analysis only queries the judge for pairs that have not been scored yet.

The judge is any :class:`~evaloop.models.base.BaseLLM`. By default it is built from a
:class:`JudgeConfig` through :class:`~evaloop.models.closed_source.OpenAILLM`, so every
OpenAI-compatible endpoint (OpenAI, a local ``vllm serve`` server, hosted gateways) can serve it.
"""

import json
import logging
import math
import os
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from evaloop.models.base import BaseLLM
from evaloop.models.closed_source import OpenAILLM

logger = logging.getLogger(__name__)

# Judge used for the paper and the public leaderboard.
DEFAULT_JUDGE_MODEL = "gpt-4-turbo-2024-04-09"
DEFAULT_JUDGE_MAX_TOKENS = 10

# Judge prompt used for the paper and the public leaderboard. Keep it byte-identical (including
# the leading newline and the two trailing spaces after {prompt2}) so scores stay comparable.
DEFAULT_JUDGE_PROMPT = (
    "\n"
    "Compare the semantic similarity of these two code generation prompts. "
    "Consider both the prompts themselves and their generated code outputs.\n"
    "\n"
    "Prompt 1: {prompt1}\n"
    "Generated Code 1: {code1}\n"
    "\n"
    "Prompt 2: {prompt2}  \n"
    "Generated Code 2: {code2}\n"
    "\n"
    "Return only a similarity score between 0 and 1, where:\n"
    "- 1.0 = semantically equivalent (same intent, requirements, and expected output)\n"
    "- 0.0 = completely different semantic meaning\n"
    "- Values between 0 and 1 represent partial semantic overlap\n"
    "\n"
    "Score:"
)
JUDGE_PROMPT_FIELDS = frozenset({"prompt1", "code1", "prompt2", "code2"})

# Fallback pattern of the original judge for answers that are not a bare number.
_SCORE_PATTERN = re.compile(r"0?\.\d+|1\.0|0|1")
# OpenAILLM reports API failures as response strings starting with this prefix.
_ERROR_PREFIX = "Error"


def validate_prompt_template(template: str) -> None:
    """Checks that a judge prompt template uses exactly the supported placeholders.

    Args:
        template: A ``str.format`` template.

    Raises:
        ValueError: If a placeholder is missing, unknown or malformed.
    """
    try:
        fields = {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}
    except ValueError as err:
        raise ValueError(f"Malformed judge prompt template: {err}") from err
    if fields != JUDGE_PROMPT_FIELDS:
        raise ValueError(
            f"Judge prompt template must use exactly the placeholders {sorted(JUDGE_PROMPT_FIELDS)}, "
            f"got {sorted(fields)} (write literal braces as '{{{{' and '}}}}')"
        )


def parse_similarity_score(response: Optional[str]) -> Optional[float]:
    """Extracts a similarity score from a judge response.

    Mirrors the original judge: a bare number is used directly, otherwise the first number that
    looks like a score is taken, and the result is clamped to ``[0, 1]``. Unlike the original
    judge, failures return ``None`` instead of 0.0 so they cannot silently lower the ASL.

    Args:
        response: Raw text returned by the judge model.

    Returns:
        The score in ``[0, 1]``, or ``None`` for empty, error or unparseable responses.
    """
    if not response:
        return None
    text = response.strip()
    if not text or text.startswith(_ERROR_PREFIX):
        return None
    try:
        score = float(text)
    except ValueError:
        match = _SCORE_PATTERN.search(text)
        if match is None:
            return None
        score = float(match.group())
    if math.isnan(score):
        return None
    return min(max(score, 0.0), 1.0)


@dataclass(frozen=True)
class BoundaryPair:
    """The two prompt/code transitions around a task's first failure.

    Attributes:
        task_id: Benchmark task identifier.
        sustained_loops: Number of loops ``i`` the task sustained before failing.
        prompt1: Prompt that fed the last successful loop.
        code1: Code generated in the last successful loop.
        prompt2: Summary written in the last successful loop (the prompt of the failing loop).
        code2: Code generated in the failing loop.
    """

    task_id: str
    sustained_loops: int
    prompt1: str
    code1: str
    prompt2: str
    code2: str

    @property
    def key(self) -> str:
        """Sidecar key of this pair, e.g. ``"Mbpp/2_cycle_3"``."""
        return f"{self.task_id}_cycle_{self.sustained_loops}"


def extract_boundary_pair(prompt_result: Mapping[str, Any], max_cycles: int) -> Optional[BoundaryPair]:
    """Builds the judge input for one task of a results file.

    Args:
        prompt_result: One entry of ``prompt_results`` in an EvaLooop results file.
        max_cycles: Maximum number of loops ``M`` of the experiment.

    Returns:
        The boundary pair, or ``None`` if the task has no failure boundary (it failed in the
        first loop or never failed) or the failing loop was not recorded.
    """
    sustained = prompt_result["successful_cycles"]
    if not 1 <= sustained < max_cycles:
        return None
    cycles = prompt_result.get("cycles", [])
    if len(cycles) <= sustained or cycles[sustained].get("success") is not False:
        return None
    last_success = cycles[sustained - 1]
    if sustained == 1:
        prompt1 = prompt_result.get("initial_prompt", "")
    else:
        prompt1 = cycles[sustained - 2].get("task_2_output", "")
    return BoundaryPair(
        task_id=str(prompt_result["task_id"]),
        sustained_loops=sustained,
        prompt1=prompt1,
        code1=last_success.get("task_1_output", ""),
        prompt2=last_success.get("task_2_output", ""),
        code2=cycles[sustained].get("task_1_output", ""),
    )


@dataclass
class JudgeConfig:
    """User-facing configuration of the similarity judge.

    Attributes:
        model: Model id sent to the endpoint.
        api_key: API key; falls back to the ``OPENAI_API_KEY`` environment variable.
        base_url: OpenAI-compatible endpoint; ``None`` uses ``OPENAI_BASE_URL`` or OpenAI itself.
        temperature: Sampling temperature (0.0 = greedy, as in the paper).
        max_tokens: Token budget of the answer. Reasoning models need a much larger budget.
        system_prompt: Optional system message; the paper's judge sends none.
        prompt_template: ``str.format`` template with ``{prompt1}``, ``{code1}``, ``{prompt2}``
            and ``{code2}``.
        request_delay: Seconds to sleep after each judge call (simple rate limiting).
    """

    model: str = DEFAULT_JUDGE_MODEL
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS
    system_prompt: Optional[str] = None
    prompt_template: str = DEFAULT_JUDGE_PROMPT
    request_delay: float = 0.0

    def __post_init__(self) -> None:
        """Validates the configuration."""
        if not self.model:
            raise ValueError("The judge model must not be empty")
        if self.max_tokens <= 0:
            raise ValueError(f"judge max_tokens must be positive, got {self.max_tokens}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"judge temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.request_delay < 0:
            raise ValueError(f"judge request_delay must not be negative, got {self.request_delay}")
        validate_prompt_template(self.prompt_template)

    def to_model_config(self) -> Dict[str, Any]:
        """Builds the ``OpenAILLM`` model config of the judge.

        Returns:
            A model config dictionary.

        Raises:
            ValueError: If no API key is available for the official OpenAI endpoint.
        """
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            if self.base_url is None:
                raise ValueError(
                    "No API key for the similarity judge: pass --judge_api_key or --judge_api_key_file, "
                    "set OPENAI_API_KEY, or reuse an existing similarity sidecar via --similarity_path"
                )
            api_key = "EMPTY"  # Self-hosted OpenAI-compatible servers usually accept any key.
        return {
            "name": self.model,
            "type": "openai",
            "model_id": self.model,
            "api_key": api_key,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": 1.0,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
        }


class SemanticSimilarityJudge:
    """Scores boundary pairs with an LLM."""

    def __init__(
        self,
        llm: BaseLLM,
        name: Optional[str] = None,
        prompt_template: str = DEFAULT_JUDGE_PROMPT,
        request_delay: float = 0.0,
    ):
        """Initializes the judge.

        Args:
            llm: Model that answers the judge prompt; its ``generate`` must return text.
            name: Judge name used in sidecar file names; defaults to ``llm.model_name``.
            prompt_template: Judge prompt template, see :func:`validate_prompt_template`.
            request_delay: Seconds to sleep after each call.
        """
        validate_prompt_template(prompt_template)
        self.llm = llm
        self.name = name or llm.model_name
        self.prompt_template = prompt_template
        self.request_delay = request_delay

    @classmethod
    def from_config(cls, config: JudgeConfig) -> "SemanticSimilarityJudge":
        """Creates a judge served through :class:`OpenAILLM`.

        Args:
            config: Judge configuration.

        Returns:
            The judge.
        """
        return cls(
            OpenAILLM(config.to_model_config()),
            name=config.model,
            prompt_template=config.prompt_template,
            request_delay=config.request_delay,
        )

    def build_prompt(self, pair: BoundaryPair) -> str:
        """Renders the judge prompt for a boundary pair."""
        return self.prompt_template.format(
            prompt1=pair.prompt1, code1=pair.code1, prompt2=pair.prompt2, code2=pair.code2
        )

    def score(self, pair: BoundaryPair) -> Optional[float]:
        """Scores one boundary pair.

        Args:
            pair: The pair to score.

        Returns:
            The similarity in ``[0, 1]``, or ``None`` if the judge gave no usable answer.
        """
        response = self.llm.generate(self.build_prompt(pair))
        score = parse_similarity_score(response)
        if score is None:
            logger.warning("Judge %s returned no usable score for %s: %.200r", self.name, pair.key, response)
        if self.request_delay:
            time.sleep(self.request_delay)
        return score


def default_similarity_path(results_path: Path, judge_name: str, model_name: Optional[str] = None) -> Path:
    """Returns the default sidecar path of a results file.

    ``<dir>/<results stem>[_<model>]_<judge>_similarity_scores.json``; the model part is only
    needed for results files that contain several models.

    Args:
        results_path: Path of the results JSON file.
        judge_name: Name of the judge model.
        model_name: Name of the evaluated model, for multi-model results files.

    Returns:
        The sidecar path.
    """
    parts = [results_path.stem]
    if model_name:
        parts.append(_sanitize_file_part(model_name))
    parts += [_sanitize_file_part(judge_name), "similarity_scores"]
    return results_path.with_name("_".join(parts) + ".json")


def load_similarity_scores(path: Path) -> Dict[str, Optional[float]]:
    """Loads a similarity sidecar.

    Args:
        path: Sidecar path.

    Returns:
        Scores keyed by boundary-pair key; ``None`` marks a failed judgement. Empty if the file
        does not exist.

    Raises:
        ValueError: If the file is not a flat mapping of keys to numbers or null.
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as sidecar:
        data = json.load(sidecar)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a similarity sidecar (expected a JSON object)")
    scores: Dict[str, Optional[float]] = {}
    for key, value in data.items():
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"{path}: score of {key!r} must be a number or null, got {value!r}")
        scores[key] = None if value is None else float(value)
    return scores


def save_similarity_scores(path: Path, scores: Mapping[str, Optional[float]]) -> None:
    """Writes a similarity sidecar atomically.

    Args:
        path: Sidecar path.
        scores: Scores keyed by boundary-pair key.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as sidecar:
        json.dump(dict(scores), sidecar, indent=2)
    os.replace(tmp_path, path)


def collect_similarity_scores(
    pairs: Sequence[BoundaryPair],
    cache_path: Path,
    judge_provider: Optional[Callable[[], SemanticSimilarityJudge]],
    rejudge_failed: bool = True,
) -> Dict[str, Optional[float]]:
    """Returns the similarity score of every pair, querying the judge only for uncached pairs.

    The sidecar is rewritten after every judge call, so an interrupted run resumes where it
    stopped.

    Args:
        pairs: Boundary pairs to score.
        cache_path: Sidecar path used as cache.
        judge_provider: Returns the judge; only called if a pair still needs scoring, so no API
            key is required when the cache is complete. ``None`` forbids judge calls.
        rejudge_failed: Whether to query the judge again for pairs cached as failed (``None``).

    Returns:
        Scores keyed by pair key (``None`` for failed judgements), in the order of ``pairs``.

    Raises:
        ValueError: If pairs need scoring but no judge is available.
    """
    cache = load_similarity_scores(cache_path)
    pending = [
        pair for pair in pairs
        if pair.key not in cache or (rejudge_failed and cache[pair.key] is None)
    ]
    if pending:
        if judge_provider is None:
            raise ValueError(f"{len(pending)} boundary pairs are not in {cache_path} and no judge is configured")
        judge = judge_provider()
        logger.info("Scoring %d boundary pairs with judge %s (cache: %s)", len(pending), judge.name, cache_path)
        for index, pair in enumerate(pending, start=1):
            cache[pair.key] = judge.score(pair)
            save_similarity_scores(cache_path, cache)
            logger.info("[%d/%d] %s: %s", index, len(pending), pair.key, cache[pair.key])
    else:
        logger.info("All %d boundary pairs are cached in %s", len(pairs), cache_path)
    return {pair.key: cache.get(pair.key) for pair in pairs}


def group_scores_by_bucket(
    pairs: Sequence[BoundaryPair],
    scores: Mapping[str, Optional[float]],
) -> Dict[int, List[float]]:
    """Groups the usable scores by the number of sustained loops of their task.

    Args:
        pairs: Scored boundary pairs.
        scores: Scores keyed by pair key; ``None`` entries are skipped.

    Returns:
        Scores keyed by bucket ``i``, the input of :func:`evaloop.analysis.metrics.asl_semantic`.
    """
    buckets: Dict[int, List[float]] = {}
    for pair in pairs:
        score = scores.get(pair.key)
        if score is not None:
            buckets.setdefault(pair.sustained_loops, []).append(score)
    return buckets


def _sanitize_file_part(name: str) -> str:
    """Makes a model name safe to use inside a file name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "unnamed"
