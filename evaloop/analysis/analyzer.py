"""Result analysis utilities for EvaLoop.

Computes the EvaLooop robustness metrics (see :mod:`evaloop.analysis.metrics` and the "Scoring"
section of README.md) for every model in a results file. The headline metric ``ASL`` needs a
semantic similarity score for each task's failure boundary; those are produced by an LLM judge
(:mod:`evaloop.analysis.similarity`) and cached in a sidecar file next to the results.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from evaloop.analysis import metrics as asl_metrics
from evaloop.analysis.similarity import (
    JudgeConfig,
    SemanticSimilarityJudge,
    collect_similarity_scores,
    default_similarity_path,
    extract_boundary_pair,
    group_scores_by_bucket,
)

# Loop budget of the paper; used when neither the caller nor the results file specify it.
DEFAULT_MAX_CYCLES = 10

# Metrics that only need the per-task number of sustained loops.
_CYCLE_METRICS: Dict[str, Callable[[List[int], int], float]] = {
    "ASL_pow": asl_metrics.asl_pow,
    "ASL_bias": asl_metrics.asl_bias,
    "pass@1": lambda cycles, _max_cycles: asl_metrics.pass_at_k(cycles, 1) / len(cycles),
    "pass_drop": asl_metrics.pass_drop,
}


class ResultAnalyzer:
    """Analyzer for EvaLoop evaluation results."""

    def __init__(
        self,
        results_path: str,
        output_dir: str,
        judge_config: Optional[JudgeConfig] = None,
        judge: Optional[SemanticSimilarityJudge] = None,
        similarity_path: Optional[str] = None,
        max_cycles: Optional[int] = None,
        rejudge_failed: bool = True,
    ):
        """
        Initialize the result analyzer.

        Args:
            results_path: Path to the results JSON file.
            output_dir: Directory to save analysis outputs.
            judge_config: Configuration of the similarity judge used by the ``ASL`` metric.
                Defaults to ``JudgeConfig()`` (the paper's judge). The judge is only created
                when uncached boundary pairs need scoring.
            judge: Ready-made judge; takes precedence over ``judge_config``.
            similarity_path: Similarity sidecar to read and extend. Defaults to
                ``<results stem>_<judge>_similarity_scores.json`` next to the results file.
                Only allowed for single-model results files.
            max_cycles: Loop budget ``M`` of the experiment. Defaults to the ``max_cycles``
                recorded in the results file, then to 10.
            rejudge_failed: Whether to query the judge again for pairs whose cached score is
                null (a previous judge call failed).
        """
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        self.results = self._load_results(self.results_path)
        self.logger.info(f"Loaded results from {self.results_path}")

        if max_cycles is not None and max_cycles <= 0:
            raise ValueError(f"max_cycles must be positive, got {max_cycles}")
        if similarity_path is not None and len(self.results) > 1:
            raise ValueError("similarity_path can only be set for results files with a single model")

        self.max_cycles = max_cycles
        self.similarity_path = Path(similarity_path) if similarity_path else None
        self.rejudge_failed = rejudge_failed
        self._judge = judge
        self._judge_config = judge_config or JudgeConfig()

    def analyze(
        self,
        metrics: Optional[Sequence[str]] = None,
        generate_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze the results and generate reports.

        Args:
            metrics: Metrics to compute, a subset of ``METRIC_DESCRIPTIONS``. Defaults to all
                of them.
            generate_plots: Whether to generate visualization plots.

        Returns:
            Dictionary containing analysis results: one entry per metric with its description
            and per-model values, plus ``cycle_histogram``, ``summary`` and, when ``ASL`` is
            computed, ``similarity`` details.

        Raises:
            ValueError: If an unknown metric is requested.
        """
        metrics = list(asl_metrics.DEFAULT_METRICS if metrics is None else metrics)
        unknown = [metric for metric in metrics if metric not in asl_metrics.METRIC_DESCRIPTIONS]
        if unknown:
            raise ValueError(
                f"Unknown metric(s) {unknown}; available: {list(asl_metrics.METRIC_DESCRIPTIONS)}"
            )

        self.logger.info(f"Analyzing results with metrics: {metrics}")

        analysis_results: Dict[str, Any] = {
            metric: {"description": asl_metrics.METRIC_DESCRIPTIONS[metric], "by_model": {}}
            for metric in metrics
        }
        analysis_results["cycle_histogram"] = {}
        if "ASL" in metrics:
            analysis_results["similarity"] = {}

        for model_result in self.results:
            self._analyze_model(model_result, metrics, analysis_results)

        # Convert results to DataFrame for summary statistics and plots
        df = self._results_to_dataframe()
        analysis_results["summary"] = self._generate_summary(df)

        # Generate plots if requested
        if generate_plots:
            self._generate_plots(df, analysis_results)

        # Save analysis results
        self._save_analysis(analysis_results)

        return analysis_results

    def _analyze_model(self, model_result: Dict[str, Any], metrics: List[str], analysis_results: Dict[str, Any]):
        """Compute the requested metrics of one model and store them in ``analysis_results``."""
        model_name = model_result["model"]
        max_cycles = self._resolve_max_cycles(model_result)
        cycles = [prompt_result["successful_cycles"] for prompt_result in model_result["prompt_results"]]

        analysis_results["cycle_histogram"][model_name] = asl_metrics.cycle_histogram(cycles, max_cycles)

        for metric in metrics:
            if metric == "ASL":
                scores_by_bucket, similarity_info = self._score_similarity(model_result, max_cycles)
                value = asl_metrics.asl_semantic(cycles, max_cycles, scores_by_bucket)
                analysis_results["similarity"][model_name] = similarity_info
            else:
                value = _CYCLE_METRICS[metric](cycles, max_cycles)
            analysis_results[metric]["by_model"][model_name] = value
            self.logger.info(f"{model_name} {metric} = {value:.4f}")

    def _resolve_max_cycles(self, model_result: Dict[str, Any]) -> int:
        """Return the loop budget ``M`` of a model run."""
        recorded = model_result.get("max_cycles")
        if self.max_cycles is not None:
            if recorded is not None and recorded != self.max_cycles:
                self.logger.warning(
                    f"Results record max_cycles={recorded}, but max_cycles={self.max_cycles} was requested; "
                    f"using {self.max_cycles}"
                )
            return self.max_cycles
        if recorded is not None:
            return int(recorded)
        self.logger.info(
            f"Results do not record max_cycles; assuming {DEFAULT_MAX_CYCLES} (override with max_cycles)"
        )
        return DEFAULT_MAX_CYCLES

    def _score_similarity(
        self, model_result: Dict[str, Any], max_cycles: int
    ) -> Tuple[Dict[int, List[float]], Dict[str, Any]]:
        """Collect the judge scores of a model's failure boundaries.

        Returns:
            The usable scores grouped by bucket and a JSON-serializable summary of the scoring.
        """
        model_name = model_result["model"]
        boundary_tasks = 0
        pairs = []
        for prompt_result in model_result["prompt_results"]:
            if 1 <= prompt_result["successful_cycles"] < max_cycles:
                boundary_tasks += 1
                pair = extract_boundary_pair(prompt_result, max_cycles)
                if pair is not None:
                    pairs.append(pair)
        if len(pairs) < boundary_tasks:
            self.logger.warning(
                f"{model_name}: {boundary_tasks - len(pairs)} boundary tasks have no recorded failing loop "
                f"and cannot be judged"
            )

        cache_path = self._similarity_path_for(model_name)
        scores = collect_similarity_scores(pairs, cache_path, self._get_judge, rejudge_failed=self.rejudge_failed)
        failed = sorted(key for key, score in scores.items() if score is None)
        if failed:
            self.logger.warning(
                f"{model_name}: {len(failed)} boundary pairs have no usable judge score and are left out of "
                f"their bucket mean (rerun to retry them): {failed[:5]}{' ...' if len(failed) > 5 else ''}"
            )

        scores_by_bucket = group_scores_by_bucket(pairs, scores)
        weights = asl_metrics.bucket_similarity(scores_by_bucket, max_cycles)
        usable = [score for score in scores.values() if score is not None]
        similarity_info = {
            "judge": self._judge_name(),
            "cache_path": str(cache_path),
            "boundary_tasks": boundary_tasks,
            "scored": len(usable),
            "failed": len(failed),
            "unjudgeable": boundary_tasks - len(pairs),
            "mean_score": sum(usable) / len(usable) if usable else None,
            "bucket_weights": {str(bucket): weights[bucket] for bucket in range(1, max_cycles)},
        }
        return scores_by_bucket, similarity_info

    def _similarity_path_for(self, model_name: str) -> Path:
        """Return the similarity sidecar of a model."""
        if self.similarity_path is not None:
            return self.similarity_path
        model_part = model_name if len(self.results) > 1 else None
        return default_similarity_path(self.results_path, self._judge_name(), model_part)

    def _get_judge(self) -> SemanticSimilarityJudge:
        """Return the judge, creating it from the configuration on first use."""
        if self._judge is None:
            self._judge = SemanticSimilarityJudge.from_config(self._judge_config)
        return self._judge

    def _judge_name(self) -> str:
        """Return the name of the configured judge."""
        return self._judge.name if self._judge is not None else self._judge_config.model

    @staticmethod
    def _load_results(results_path: Path) -> List[Dict[str, Any]]:
        """Load a results file as a list of model results.

        Accepts the list written by ``EvaLoopEvaluator`` as well as a single model result.

        Raises:
            ValueError: If the file does not look like an EvaLooop results file.
        """
        with open(results_path, 'r') as f:
            results = json.load(f)

        if isinstance(results, dict) and "prompt_results" in results:
            results = [results]
        if not (
            isinstance(results, list)
            and results
            and all(isinstance(model_result, dict) and "prompt_results" in model_result for model_result in results)
        ):
            raise ValueError(
                f"{results_path} is not an EvaLooop results file (expected a list of model results with "
                f"'prompt_results')"
            )
        for index, model_result in enumerate(results):
            model_result.setdefault("model", f"model_{index}")
        return results

    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for model_result in self.results:
            for prompt_result in model_result["prompt_results"]:
                data.append({
                    "model": model_result["model"],
                    "task_id": prompt_result["task_id"],
                    "successful_cycles": prompt_result["successful_cycles"],
                })
        return pd.DataFrame(data)

    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "total_models": df["model"].nunique() if "model" in df.columns else 0,
            "total_tasks": len(df),
        }

        if "successful_cycles" in df.columns:
            summary.update({
                "mean_cycles": df["successful_cycles"].mean(),
                "max_cycles_achieved": df["successful_cycles"].max(),
                "min_cycles_achieved": df["successful_cycles"].min(),
                "median_cycles": df["successful_cycles"].median(),
                "std_cycles": df["successful_cycles"].std(),
            })

        return summary

    def _generate_plots(self, df: pd.DataFrame, analysis_results: Dict[str, Any]):
        """Generate visualization plots."""
        sns.set_style("whitegrid")

        if "successful_cycles" in df.columns:
            self._plot_asl_comparison(analysis_results)
            self._plot_cycle_distribution(df)
            self._plot_success_heatmap(df)

        self.logger.info(f"Plots saved to {self.output_dir}")

    def _plot_asl_comparison(self, analysis_results: Dict[str, Any]):
        """Plot the headline ASL (semantic if computed, else quadratic) of every model."""
        metric = next((name for name in ("ASL", "ASL_pow") if name in analysis_results), None)
        if metric is None:
            return

        plt.figure(figsize=(12, 6))

        model_asl = pd.Series(analysis_results[metric]["by_model"]).sort_values(ascending=False)

        bars = plt.bar(range(len(model_asl)), model_asl.values)
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.title(f"Average Sustainable Loops ({metric}) by Model")
        plt.xticks(range(len(model_asl)), model_asl.index, rotation=45, ha='right')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / "asl_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cycle_distribution(self, df: pd.DataFrame):
        """Plot distribution of successful cycles."""
        plt.figure(figsize=(12, 8))

        models = df["model"].unique()
        for i, model in enumerate(models):
            model_data = df[df["model"] == model]["successful_cycles"]
            plt.subplot(len(models), 1, i+1)
            plt.hist(model_data, bins=range(int(model_data.max()) + 2), alpha=0.7, edgecolor='black')
            plt.title(f"Cycle Distribution - {model}")
            plt.xlabel("Successful Cycles")
            plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(self.output_dir / "cycle_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_success_heatmap(self, df: pd.DataFrame):
        """Plot success heatmap if we have task-level data."""
        if "task_id" not in df.columns:
            return

        # Create pivot table
        pivot_data = df.pivot_table(
            index="task_id",
            columns="model",
            values="successful_cycles",
            fill_value=0
        )

        # Limit to first 50 tasks for readability
        if len(pivot_data) > 50:
            pivot_data = pivot_data.head(50)

        plt.figure(figsize=(12, max(8, len(pivot_data) * 0.3)))
        sns.heatmap(pivot_data, annot=False, cmap="YlOrRd", cbar_kws={'label': 'Successful Cycles'})
        plt.title("Task Success Heatmap")
        plt.xlabel("Model")
        plt.ylabel("Task ID")
        plt.tight_layout()
        plt.savefig(self.output_dir / "success_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _save_analysis(self, analysis_results: Dict[str, Any]):
        """Save analysis results to file."""
        output_file = self.output_dir / "analysis_results.json"

        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)

        self.logger.info(f"Analysis results saved to {output_file}")

        # Also save a readable summary
        summary_file = self.output_dir / "analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("EvaLoop Analysis Summary\n")
            f.write("========================\n\n")

            for metric, results in analysis_results.items():
                if isinstance(results, dict) and "description" in results:
                    f.write(f"{metric}: {results['description']}\n")
                    for model, value in results["by_model"].items():
                        f.write(f"    {model}: {value:.4f}\n")
                    f.write("\n")

            for model, info in analysis_results.get("similarity", {}).items():
                f.write(
                    f"Similarity judge for {model}: {info['judge']}, {info['scored']}/{info['boundary_tasks']} "
                    f"boundary tasks scored, {info['failed']} failed (cache: {info['cache_path']})\n"
                )

        self.logger.info(f"Analysis summary saved to {summary_file}")
