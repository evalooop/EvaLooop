"""Result analysis utilities for EvaLoop."""

import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional


class ResultAnalyzer:
    """Analyzer for EvaLoop evaluation results."""
    
    def __init__(self, results_path: str, output_dir: str):
        """
        Initialize the result analyzer.
        
        Args:
            results_path: Path to the results JSON file.
            output_dir: Directory to save analysis outputs.
        """
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Load results
        with open(self.results_path, 'r') as f:
            self.results = json.load(f)
        
        self.logger.info(f"Loaded results from {self.results_path}")
    
    def analyze(
        self,
        metrics: List[str] = None,
        generate_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze the results and generate reports.
        
        Args:
            metrics: List of metrics to compute.
            generate_plots: Whether to generate visualization plots.
            
        Returns:
            Dictionary containing analysis results.
        """
        if metrics is None:
            metrics = ["ASL_std", "ASL_base", "pass_rate"]
        
        self.logger.info(f"Analyzing results with metrics: {metrics}")
        
        analysis_results = {}
        
        # Convert results to DataFrame for easier analysis
        df = self._results_to_dataframe()
        
        # Compute metrics
        for metric in metrics:
            if metric == "ASL_std":
                analysis_results[metric] = self._compute_asl_standard(df)
            elif metric == "ASL_base":
                analysis_results[metric] = self._compute_asl_base(df)
            elif metric == "pass_rate":
                analysis_results[metric] = self._compute_pass_rate(df)
            else:
                self.logger.warning(f"Unknown metric: {metric}")
        
        # Generate summary statistics
        analysis_results["summary"] = self._generate_summary(df)
        
        # Generate plots if requested
        if generate_plots:
            self._generate_plots(df, analysis_results)
        
        # Save analysis results
        self._save_analysis(analysis_results)
        
        return analysis_results
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        
        # Handle different result formats
        if isinstance(self.results, list):
            # Code generation/summarization format
            for model_result in self.results:
                model_name = model_result["model"]
                
                for prompt_result in model_result.get("prompt_results", []):
                    data.append({
                        "model": model_name,
                        "task_id": prompt_result["task_id"],
                        "successful_cycles": prompt_result["successful_cycles"],
                        "max_cycles_reached": prompt_result["max_cycles_reached"]
                    })
        else:
            # Translation format or other formats
            for model_name, model_data in self.results.items():
                if isinstance(model_data, tuple):
                    # Translation format with (detailed, analysis) tuple
                    _, analysis_data = model_data
                    for cycle, tasks in analysis_data["cycle_results"].items():
                        for task in tasks:
                            data.append({
                                "model": model_name,
                                "cycle": cycle,
                                "task_id": task["task_id"],
                                "successful": True
                            })
        
        return pd.DataFrame(data)
    
    def _compute_asl_standard(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute ASL (Average Successful Length) standard metric."""
        if "successful_cycles" not in df.columns:
            return {"error": "successful_cycles column not found"}
        
        asl_by_model = df.groupby("model")["successful_cycles"].mean().to_dict()
        overall_asl = df["successful_cycles"].mean()
        
        return {
            "by_model": asl_by_model,
            "overall": overall_asl,
            "description": "Average number of successful cycles per task"
        }
    
    def _compute_asl_base(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute ASL base metric (tasks that completed at least one cycle)."""
        if "successful_cycles" not in df.columns:
            return {"error": "successful_cycles column not found"}
        
        # Filter tasks that completed at least one cycle
        successful_df = df[df["successful_cycles"] > 0]
        
        if len(successful_df) == 0:
            return {"error": "No successful tasks found"}
        
        asl_base_by_model = successful_df.groupby("model")["successful_cycles"].mean().to_dict()
        overall_asl_base = successful_df["successful_cycles"].mean()
        
        return {
            "by_model": asl_base_by_model,
            "overall": overall_asl_base,
            "description": "Average successful cycles for tasks that completed at least one cycle"
        }
    
    def _compute_pass_rate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute pass rate (percentage of tasks that completed at least one cycle)."""
        if "successful_cycles" not in df.columns:
            return {"error": "successful_cycles column not found"}
        
        total_tasks = len(df)
        successful_tasks = len(df[df["successful_cycles"] > 0])
        
        overall_pass_rate = (successful_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        # Pass rate by model
        pass_rate_by_model = {}
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            model_total = len(model_df)
            model_successful = len(model_df[model_df["successful_cycles"] > 0])
            pass_rate_by_model[model] = (model_successful / model_total) * 100 if model_total > 0 else 0
        
        return {
            "by_model": pass_rate_by_model,
            "overall": overall_pass_rate,
            "description": "Percentage of tasks that completed at least one cycle"
        }
    
    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "total_models": df["model"].nunique() if "model" in df.columns else 0,
            "total_tasks": len(df),
        }
        
        if "successful_cycles" in df.columns:
            summary.update({
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
            self._plot_asl_comparison(df)
            self._plot_cycle_distribution(df)
            self._plot_success_heatmap(df)
        
        self.logger.info(f"Plots saved to {self.output_dir}")
    
    def _plot_asl_comparison(self, df: pd.DataFrame):
        """Plot ASL comparison between models."""
        plt.figure(figsize=(12, 6))
        
        model_asl = df.groupby("model")["successful_cycles"].mean().sort_values(ascending=False)
        
        bars = plt.bar(range(len(model_asl)), model_asl.values)
        plt.xlabel("Model")
        plt.ylabel("Average Successful Cycles")
        plt.title("Average Successful Length (ASL) by Model")
        plt.xticks(range(len(model_asl)), model_asl.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
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
                    f.write(f"{metric.upper()}: {results['description']}\n")
                    if "overall" in results:
                        f.write(f"  Overall: {results['overall']:.3f}\n")
                    if "by_model" in results:
                        f.write("  By Model:\n")
                        for model, value in results["by_model"].items():
                            f.write(f"    {model}: {value:.3f}\n")
                    f.write("\n")
        
        self.logger.info(f"Analysis summary saved to {summary_file}")
