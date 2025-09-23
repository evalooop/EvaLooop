import os
import sys
import json
import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

def load_results(results_file: str) -> Dict[str, Any]:
    """Load results from a YAML file."""
    with open(results_file, "r") as f:
        return yaml.safe_load(f)

def convert_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert results to a DataFrame for analysis."""
    data = []
    
    for model_result in results:
        model_name = model_result["model"]
        avg_cycles = model_result.get("average_successful_cycles", 0)
        
        # Add overall model data
        data.append({
            "model": model_name,
            "prompt": "Average",
            "successful_cycles": avg_cycles
        })
        
        # Add per-prompt details
        for i, prompt_result in enumerate(model_result.get("prompt_results", [])):
            cycles = prompt_result.get("successful_cycles", 0)
            
            data.append({
                "model": model_name,
                "prompt": f"Prompt {i+1}",
                "successful_cycles": cycles
            })
    
    return pd.DataFrame(data)

def plot_results(df: pd.DataFrame, output_dir: str, experiment_name: str):
    """Generate plots for the results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style
    sns.set(style="whitegrid")
    
    # Plot 1: Average successful cycles by model
    plt.figure(figsize=(10, 6))
    avg_df = df[df["prompt"] == "Average"]
    bars = sns.barplot(x="model", y="successful_cycles", data=avg_df)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars.patches):
        bars.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.1,
            f"{bar.get_height():.2f}",
            ha='center',
            va='bottom'
        )
    
    plt.title(f"Average Successful Cycles by Model - {experiment_name}")
    plt.xlabel("Model")
    plt.ylabel("Average Successful Cycles")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_avg_cycles.png"))
    
    # Plot 2: Successful cycles by model and prompt
    plt.figure(figsize=(12, 8))
    prompt_df = df[df["prompt"] != "Average"]
    sns.barplot(x="model", y="successful_cycles", hue="prompt", data=prompt_df)
    plt.title(f"Successful Cycles by Model and Prompt - {experiment_name}")
    plt.xlabel("Model")
    plt.ylabel("Successful Cycles")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Prompt", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_by_prompt.png"))
    
    # Plot 3: Heatmap of successful cycles by model and prompt
    plt.figure(figsize=(10, 8))
    heatmap_df = prompt_df.pivot(index="prompt", columns="model", values="successful_cycles")
    sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt=".1f")
    plt.title(f"Heatmap of Successful Cycles - {experiment_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_heatmap.png"))


def calculate_average_max_rounds(rounds_dict, total_codes=None):
    """
    Calculate the average maximum round that codes passed
    
    Parameters:
    rounds_dict: Dictionary where keys are round numbers and values are the number of codes passing that round
    total_codes: Total number of codes. If None, uses the count of codes passing the first round.
    
    Returns:
    Average maximum round passed by all codes
    """
    # Convert string keys to integers if necessary
    rounds_dict = {int(k): v for k, v in rounds_dict.items()}
    
    # Sort rounds in ascending order
    sorted_rounds = sorted(rounds_dict.items())
    
    # If total_codes is not provided, use the count of codes passing the first round
    if total_codes is None and sorted_rounds:
        total_codes = sorted_rounds[0][1]
    elif total_codes is None:
        return 0  # No rounds in dictionary
    
    # If no codes passed any round, return 0
    if not rounds_dict or total_codes == 0:
        return 0
        
    total_weighted_rounds = 0  # Sum of weighted rounds
    remaining_codes = total_codes
    
    # For each round, calculate how many codes passed only up to this round
    for i in range(len(sorted_rounds)):
        current_round, code_count = sorted_rounds[i]
        
        # For the last round, all remaining codes passed only up to this round
        if i == len(sorted_rounds) - 1:
            only_current_round = code_count
        else:
            # Calculate codes that passed current round but not the next round
            next_round_count = sorted_rounds[i + 1][1]
            only_current_round = code_count - next_round_count
        
        # Add weighted contribution of current round
        total_weighted_rounds += current_round * only_current_round
    
    return total_weighted_rounds / total_codes


def analyze_and_save_results(entire_results: dict, pass_results_per_cycle: dict, output_dir: str, experiment_name: str):
    """Analyze and save results."""
    cycle_passed_results = pass_results_per_cycle['cycle_results']
    # Get the number of passed tasks for each cycle and save to entire_results
    entire_results["cycle_passed_tasks"] = {}
    for cycle, result in cycle_passed_results.items():
        entire_results["cycle_passed_tasks"][cycle] = len(result)
    
    # Calculate the average cycle metric for robustness evaluation
    per_cycle_passed_task_num = calculate_average_max_rounds(entire_results["cycle_passed_tasks"], 375)
    entire_results["average_cycle_metric"] = per_cycle_passed_task_num
    
    # Save the entire results to a json file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{experiment_name}_entire_results.json"), "w") as f:
        json.dump(entire_results, f, indent=4)
        


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM robustness evaluation results")
    parser.add_argument("results", help="Path to results YAML file")
    parser.add_argument("--output-dir", default="results/plots", help="Directory for saving plots")
    args = parser.parse_args()
    
    # Extract experiment name from results file
    experiment_name = os.path.basename(args.results).split("_results")[0]
    
    # Load results
    results = load_results(args.results)
    
    # Convert to DataFrame
    df = convert_to_dataframe(results)
    
    # Generate plots
    plot_results(df, args.output_dir, experiment_name)
    
    print(f"Analysis complete. Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
