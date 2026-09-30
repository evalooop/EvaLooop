# EvaLooop: LLM Robustness Evaluation Framework

[![PyPI version](https://badge.fury.io/py/evaloop.svg)](https://badge.fury.io/py/evaloop)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EvaLoop is a comprehensive framework for evaluating the robustness of Large Language Models (LLMs) through iterative dual-task cycles. The framework measures how many cycles an LLM can execute before generating code that fails functional testing, providing a quantitative metric for model robustness.

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (recommended)
pip install evaloop

# Or install from source
git clone https://github.com/your-org/evaloop
cd evaloop
pip install -e .
```

### Basic Usage

```bash
# Evaluate OpenAI models on code generation + summarization
evaloop evaluate --models "gpt-4,gpt-3.5-turbo" --dataset "mbpp_plus"

# Evaluate open-source models with custom configuration
evaloop evaluate --models "meta-llama/Llama-3-70b-instruct-hf" \
                  --gpu_ids "0,1,2,3" \
                  --tensor_parallel_size 4 \
                  --max_cycles 15

# Run code translation experiments
evaloop translate --models "gpt-4,deepseek-coder-33b" \
                   --languages "python,java" \
                   --max_cycles 8

# Analyze results (the default ASL metric calls an LLM judge; see "Scoring")
evaloop analyze --results_path "results/experiment_results.json" \
                 --judge_api_key_file api.key
```

## 🎯 Key Features

- **🔥 Fire-based CLI**: Simple, intuitive command-line interface powered by Python Fire
- **🤖 Multi-Model Support**: OpenAI GPT, LLaMA, DeepSeek, Qwen, and more
- **⚡ High Performance**: VLLM integration for efficient large model inference
- **📊 Rich Analytics**: Comprehensive result analysis with visualization
- **🔧 Extensible**: Plugin architecture for custom models and tasks
- **📦 Easy Distribution**: Installable package with proper dependency management

## 📋 Evaluation Methodology

EvaLoop evaluates LLM robustness through iterative task cycles:

1. **Initial Input**: Start with a natural language prompt describing a coding task
2. **Task Execution**: Generate code from the prompt using the target LLM
3. **Functional Testing**: Execute the generated code against predefined test cases
4. **Task Alternation**: Use the output as input for the complementary task
5. **Iteration**: Continue cycling through tasks until code fails testing or max cycles reached
6. **Robustness Scoring**: The number of loops each task sustains is aggregated into the Average Sustainable Loops (ASL) score, which weights long survival quadratically and discounts semantic drift at the failure boundary (see [Scoring](#-scoring))

### Supported Task Types

- **Code Generation**: Create executable code from natural language descriptions
- **Code Summarization**: Generate natural language descriptions from source code
- **Code Translation**: Convert code between programming languages (Python ↔ Java/C++/Ruby)

## 🛠️ CLI Reference

### Core Commands

#### `evaluate` - Code Generation & Summarization

Run iterative code generation and summarization cycles:

```bash
evaloop evaluate [OPTIONS]
```

**Key Options:**
- `--models`: Comma-separated model names/paths (required)
- `--dataset`: Dataset to use (`mbpp_plus`, `humaneval`) 
- `--max_cycles`: Maximum evaluation cycles (default: 10)
- `--temperature`: Generation temperature (default: 0.0)
- `--output_dir`: Results output directory (default: `results/`)
- `--gpu_ids`: GPU IDs for VLLM models (e.g., `"0,1,2,3"`)
- `--tensor_parallel_size`: GPUs for tensor parallelism (default: 1)

**Examples:**

```bash
# Basic OpenAI evaluation
evaloop evaluate --models "gpt-4,gpt-3.5-turbo"

# Large model with multi-GPU setup
evaloop evaluate --models "meta-llama/Llama-3-70b-instruct-hf" \
                  --gpu_ids "0,1,2,3" \
                  --tensor_parallel_size 4 \
                  --gpu_memory_utilization 0.85

# Custom experiment parameters
evaloop evaluate --models "deepseek-coder-33b" \
                  --max_cycles 20 \
                  --temperature 0.2 \
                  --batch_size 4 \
                  --experiment_name "deepseek_robust_eval"
```

#### `translate` - Code Translation

Run code translation evaluation between programming languages:

```bash
evaloop translate [OPTIONS]
```

**Key Options:**
- `--models`: Comma-separated model names/paths (required)
- `--languages`: Source and target languages (default: `"python,java"`)
- `--max_cycles`: Maximum translation cycles (default: 8)

**Examples:**

```bash
# Python to Java translation
evaloop translate --models "gpt-4,claude-3" --languages "python,java"

# Multi-language translation
evaloop translate --models "deepseek-coder-33b" \
                   --languages "python,cpp" \
                   --max_cycles 10
```

#### `analyze` - Result Analysis

Analyze evaluation results and generate reports:

```bash
evaloop analyze --results_path "results/experiment_results.json" [OPTIONS]
```

**Key Options:**
- `--results_path`: Path to results JSON file (required)
- `--metrics`: Metrics to compute (default: `"ASL,ASL_pow,ASL_bias,pass@1,pass_drop"`, see [Scoring](#-scoring))
- `--generate_plots`: Generate visualization plots (default: True)
- `--output_dir`: Analysis output directory (default: `<results dir>/analysis`)
- `--max_cycles`: Loop budget M of the run (default: value recorded in the results file, else 10)

**Similarity judge options** (only used by the `ASL` metric, and only for pairs not cached yet):
- `--judge_model`: Judge model id (default: `gpt-4-turbo-2024-04-09`, the paper's judge)
- `--judge_api_key` / `--judge_api_key_file`: Judge API key (default: `OPENAI_API_KEY`)
- `--judge_base_url`: Any OpenAI-compatible endpoint, e.g. a local `vllm serve` server (default: OpenAI)
- `--judge_max_tokens`: Answer token budget (default: 10; raise it for reasoning models)
- `--judge_temperature`: Judge temperature (default: 0.0)
- `--judge_system_prompt`: Optional system message (default: none)
- `--judge_prompt_file`: Custom prompt template with `{prompt1}`, `{code1}`, `{prompt2}`, `{code2}`
- `--judge_delay`: Seconds between judge calls (default: 0)
- `--similarity_path`: Similarity sidecar to read and extend (default: next to the results file)
- `--rejudge_failed`: Retry pairs whose previous judge call failed (default: True)

**Examples:**

```bash
# Default analysis: all metrics, paper judge (needs an OpenAI key the first time)
evaloop analyze --results_path "results/my_experiment_results.json" --judge_api_key_file api.key

# Use a self-hosted judge through vLLM's OpenAI-compatible server
vllm serve Qwen/Qwen2.5-72B-Instruct --port 8000
evaloop analyze --results_path "results/exp.json" \
                 --judge_model "Qwen/Qwen2.5-72B-Instruct" \
                 --judge_base_url "http://localhost:8000/v1"

# Skip the judge: metrics that only need the loop counts
evaloop analyze --results_path "results/exp.json" \
                 --metrics "ASL_pow,ASL_bias,pass@1,pass_drop" \
                 --output_dir "analysis/custom/"
```

**Outputs** (in `--output_dir`): `analysis_results.json` (all metric values per model, the loop
histogram `n_i`, and judge statistics such as the bucket weights `s_i`), `analysis_summary.txt`,
and the plots. Judge scores go to the similarity sidecar next to the results file.

### Utility Commands

#### `list_models` - Available Models

List all pre-configured models:

```bash
evaloop list_models
```

#### `validate_setup` - System Validation

Validate your EvaLoop installation and environment:

```bash
evaloop validate_setup [OPTIONS]
```

**Options:**
- `--check_gpu`: Check GPU availability (default: True)
- `--check_api_keys`: Check API key configuration (default: True)

## 🔧 Configuration

### Environment Variables

Set up API keys for closed-source models:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional
```

### Model Configuration

EvaLoop automatically configures models based on their names/paths:

- **OpenAI Models**: `gpt-4`, `gpt-3.5-turbo`, `gpt-4-turbo`
- **HuggingFace Paths**: `meta-llama/Llama-3-70b-instruct-hf`
- **Pre-registered Names**: Use `evaloop list_models` to see available options

Large models (70B+) automatically use VLLM for efficient inference.

## 📊 Results and Analysis

### Output Structure

Results are saved in JSON format with detailed information:

```json
{
  "model": "gpt-4",
  "max_cycles": 10,
  "prompt_results": [
    {
      "task_id": "mbpp_1",
      "initial_prompt": "Write a function to find the minimum element",
      "cycles": [...],
      "successful_cycles": 5,
      "max_cycles_reached": false
    }
  ],
  "average_successful_cycles": 4.2
}
```

### Metrics

`evaloop analyze` reports `ASL` (the headline score, used by default), `ASL_pow`, `ASL_bias`,
`pass@1` and `pass_drop`. How each one is computed is described in [Scoring](#-scoring).

### Visualization

Analysis generates several plots:
- Model comparison bar charts
- Cycle distribution histograms  
- Task success heatmaps
- Performance trend analysis

## 📐 Scoring

The scores below are the ones used in the paper and on the [leaderboard](https://evalooop.github.io/);
`evaloop analyze` reproduces the published values exactly when given the same results and judge scores.

### Notation

- **T**: number of tasks in the run (378 for MBPP Plus).
- **M**: maximum number of loops (`max_cycles`, 10 by default).
- **i**: number of loops a task *sustains* (`successful_cycles` in the results file): its code passed
  the tests in loops 1..i and failed in loop i+1. `i = 0` means it already failed in the first loop,
  `i = M` means it never failed.
- **n_i**: number of tasks that sustain exactly i loops.
- **pass@k**: number of tasks that sustain at least k loops (so pass@1 is the usual one-shot pass count).

### ASL: the headline score (default)

```
ASL = Σ_{i=1..M} n_i · i² · s_i / (T · M)          range [0, M]
```

- **i²** rewards sustained correctness quadratically: surviving 8 loops counts 4× as much as surviving 4.
- **s_i** is the semantic similarity weight of bucket i. It discounts tasks whose last successful
  loop had already drifted away from the original intent.

**How s_i is obtained.** For every task with a failure boundary (`1 ≤ i < M`), an LLM judge
compares the last successful transition with the failing one:

| | Prompt | Code |
|---|---|---|
| Side 1 | prompt that fed loop i (the original task if i = 1, else the summary written in loop i−1) | code generated in loop i (passed) |
| Side 2 | summary written in loop i, i.e. the prompt of loop i+1 | code generated in loop i+1 (failed) |

The judge returns a score `sim ∈ [0, 1]`. A task that sustained i loops went through i prompt
transitions; the first i−1 produced passing code, so they count as similarity 1.0, and only the
boundary transition uses the judge score. With `m_i` the mean judge score of the tasks in bucket i:

```
s_i = (m_i + (i − 1)) / i        for 1 ≤ i < M
s_i = 1                          for i = M (never failed) and for buckets without judge scores
```

Because the mean is linear, this equals giving every boundary task the weight `i · (i − 1 + sim)`
instead of `i²`, so `ASL ≤ ASL_pow` and the gap is `Σ_tasks i · (1 − sim) / (T · M)`.

*Example:* with T = 378 and M = 10, a task that sustains 4 loops and gets a judge score of 0.85
contributes `4 · (3 + 0.85) / 3780 = 0.00407` to ASL, versus `16 / 3780 = 0.00423` without the
similarity weighting.

### All metrics

| Metric | Formula | Range | What it measures | Leaderboard field |
|---|---|---|---|---|
| `ASL` | Σ n_i · i² · s_i / (T · M) | [0, M] | Headline robustness score (default) | `semanticSimilarityScore`, shown as "ASL" |
| `ASL_pow` | Σ n_i · i² / (T · M) | [0, M] | ASL without semantic weighting (all s_i = 1) | `aslScore` |
| `ASL_bias` | Σ n_i · i² / (pass@1 · M) | [0, M] | Robustness on the tasks the model solves at all, independent of one-shot accuracy (0 if pass@1 = 0) | `robustnessScore` |
| `pass@1` | pass@1 / T | [0, 1] | One-shot accuracy | `successRate` |
| `pass_drop` | (pass@1 − pass@M) / T | [0, 1] | Absolute accuracy lost within M loops | "drop" (as a percentage) |

Only `ASL` needs the judge; the other metrics use the loop counts alone. The leaderboard also
averages repeated runs of a model and only lists models with `pass@1 > 0.2` and `ASL_bias > 3`;
those steps happen outside this package.

### Similarity judge

- **Default:** `gpt-4-turbo-2024-04-09` at temperature 0 with a 10-token answer budget, no system
  message and the paper's prompt (`evaloop.analysis.similarity.DEFAULT_JUDGE_PROMPT`). Scores
  from a different judge or prompt are not comparable with the leaderboard, so compare models
  only under the same judge.
- **Changing the judge:** the judge is served through the same API layer as the evaluated models
  (`OpenAILLM`), so any OpenAI-compatible endpoint works via `--judge_model`, `--judge_base_url`
  and `--judge_api_key(_file)`. In Python, pass `judge_config=JudgeConfig(...)` to
  `ResultAnalyzer`, or wrap any `BaseLLM` in `SemanticSimilarityJudge(llm)` and pass it as
  `judge=...`.
- **Caching:** scores are stored in a sidecar next to the results file,
  `<results stem>_<judge>_similarity_scores.json`, as `{"<task_id>_cycle_<i>": score}` (the format
  of the published data archive). The judge is only called for pairs missing from the sidecar, so
  reruns are free and an interrupted run resumes where it stopped. No API key is needed when the
  sidecar is complete.
- **Failures:** if the judge call fails or its answer cannot be parsed, the pair is stored as
  `null`, left out of its bucket mean (with a warning), and retried on the next run
  (`--rejudge_failed=False` keeps it as is). The original scripts recorded such failures as 0.0.
- **Parsing:** a bare number is used directly; otherwise the first number that looks like a score
  is taken. Results are clamped to [0, 1].

### Programmatic scoring

```python
from evaloop.analysis.analyzer import ResultAnalyzer
from evaloop.analysis.similarity import JudgeConfig

analyzer = ResultAnalyzer(
    "results/exp_results.json",
    "results/analysis",
    judge_config=JudgeConfig(model="gpt-4-turbo-2024-04-09", api_key="sk-..."),
)
results = analyzer.analyze()          # all metrics; results["ASL"]["by_model"][model_name]
```

The pure metric functions (`asl_semantic`, `asl_pow`, `asl_bias`, `pass_at_k`, `pass_drop`,
`bucket_similarity`) live in `evaloop.analysis.metrics`.

## 🧪 Advanced Usage

### Custom Model Registration

```python
from evaloop.models.registry import ModelRegistry

registry = ModelRegistry()
registry.register_model("my-custom-model", {
    "name": "my-custom-model",
    "type": "vllm",
    "path": "path/to/my/model",
    "max_model_len": 4096,
    "tensor_parallel_size": 2
})
```

### Programmatic API

```python
from evaloop.core.config import EvaluationConfig
from evaloop.core.evaluator import EvaLoopEvaluator

# Create configuration
config = EvaluationConfig(
    models=["gpt-4", "deepseek-coder-33b"],
    dataset="mbpp_plus",
    max_cycles=10,
    temperature=0.0
)

# Run evaluation
evaluator = EvaLoopEvaluator(config)
results = evaluator.run_code_generation_summarization()
```

### Batch Processing

```bash
# Process multiple experiments
for model in "gpt-4" "gpt-3.5-turbo" "deepseek-coder-33b"; do
    evaloop evaluate --models "$model" \
                     --experiment_name "batch_${model}" \
                     --output_dir "results/batch/"
done

# Analyze all results
for result in results/batch/*_results.json; do
    evaloop analyze --results_path "$result"
done
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/your-org/evaloop
cd evaloop
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest tests/
pytest tests/ -m "not slow"  # Skip slow integration tests
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Python Fire](https://github.com/google/python-fire) for CLI interface
- Uses [VLLM](https://github.com/vllm-project/vllm) for efficient model inference
- Evaluation datasets from [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) and [HumanEval](https://github.com/openai/human-eval)
- Code testing powered by [EvalPlus](https://github.com/evalplus/evalplus)

## 📞 Support

- 📖 [Documentation](https://evaloop.readthedocs.io/)
- 🐛 [Issue Tracker](https://github.com/your-org/evaloop/issues)
- 💬 [Discussions](https://github.com/your-org/evaloop/discussions)

---

**EvaLoop** - Robust LLM Evaluation Made Simple 🔄
