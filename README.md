# EvaLooop: LLM Robustness Evaluation Framework

[![arXiv](https://img.shields.io/badge/arXiv-2505.12185-b31b1b.svg)](https://arxiv.org/abs/2505.12185)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Leaderboard](https://img.shields.io/badge/leaderboard-evalooop.github.io-blue)](https://evalooop.github.io/)

EvaLoop is a framework for evaluating the robustness of Large Language Models (LLMs) through iterative dual-task cycles. The framework measures how many cycles an LLM can execute before generating code that fails functional testing, providing a quantitative metric for model robustness. It accompanies the paper [*EvaLoop: Assessing LLM Robustness in Programming from a Self-consistency Perspective*](https://arxiv.org/abs/2505.12185); results across models are published on the [leaderboard](https://evalooop.github.io/).

## 🚀 Quick Start

### Installation

EvaLoop is not yet published on PyPI — install from source:

```bash
git clone https://github.com/evalooop/EvaLooop
cd EvaLooop
pip install -e .
```

Optional extras:

```bash
pip install -e ".[vllm]"   # GPU batch inference via vLLM (Linux + CUDA only)
pip install -e ".[dev]"    # development tools (pytest, ruff, ...)
```

After installing, check your environment:

```bash
evaloop validate_setup                    # full check
evaloop validate_setup --check_gpu=False  # skip GPU checks (e.g. on a laptop)
```

### Basic Usage

```bash
# Evaluate an OpenAI model on the generation<->summarization loop
evaloop evaluate --model "gpt-4"

# Evaluate an open-source model with multi-GPU vLLM
evaloop evaluate --model "Qwen/Qwen2.5-Coder-32B-Instruct" \
                 --gpu_ids "0,1,2,3" \
                 --tensor_parallel_size 4 \
                 --max_cycles 15

# Analyze results
evaloop analyze --results_path "results/experiment_results.json" \
                --generate_plots
```

## 🔄 How the Loop Works

Each evaluation cycle alternates between two tasks:

1. **Initial input**: a natural-language prompt describing a coding task (from MBPP+), formatted as a docstring whose final line is an `assert` statement.
2. **Code generation**: the model generates code from the prompt.
3. **Functional testing**: the generated code is executed against the dataset's test cases ([EvalPlus](https://github.com/evalplus/evalplus) MBPP+).
4. **Code summarization**: the model summarizes the generated code back into a natural-language description.
5. **Iteration**: the summary becomes the next cycle's prompt; the loop repeats until the generated code fails testing or `--max_cycles` is reached.

The number of cycles a model survives, averaged over the dataset, is its **ASL (Average Successful Loops)** score.

### Design note: the retained assert line

The summarization step deliberately re-attaches the final `assert` line from the prompt the cycle *started* with, so each new prompt looks like:

```
"""
<model-generated summary>
assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
"""
```

This is intentional, not an artifact: keeping one assert anchors the function name and signature across cycles, so the loop measures **semantic drift** in the model's understanding rather than trivial failures from a renamed function. You will see this line in every generated prompt in the result data. The implementation lives in `CodeSummarizationTask.postprocess` (`evaloop/tasks/code_summarization.py`).

### Supported Task Types

| Task type | Config name | Status |
|---|---|---|
| Code generation | `code_generation` | ✅ Supported |
| Code summarization | `code_summarization` | ✅ Supported |
| Code translation | `code_translation` | 🧪 Experimental — the task class exists, but config/evaluator support is not yet in the public release (`run_code_translation()` raises `NotImplementedError`) |

Want to add your own loop? A new task type is one new file plus a registry entry — no changes to the loop runner. See the [Contributing Guide](CONTRIBUTING.md).

## 🛠️ CLI Reference

### `evaluate` — Code Generation ↔ Summarization

```bash
evaloop evaluate --model MODEL [OPTIONS]
```

**Key options:**
- `--model`: A single model name or HuggingFace path (required). Examples: `"gpt-4"`, `"Qwen/Qwen2.5-Coder-32B-Instruct"`
- `--dataset`: Dataset to use. Currently only `mbpp_plus` is supported (HumanEval is planned).
- `--max_cycles`: Maximum evaluation cycles (default: 10)
- `--temperature`: Generation temperature (default: 0.0)
- `--output_dir`: Results output directory (default: `results/`)
- `--gpu_ids`: GPU IDs for vLLM models (e.g., `"0,1,2,3"`)
- `--tensor_parallel_size`: GPUs for tensor parallelism (default: 1)

**Examples:**

```bash
# Basic OpenAI evaluation
evaloop evaluate --model "gpt-4"

# Large model with multi-GPU setup
evaloop evaluate --model "meta-llama/Llama-3-70b-instruct-hf" \
                 --gpu_ids "0,1,2,3" \
                 --tensor_parallel_size 4 \
                 --gpu_memory_utilization 0.85

# Custom experiment parameters
evaloop evaluate --model "deepseek-coder-33b" \
                 --max_cycles 20 \
                 --temperature 0.2 \
                 --experiment_name "deepseek_robust_eval"
```

To evaluate several models, run `evaloop evaluate` once per model (see [Batch Processing](#batch-processing)).

### `analyze` — Result Analysis

```bash
evaloop analyze --results_path "results/experiment_results.json" [OPTIONS]
```

**Key options:**
- `--results_path`: Path to results JSON file (required)
- `--metrics`: Metrics to compute (default: `"ASL_std,ASL_base"`)
- `--generate_plots`: Generate visualization plots (default: True)
- `--output_dir`: Analysis output directory

### `list_models` — Available Models

```bash
evaloop list_models
```

### `validate_setup` — System Validation

```bash
evaloop validate_setup [--check_gpu=False] [--check_api_keys=False]
```

## 🔧 Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional
```

### Model Configuration

EvaLoop automatically configures models based on their names/paths:

- **OpenAI models**: `gpt-4`, `gpt-3.5-turbo`, `gpt-4-turbo`
- **HuggingFace paths**: `meta-llama/Llama-3-70b-instruct-hf`
- **Pre-registered names**: use `evaloop list_models` to see available options

Large models automatically use vLLM for efficient inference when it is installed (`pip install -e ".[vllm]"`).

## 📊 Results and Analysis

### Output Structure

Results are saved in JSON format:

```json
{
  "model": "gpt-4",
  "prompt_results": [
    {
      "task_id": "Mbpp/2",
      "initial_prompt": "Write a function to find the shared elements ...",
      "cycles": [...],
      "successful_cycles": 5,
      "max_cycles_reached": false
    }
  ],
  "average_successful_cycles": 4.2
}
```

### Metrics

- **ASL_std**: Average Successful Loops (all tasks)
- **ASL_base**: Average Successful Loops (tasks with ≥1 successful cycle)
- **pass_rate**: Percentage of tasks completing at least one cycle

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

config = EvaluationConfig(
    model="gpt-4",
    dataset="mbpp_plus",
    max_cycles=10,
    temperature=0.0,
)

evaluator = EvaLoopEvaluator(config)
results = evaluator.run_code_generation_summarization()
```

### Batch Processing

```bash
# Process multiple experiments
for model in "gpt-4" "gpt-3.5-turbo" "deepseek-coder-33b"; do
    evaloop evaluate --model "$model" \
                     --experiment_name "batch_${model}" \
                     --output_dir "results/batch/"
done

# Analyze all results
for result in results/batch/*_results.json; do
    evaloop analyze --results_path "$result"
done
```

## 🤝 Contributing

We welcome contributions — new loop/task types especially! Please see the [Contributing Guide](CONTRIBUTING.md) for the development setup, the "add a new loop type" walkthrough, and PR expectations.

### Development Setup

```bash
git clone https://github.com/evalooop/EvaLooop
cd EvaLooop
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
ruff check evaloop tests
```

The unit test suite runs without GPU, API keys, or the heavy model dependencies.

## 📖 Citation

If you use EvaLoop in your research, please cite:

```bibtex
@article{fang2024evaloop,
  title={EvaLoop: Assessing LLM Robustness in Programming from a Self-consistency Perspective},
  author={Fang, Sen and Ding, Weiyuan and Xu, Bowen},
  journal={arXiv preprint arXiv:2505.12185},
  year={2024},
  url={https://arxiv.org/abs/2505.12185}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Python Fire](https://github.com/google/python-fire) for CLI interface
- Uses [vLLM](https://github.com/vllm-project/vllm) for efficient model inference
- Evaluation dataset from [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)
- Code testing powered by [EvalPlus](https://github.com/evalplus/evalplus)

## 📞 Support

- 🏆 [Leaderboard](https://evalooop.github.io/)
- 📄 [Paper](https://arxiv.org/abs/2505.12185)
- 🐛 [Issue Tracker](https://github.com/evalooop/EvaLooop/issues)

---

**EvaLoop** - Robust LLM Evaluation Made Simple 🔄
