# EvaLoop: LLM Robustness Evaluation Framework

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

# Analyze results
evaloop analyze --results_path "results/experiment_results.json" \
                 --generate_plots
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
6. **Robustness Scoring**: The robustness score equals the number of successful cycles completed

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
- `--metrics`: Metrics to compute (default: `"ASL_std,ASL_base,pass_rate"`)
- `--generate_plots`: Generate visualization plots (default: True)
- `--output_dir`: Analysis output directory

**Examples:**

```bash
# Basic analysis with plots
evaloop analyze --results_path "results/my_experiment_results.json"

# Custom metrics and output location
evaloop analyze --results_path "results/exp.json" \
                 --metrics "ASL_std,pass_rate" \
                 --output_dir "analysis/custom/"
```

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

- **ASL_std**: Average Successful Length (all tasks)
- **ASL_base**: Average Successful Length (tasks with ≥1 successful cycle)
- **pass_rate**: Percentage of tasks completing at least one cycle

### Visualization

Analysis generates several plots:
- Model comparison bar charts
- Cycle distribution histograms  
- Task success heatmaps
- Performance trend analysis

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
