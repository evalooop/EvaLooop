# EvaLoop: LLM Robustness Evaluation Framework

This repository contains the implementation for evaluating the robustness of Large Language Models (LLMs) through iterative dual-task cycles. The framework measures how many cycles an LLM can execute before generating code that fails functional testing.

## Abstract

We present EvaLoop, a novel evaluation framework that assesses LLM robustness through iterative task cycles. The system alternates between complementary tasks such as code generation and summarization, or code translation between programming languages. By measuring the number of successful cycles before failure, we provide a quantitative metric for LLM robustness across different models and tasks.

## Methodology

The evaluation pipeline works through the following steps:

1. **Initial Input**: Start with a natural language prompt describing a coding task
2. **Task Execution**: Generate code from the prompt using the target LLM
3. **Functional Testing**: Execute the generated code against predefined test cases
4. **Task Alternation**: Use the output as input for the complementary task (e.g., code → summary → code)
5. **Iteration**: Continue cycling through tasks until code fails testing or maximum cycles reached
6. **Robustness Scoring**: The robustness score equals the number of successful cycles completed

## Supported Tasks

- **Code Generation**: Create executable code from natural language descriptions
- **Code Summarization**: Generate natural language descriptions from source code  
- **Code Translation**: Convert code between programming languages (Python ↔ Java)

## Supported Models

### Open-Source Models
- LLaMA series (via Hugging Face Transformers)
- DeepSeek Coder series
- Qwen series (via vLLM for large models)
- Any Hugging Face compatible model

### Closed-Source Models  
- OpenAI GPT series (GPT-3.5-turbo, GPT-4, GPT-4o)
- Support for other API-based models can be easily added

## Project Structure

```
├── config/                 # Model and task configurations
│   ├── models.yaml         # Model specifications and parameters
│   └── tasks.yaml          # Task definitions and prompt templates
├── src/                    # Core framework implementation
│   ├── models/             # Model interface implementations
│   ├── tasks/              # Task-specific logic
│   ├── evaluation/         # Evaluation pipeline and testing
│   └── utils/              # Utility functions
├── data/                   # Test cases and datasets
│   └── test_cases/         # Functional test suites
├── experiments/            # Experiment configurations
├── scripts/                # Execution and analysis scripts
└── results/                # Output directory for results
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd EvaLoop

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Configure Models

Edit `config/models.yaml` to specify your target models:

```yaml
# Example: Open-source model
llama-3-70b:
  name: llama-3-70b
  type: huggingface
  path: meta-llama/Llama-3-70b-instruct-hf
  device: cuda
  max_length: 4096

# Example: Closed-source model  
gpt-4o:
  name: gpt-4o
  type: openai
  model_id: gpt-4o
  api_key: ${OPENAI_API_KEY}
  max_tokens: 4096
```

### 2. Set Up Environment Variables

```bash
# For OpenAI models
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run Experiments

```bash
# Run code generation + summarization experiment
python scripts/run_experiment_cgs.py experiments/generation_summarization.yaml

# Run code translation experiment  
python scripts/run_experiment_ct.py experiments/code_translation.yaml

# Run with custom configuration
python scripts/run_experiment_test.py your_experiment.yaml --output-dir results/
```

### 4. Analyze Results

```bash
# Generate analysis plots and statistics
python scripts/analyze_results.py results/your_experiment_results.yaml
```

## Experiment Configuration

Create experiment YAML files specifying:

```yaml
name: your_experiment_name
description: "Experiment description"

models:
  - name: model1
    # model configuration
  - name: model2  
    # model configuration

tasks:
  - type: code_generation
    language: python
    prompt_template: "Generate Python code for: {description}"
  - type: code_summarization
    prompt_template: "Summarize this code: {code}"

max_cycles: 10
prompts:
  - "Write a function that calculates Fibonacci numbers"
  - "Create a palindrome checker function"
  - "Implement binary search algorithm"
```

## Key Features

- **Modular Design**: Easy to add new models, tasks, and evaluation metrics
- **Multi-GPU Support**: Efficient inference for large models using vLLM
- **Comprehensive Testing**: Functional testing with execution sandboxing
- **Rich Analytics**: Detailed result analysis with visualization tools
- **Extensible Framework**: Plugin architecture for custom tasks and models
