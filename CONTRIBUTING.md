# Contributing to EvaLoop

Thanks for your interest in contributing! This guide covers the development
setup, what kinds of contributions we're looking for, and what we expect
from a pull request.

## Development Setup

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/<your-username>/EvaLooop
cd EvaLooop
pip install -e ".[dev]"
```

Requirements: Python ≥ 3.9. The `vllm` extra (`pip install -e ".[vllm]"`) is
only needed for GPU batch inference on Linux + CUDA — you do **not** need it
(or a GPU) for development.

Smoke-test your setup:

```bash
evaloop validate_setup --check_gpu=False
```

## Running Tests and Lint

These are exactly what CI runs on your PR:

```bash
pytest
ruff check evaloop tests
```

The unit test suite is designed to run without torch, vLLM, API keys, or any
network access — if your change makes a test require one of those, it belongs
behind the `slow` or `integration` marker.

## What Contributions Are Welcome

- **New loop / task types** (the main way to extend EvaLoop — see below)
- **New dataset adapters** (e.g. HumanEval support)
- **Bug fixes**
- **Documentation improvements**

## Adding a New Loop Type

A new loop type requires **no changes to the loop runner**
(`evaloop/evaluation/cycle.py`). The runner interacts with tasks only through
the `BaseTask` interface: `create_prompt` → model → `extract_output` →
`postprocess`.

1. **Create your task** in a new file `evaloop/tasks/your_task.py`:

   ```python
   from typing import Any, Dict

   from .base import BaseTask
   from .factory import register_task


   @register_task("code_refactoring")
   class CodeRefactoringTask(BaseTask):
       """Example: ask the model to refactor code, feeding it back each cycle."""

       # Set True if this task's output is runnable code that should be
       # executed against the dataset's test cases each cycle.
       produces_testable_code = True

       def create_prompt(self, input_text: str) -> str:
           return f"Refactor the following code, return only code:\n\n{input_text}"

       def extract_output(self, raw_response: str) -> str:
           return raw_response.strip()

       # Optional: post-process the extracted output. `context` includes
       # "cycle_input" (the text the first task of this cycle received),
       # "task_id", and "cycle_number". See CodeSummarizationTask.postprocess
       # for a real example (it re-attaches the prompt's assert line).
   ```

2. **Register it** by adding one import line in `evaloop/tasks/__init__.py`:

   ```python
   from .your_task import CodeRefactoringTask
   ```

3. **Wire up a config**: task configs come from
   `EvaluationConfig.get_task_configs()` in `evaloop/core/config.py` — add a
   branch for your loop's task sequence there (this is currently the one
   place a new loop touches existing code).

4. **Add unit tests** in `tests/` covering `create_prompt`, `extract_output`
   on canned model responses, and (if you use it) `postprocess`. Follow the
   patterns in `tests/test_tasks.py` and `tests/test_cycle.py` — a mocked
   model is enough; tests must not call real APIs.

## Pull Request Expectations

- **One feature per PR.** Small, focused PRs get reviewed much faster.
- **CI must pass** (pytest + ruff). PRs with failing CI will not be reviewed
  until they are green.
- **Include a unit test** for whatever you add or fix.
- **For new loop types**, the PR description must include a short results
  section: which models you ran, on what dataset/subset, and the ASL numbers
  you observed. The PR template has a section for this.

## Review Policy

PRs are reviewed by the maintainers (Sen Fang, Weiyuan Ding, Bowen Xu).
Expected turnaround is **up to two weeks**, especially during high-volume
periods — please be patient, and feel free to ping the thread if a green PR
has been waiting longer than that.
