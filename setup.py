#!/usr/bin/env python3
"""Setup script for EvaLoop package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "fire>=0.5.0",
        "openai>=1.0.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "evalplus>=0.2.0",
        "accelerate>=0.26.0",
        "vllm>=0.2.0",
        "datasets>=2.0.0",
    ]

setup(
    name="evaloop",
    version="0.1.0",
    description="EvaLoop: LLM Robustness Evaluation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="EvaLoop Team",
    author_email="evaloop@example.com",
    url="https://github.com/your-org/evaloop",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "pylint>=2.17.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "evaloop=evaloop.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "evaloop": [
            "data/**/*",
            "config/**/*",
        ],
    },
    keywords=[
        "llm",
        "evaluation",
        "robustness",
        "code-generation",
        "machine-learning",
        "ai",
        "testing",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-org/evaloop/issues",
        "Source": "https://github.com/your-org/evaloop",
        "Documentation": "https://evaloop.readthedocs.io/",
    },
)
