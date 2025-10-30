"""
Setup file for the continual learning LLM package.

Install in development mode with:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="continual-llm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A real-time adaptive LLM with continual learning capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Custom_ML_Agent",
    packages=find_packages(include=["src", "src.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "regex>=2023.0.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
        "data": [
            "pdfplumber>=0.10.0",
            "PyPDF2>=3.0.0",
            "pandas>=2.0.0",
        ],
        "training": [
            "transformers>=4.35.0",
            "accelerate>=0.25.0",
            "wandb>=0.16.0",
            "tensorboard>=2.15.0",
        ],
        "all": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pdfplumber>=0.10.0",
            "PyPDF2>=3.0.0",
            "pandas>=2.0.0",
            "transformers>=4.35.0",
            "accelerate>=0.25.0",
            "wandb>=0.16.0",
            "tensorboard>=2.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "continual-llm-train=scripts.pretrain:main",
            "continual-llm-tokenizer=scripts.train_tokenizer:main",
            "continual-llm-learn=scripts.continual_learn:main",
            "continual-llm-generate=scripts.generate:main",
        ],
    },
)
