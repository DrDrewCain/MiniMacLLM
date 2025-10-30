"""
Evaluation module for continual learning LLM.

This module provides tools for:
- Computing perplexity and cross-entropy loss
- Measuring catastrophic forgetting
- Evaluating generation quality
- Domain-specific evaluation (math, code, etc.)
"""

from .metrics import (
    compute_perplexity,
    compute_forgetting_rate,
    compute_accuracy,
    evaluate_generation_quality,
    PerplexityResult,
    ForgettingMetrics
)

from .evaluator import (
    Evaluator,
    EvaluationConfig,
    EvaluationResult
)

__all__ = [
    'compute_perplexity',
    'compute_forgetting_rate',
    'compute_accuracy',
    'evaluate_generation_quality',
    'PerplexityResult',
    'ForgettingMetrics',
    'Evaluator',
    'EvaluationConfig',
    'EvaluationResult'
]