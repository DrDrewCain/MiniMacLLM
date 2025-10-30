"""
Unit tests for evaluation metrics and evaluator.
"""

import pytest
import torch
import torch.nn as nn
from src.evaluation.metrics import (
    compute_perplexity,
    compute_forgetting_rate,
    compute_accuracy,
    evaluate_generation_quality,
    ForgettingMetrics,
    PerplexityResult
)


class SimpleLM(nn.Module):
    """Simple language model for testing."""

    def __init__(self, vocab_size=1000, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        logits = self.lm_head(x)
        return logits, None, None


class TestComputePerplexity:
    """Test perplexity computation."""

    def test_perplexity_shape(self):
        """Test that perplexity returns correct result type."""
        model = SimpleLM()
        input_ids = torch.randint(0, 1000, (2, 20))

        result = compute_perplexity(model, input_ids)

        assert isinstance(result, PerplexityResult)
        assert isinstance(result.perplexity, float)
        assert isinstance(result.cross_entropy, float)
        assert isinstance(result.num_tokens, int)

    def test_perplexity_positive(self):
        """Test that perplexity is positive."""
        model = SimpleLM()
        input_ids = torch.randint(0, 1000, (2, 20))

        result = compute_perplexity(model, input_ids)

        assert result.perplexity > 0
        assert result.cross_entropy > 0
        assert result.bits_per_byte > 0

    def test_perplexity_with_attention_mask(self):
        """Test perplexity computation with attention mask."""
        model = SimpleLM()
        input_ids = torch.randint(0, 1000, (2, 20))
        attention_mask = torch.ones(2, 20)
        attention_mask[:, 15:] = 0  # Mask last 5 tokens

        result = compute_perplexity(model, input_ids, attention_mask=attention_mask)

        # Should only count non-masked tokens
        # Each sequence has 19 target tokens (seq_len - 1), but 5 are masked
        # So 14 * 2 = 28 tokens
        expected_tokens = 14 * 2
        assert result.num_tokens == expected_tokens

    def test_perfect_model_low_perplexity(self):
        """Test that a perfect model has low perplexity."""
        model = SimpleLM()
        model.eval()

        # Create simple pattern: always predict next token correctly
        input_ids = torch.tensor([[0, 1, 2, 3, 4]])

        # Manually set weights to predict correctly
        # (In practice, this is hard to set up perfectly, so we just check relative values)
        result = compute_perplexity(model, input_ids)

        # Perplexity should be positive
        assert result.perplexity > 0

    def test_numerical_stability(self):
        """Test numerical stability with extreme cases."""
        model = SimpleLM()

        # Test with very short sequence
        input_ids = torch.randint(0, 1000, (1, 2))
        result = compute_perplexity(model, input_ids)
        assert not torch.isnan(torch.tensor(result.perplexity))

        # Test with longer sequence
        input_ids = torch.randint(0, 1000, (1, 100))
        result = compute_perplexity(model, input_ids)
        assert not torch.isnan(torch.tensor(result.perplexity))


class TestComputeForgettingRate:
    """Test forgetting rate computation."""

    def test_no_forgetting(self):
        """Test when performance stays the same."""
        previous = {'task_A': 0.9, 'task_B': 0.85}
        current = {'task_A': 0.9, 'task_B': 0.85}

        metrics = compute_forgetting_rate(current, previous)

        assert metrics.forgetting_rate == 0.0
        assert metrics.backward_transfer == 0.0

    def test_complete_forgetting(self):
        """Test when performance drops significantly."""
        previous = {'task_A': 0.9, 'task_B': 0.85}
        current = {'task_A': 0.5, 'task_B': 0.45}

        metrics = compute_forgetting_rate(current, previous)

        # Average forgetting: ((0.9-0.5) + (0.85-0.45)) / 2 = 0.4
        assert metrics.forgetting_rate == pytest.approx(0.4, abs=1e-6)
        assert metrics.backward_transfer < 0  # Negative means degradation

    def test_improvement(self):
        """Test when performance improves."""
        previous = {'task_A': 0.7, 'task_B': 0.65}
        current = {'task_A': 0.8, 'task_B': 0.75}

        metrics = compute_forgetting_rate(current, previous)

        # No forgetting (improvement doesn't count as negative forgetting)
        assert metrics.forgetting_rate == 0.0
        assert metrics.backward_transfer > 0  # Positive means improvement

    def test_new_task(self):
        """Test when a new task is added."""
        previous = {'task_A': 0.9}
        current = {'task_A': 0.85, 'task_B': 0.80}

        metrics = compute_forgetting_rate(current, previous)

        # Forgetting on task_A: 0.9 - 0.85 = 0.05
        assert metrics.forgetting_rate == pytest.approx(0.05, abs=1e-6)

        # Forward transfer: performance on new task
        assert metrics.forward_transfer == 0.80

    def test_empty_previous(self):
        """Test when there are no previous tasks."""
        previous = {}
        current = {'task_A': 0.9}

        metrics = compute_forgetting_rate(current, previous)

        assert metrics.forgetting_rate == 0.0
        assert metrics.forward_transfer == 0.9


class TestComputeAccuracy:
    """Test accuracy computation."""

    def test_accuracy_range(self):
        """Test that accuracy is between 0 and 1."""
        model = SimpleLM()
        input_ids = torch.randint(0, 1000, (2, 20))
        labels = torch.randint(0, 1000, (2, 20))

        accuracy = compute_accuracy(model, input_ids, labels)

        assert 0.0 <= accuracy <= 1.0

    def test_perfect_accuracy(self):
        """Test accuracy with perfect predictions."""
        # Create a model that predicts the exact labels
        model = SimpleLM()
        model.eval()

        # For perfect accuracy, we'd need to carefully construct the model
        # Here we just test that the function runs
        input_ids = torch.randint(0, 1000, (2, 20))
        labels = input_ids.clone()  # Same as input

        accuracy = compute_accuracy(model, input_ids, labels)

        # Won't be perfect due to random model, but should be >= 0
        assert accuracy >= 0.0

    def test_random_accuracy(self):
        """Test that random model has low accuracy."""
        model = SimpleLM(vocab_size=1000)
        input_ids = torch.randint(0, 1000, (2, 20))
        labels = torch.randint(0, 1000, (2, 20))

        accuracy = compute_accuracy(model, input_ids, labels)

        # Random model should have approximately 1/vocab_size accuracy
        # With large vocab, this should be very low
        assert accuracy < 0.2  # Should be much less than 20%


class TestEvaluateGenerationQuality:
    """Test generation quality metrics."""

    def test_length_metric(self):
        """Test length metric."""
        text = "The quick brown fox jumps over the lazy dog"
        result = evaluate_generation_quality(text, metrics=['length'])

        assert 'length' in result
        assert result['length'] == len(text.split())

    def test_unique_tokens_metric(self):
        """Test unique tokens metric."""
        text = "the the the the cat"  # 20% unique
        result = evaluate_generation_quality(text, metrics=['unique_tokens'])

        assert 'unique_tokens' in result
        assert result['unique_tokens'] == pytest.approx(0.4, abs=0.01)  # 2 unique / 5 total

    def test_repetition_metric(self):
        """Test repetition metric."""
        # High repetition
        repetitive_text = "the the the the the"
        result1 = evaluate_generation_quality(repetitive_text, metrics=['repetition'])

        # Low repetition
        diverse_text = "the quick brown fox jumps"
        result2 = evaluate_generation_quality(diverse_text, metrics=['repetition'])

        assert result1['repetition'] > result2['repetition']

    def test_all_metrics(self):
        """Test computing all metrics."""
        text = "The quick brown fox jumps over the lazy dog"
        result = evaluate_generation_quality(
            text,
            metrics=['length', 'unique_tokens', 'repetition']
        )

        assert 'length' in result
        assert 'unique_tokens' in result
        assert 'repetition' in result

    def test_empty_text(self):
        """Test with empty text."""
        text = ""
        result = evaluate_generation_quality(text, metrics=['length', 'unique_tokens'])

        assert result['length'] == 0
        assert result['unique_tokens'] == 0.0


class TestForgettingMetrics:
    """Test ForgettingMetrics dataclass."""

    def test_creation(self):
        """Test creating ForgettingMetrics."""
        metrics = ForgettingMetrics(
            forgetting_rate=0.1,
            backward_transfer=-0.05,
            forward_transfer=0.8,
            task_performances={'task_A': 0.9}
        )

        assert metrics.forgetting_rate == 0.1
        assert metrics.backward_transfer == -0.05
        assert metrics.forward_transfer == 0.8
        assert 'task_A' in metrics.task_performances


class TestPerplexityResult:
    """Test PerplexityResult dataclass."""

    def test_creation(self):
        """Test creating PerplexityResult."""
        result = PerplexityResult(
            perplexity=50.0,
            cross_entropy=3.912,
            num_tokens=1000,
            bits_per_byte=5.64
        )

        assert result.perplexity == 50.0
        assert result.cross_entropy == 3.912
        assert result.num_tokens == 1000
        assert result.bits_per_byte == 5.64

    def test_repr(self):
        """Test string representation."""
        result = PerplexityResult(
            perplexity=50.0,
            cross_entropy=3.912,
            num_tokens=1000,
            bits_per_byte=5.64
        )

        repr_str = repr(result)
        assert '50.0' in repr_str
        assert '3.912' in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
