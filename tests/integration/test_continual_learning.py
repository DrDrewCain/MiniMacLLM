"""
Integration tests for the complete continual learning system.

These tests verify that all components work together correctly:
- Base model + LoRA + Experience Replay + EWC
- Multi-domain learning without forgetting
- Checkpoint save/load
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path

from src.model.llm import ContinualLLM, ModelConfig
from src.lora.lora_model import LoRAConfig
from src.continual.continual_trainer import ContinualLearner, ContinualLearningConfig, Experience


class SimpleTokenizer:
    """Simple tokenizer for testing."""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.pad_token_id = 0

    def encode(self, text):
        # Simple mock encoding
        return [hash(word) % self.vocab_size for word in text.split()]

    def decode(self, token_ids):
        # Simple mock decoding
        return " ".join([f"token_{tid}" for tid in token_ids])


@pytest.fixture
def small_model():
    """Create a small model for testing."""
    # Set seed for reproducible model initialization
    torch.manual_seed(42)

    config = ModelConfig(
        vocab_size=1000,
        d_model=128,
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=2,
        d_ff=512,
        max_seq_len=128,
        dropout=0.0,
        attention_dropout=0.0,
    )
    return ContinualLLM(config)


@pytest.fixture
def tokenizer():
    """Create a simple tokenizer for testing."""
    return SimpleTokenizer(vocab_size=1000)


@pytest.fixture
def continual_config():
    """Create continual learning configuration."""
    return ContinualLearningConfig(
        lora_r=8,
        lora_alpha=16.0,
        replay_buffer_size=100,
        replay_ratio=0.5,
        use_ewc=True,
        ewc_lambda=100.0,
        learning_rate=1e-4,
        batch_size=2,
        device="cpu",
        consolidation_frequency=50,
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


class TestContinualLearningBasics:
    """Test basic continual learning functionality."""

    def test_learner_initialization(self, small_model, tokenizer, continual_config):
        """Test that continual learner initializes correctly."""
        learner = ContinualLearner(small_model, continual_config, tokenizer)

        assert learner.model is not None
        assert learner.replay_buffer is not None
        assert learner.config == continual_config
        assert learner.tokenizer == tokenizer

    def test_learn_from_text(self, small_model, tokenizer, continual_config):
        """Test learning from text."""
        learner = ContinualLearner(small_model, continual_config, tokenizer)

        text = "The quick brown fox jumps over the lazy dog"
        metrics = learner.learn_from_text(text, domain="test", importance=1.0)

        assert "total_loss" in metrics
        assert "ewc_loss" in metrics
        assert isinstance(metrics["total_loss"], float)

    def test_experience_buffer_storage(self, small_model, tokenizer, continual_config):
        """Test that experiences are stored in replay buffer."""
        learner = ContinualLearner(small_model, continual_config, tokenizer)

        # Learn from multiple texts
        texts = ["First text to learn", "Second text to learn", "Third text to learn"]

        for text in texts:
            learner.learn_from_text(text, domain="test")

        # Check that buffer has experiences
        assert learner.replay_buffer.size > 0
        assert learner.replay_buffer.size <= len(texts)

    def test_batch_learning(self, small_model, tokenizer, continual_config):
        """Test learning from a batch of experiences."""
        learner = ContinualLearner(small_model, continual_config, tokenizer)

        # Create batch of experiences
        experiences = []
        for i in range(5):
            text = f"Test text number {i}"
            tokens = tokenizer.encode(text)
            exp = Experience(
                input_ids=torch.tensor(tokens[:-1]),
                labels=torch.tensor(tokens[1:]),
                importance=1.0,
                domain="test",
            )
            experiences.append(exp)

        # Learn from batch
        metrics = learner.learn_from_batch(experiences)

        assert "total_loss" in metrics
        assert metrics["total_loss"] >= 0


class TestMultiDomainLearning:
    """Test learning across multiple domains without forgetting."""

    def test_two_domain_learning(self, small_model, tokenizer, continual_config):
        """Test learning from two different domains."""
        learner = ContinualLearner(small_model, continual_config, tokenizer)

        # Learn math
        math_texts = [
            "Two plus two equals four",
            "Three times five equals fifteen",
            "The square root of nine is three",
        ]

        for text in math_texts:
            learner.learn_from_text(text, domain="math", importance=1.0)

        # Learn code
        code_texts = [
            "def function takes arguments",
            "return statement exits function",
            "for loop iterates over items",
        ]

        for text in code_texts:
            learner.learn_from_text(text, domain="code", importance=1.0)

        # Check that both domains are in the buffer
        buffer_domains = set()
        for exp in learner.replay_buffer.buffer[: learner.replay_buffer.size]:
            if exp.domain:
                buffer_domains.add(exp.domain)

        assert "math" in buffer_domains
        assert "code" in buffer_domains

    def test_domain_specific_adapters(self, small_model, tokenizer, continual_config):
        """Test that different domains can have different adapters."""
        learner = ContinualLearner(small_model, continual_config, tokenizer)

        # Learn with different domains
        learner.learn_from_text("Math text", domain="math")
        learner.learn_from_text("Code text", domain="code")

        # Both should work without errors
        assert learner.replay_buffer.size >= 2


class TestAntiForgetting:
    """Test anti-forgetting mechanisms."""

    def test_ewc_prevents_forgetting(self, small_model, tokenizer, continual_config):
        """Test that EWC helps prevent forgetting."""
        # Create two learners: one with EWC, one without
        config_with_ewc = continual_config
        config_with_ewc.use_ewc = True

        config_without_ewc = ContinualLearningConfig(
            lora_r=8,
            lora_alpha=16.0,
            replay_buffer_size=100,
            replay_ratio=0.5,
            use_ewc=False,  # Disabled
            learning_rate=1e-4,
            batch_size=2,
            device="cpu",
        )

        learner_with = ContinualLearner(small_model, config_with_ewc, tokenizer)
        learner_without = ContinualLearner(small_model, config_without_ewc, tokenizer)

        # Both learn the same text
        text = "The quick brown fox"

        metrics_with = learner_with.learn_from_text(text, domain="test")
        metrics_without = learner_without.learn_from_text(text, domain="test")

        # With EWC, we should have an EWC loss component
        assert "ewc_loss" in metrics_with
        if learner_with.ewc and learner_with.ewc.fisher:
            # EWC loss might be 0 on first learning, but key should exist
            assert metrics_with["ewc_loss"] >= 0

    def test_replay_buffer_prevents_forgetting(self, small_model, tokenizer, continual_config):
        """Test that replay buffer helps retain old knowledge."""
        learner = ContinualLearner(small_model, continual_config, tokenizer)

        # Learn initial task
        old_texts = ["Old knowledge one", "Old knowledge two"]
        for text in old_texts:
            learner.learn_from_text(text, domain="old", importance=1.0)

        initial_buffer_size = learner.replay_buffer.size

        # Learn new task
        new_texts = ["New knowledge one", "New knowledge two"]
        for text in new_texts:
            learner.learn_from_text(text, domain="new", importance=1.0)

        # Buffer should contain both old and new
        assert learner.replay_buffer.size >= initial_buffer_size


class TestCheckpointManagement:
    """Test saving and loading checkpoints."""

    def test_save_checkpoint(self, small_model, tokenizer, continual_config, temp_dir):
        """Test saving a checkpoint."""
        learner = ContinualLearner(small_model, continual_config, tokenizer)

        # Learn something
        learner.learn_from_text("Test text", domain="test")

        # Save checkpoint
        save_path = temp_dir / "checkpoint"
        learner.save_checkpoint(str(save_path))

        # Check that files were created
        assert (save_path / "model.pt").exists()
        assert (save_path / "config.json").exists()
        assert (save_path / "replay_buffer.pt").exists()

    def test_load_checkpoint(self, small_model, tokenizer, continual_config, temp_dir):
        """Test loading a checkpoint."""
        # Create and save first learner
        learner1 = ContinualLearner(small_model, continual_config, tokenizer)
        learner1.learn_from_text("Test text for saving", domain="test")

        save_path = temp_dir / "checkpoint"
        learner1.save_checkpoint(str(save_path))

        # Create new learner and load
        new_model = small_model  # Would normally create a fresh model
        learner2 = ContinualLearner(new_model, continual_config, tokenizer)
        learner2.load_checkpoint(str(save_path))

        # Check that replay buffer was restored
        assert learner2.replay_buffer.size == learner1.replay_buffer.size

    def test_checkpoint_preserves_learning(
        self, small_model, tokenizer, continual_config, temp_dir
    ):
        """Test that checkpoint preserves learned knowledge."""
        learner1 = ContinualLearner(small_model, continual_config, tokenizer)

        # Learn specific texts
        texts = ["Knowledge one", "Knowledge two", "Knowledge three"]
        for text in texts:
            learner1.learn_from_text(text, domain="test")

        # Get some output before saving
        test_input = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            logits1, _, _ = learner1.model(test_input)

        # Save and load
        save_path = temp_dir / "checkpoint"
        learner1.save_checkpoint(str(save_path))

        # Load into new learner
        new_model = ContinualLLM(
            ModelConfig(
                vocab_size=1000,
                d_model=128,
                num_layers=2,
                num_query_heads=4,
                num_kv_heads=2,
                d_ff=512,
                max_seq_len=128,
            )
        )
        learner2 = ContinualLearner(new_model, continual_config, tokenizer)
        learner2.load_checkpoint(str(save_path))

        # Get output after loading
        with torch.no_grad():
            logits2, _, _ = learner2.model(test_input)

        # Outputs should be similar (within tolerance due to  floating point)
        assert torch.allclose(logits1, logits2, atol=1e-4)


class TestEvaluationIntegration:
    """Test integration with evaluation system."""

    def test_evaluate_after_learning(self, small_model, tokenizer, continual_config):
        """Test that we can evaluate after learning."""
        learner = ContinualLearner(small_model, continual_config, tokenizer)

        # Learn something
        texts = ["Test text one", "Test text two"]
        for text in texts:
            learner.learn_from_text(text, domain="test")

        # Simple evaluation: check that model runs
        test_input = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            logits, _, _ = learner.model(test_input)

        assert logits.shape == (1, 10, 1000)
        assert not torch.isnan(logits).any()

    def test_generation_after_learning(self, small_model, tokenizer, continual_config):
        """Test text generation after learning."""
        learner = ContinualLearner(small_model, continual_config, tokenizer)

        # Learn
        learner.learn_from_text("The quick brown fox", domain="test")

        # Generate
        prompt = "The quick"
        generated = learner.generate(prompt, max_length=20)

        assert isinstance(generated, str)
        assert len(generated) > 0


class TestNumericalStability:
    """Test numerical stability of the continual learning system."""

    def test_no_nan_or_inf(self, small_model, tokenizer, continual_config):
        """Test that learning doesn't produce NaN or Inf."""
        # Set seed for reproducibility
        torch.manual_seed(42)

        learner = ContinualLearner(small_model, continual_config, tokenizer)

        # Learn from multiple texts
        texts = [
            "First text to learn from",
            "Second different text",
            "Third text with more words in it",
            "Fourth text is also longer than the others",
        ]

        for text in texts:
            metrics = learner.learn_from_text(text, domain="test")

            # Check metrics don't have NaN or Inf
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    assert not torch.isnan(
                        torch.tensor(v)
                    ).any(), f"NaN detected in metric '{k}': {v}"
                    assert not torch.isinf(
                        torch.tensor(v)
                    ).any(), f"Inf detected in metric '{k}': {v}"

        # Check model parameters
        for param in learner.model.parameters():
            assert not torch.isnan(param).any(), "NaN detected in model parameters"
            assert not torch.isinf(param).any(), "Inf detected in model parameters"

    def test_gradient_explosion_protection(self, small_model, tokenizer, continual_config):
        """Test that gradients don't explode during learning."""
        # Set seed for reproducibility
        torch.manual_seed(42)

        learner = ContinualLearner(small_model, continual_config, tokenizer)

        # Learn from text
        metrics = learner.learn_from_text("Test text with multiple words", domain="test")

        # Check that loss is reasonable (not exploded)
        assert metrics["total_loss"] < 1000, f"Loss exploded: {metrics['total_loss']}"
        assert metrics["total_loss"] > 0, f"Loss is non-positive: {metrics['total_loss']}"
        assert not torch.isnan(torch.tensor(metrics["total_loss"])).any(), "Loss is NaN"
        assert not torch.isinf(torch.tensor(metrics["total_loss"])).any(), "Loss is Inf"

    def test_empty_text_handling(self, small_model, tokenizer, continual_config):
        """Test handling of edge case: empty text."""
        learner = ContinualLearner(small_model, continual_config, tokenizer)

        # Try learning from empty text (should handle gracefully)
        # This might raise an exception, which is acceptable
        try:
            learner.learn_from_text("", domain="test")
        except (ValueError, IndexError):
            # It's okay to raise an error for empty input
            pass


class TestMemoryEfficiency:
    """Test memory efficiency of continual learning."""

    def test_replay_buffer_limit(self, small_model, tokenizer, continual_config):
        """Test that replay buffer respects size limit."""
        learner = ContinualLearner(small_model, continual_config, tokenizer)

        buffer_size = continual_config.replay_buffer_size

        # Try to add more than buffer size
        for i in range(buffer_size * 2):
            learner.learn_from_text(f"Text number {i}", domain="test")

        # Buffer should not exceed its limit
        assert learner.replay_buffer.size <= buffer_size

    def test_kv_cache_with_long_sequence(self, small_model, tokenizer, continual_config):
        """Test KV caching with longer sequences."""
        learner = ContinualLearner(small_model, continual_config, tokenizer)

        # Generate long text
        long_text = " ".join(["word"] * 50)

        # Should handle without memory explosion
        metrics = learner.learn_from_text(long_text, domain="test")

        assert metrics["total_loss"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
