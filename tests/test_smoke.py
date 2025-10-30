"""
Smoke tests - Quick validation that core components work.

Run with: pytest tests/test_smoke.py -v
"""

import pytest
import torch
from pathlib import Path


class TestImports:
    """Test that all modules can be imported."""

    def test_import_model_modules(self):
        """Test importing model modules."""
        from src.model import normalization
        from src.model import embeddings
        from src.model import attention
        from src.model import feedforward
        from src.model import transformer_block
        from src.model import llm
        assert True

    def test_import_lora_modules(self):
        """Test importing LoRA modules."""
        from src.lora import lora_layer
        from src.lora import lora_model
        assert True

    def test_import_continual_modules(self):
        """Test importing continual learning modules."""
        from src.continual import experience_replay
        from src.continual import ewc
        from src.continual import continual_trainer
        assert True

    def test_import_evaluation_modules(self):
        """Test importing evaluation modules."""
        from src.evaluation import metrics
        from src.evaluation import evaluator
        assert True

    def test_import_tokenization_modules(self):
        """Test importing tokenization modules."""
        from src.tokenization import bpe_tokenizer
        from src.tokenization import vocab_trainer
        assert True

    def test_import_data_modules(self):
        """Test importing data modules."""
        from src.data import data_pipeline
        assert True


class TestBasicFunctionality:
    """Test basic functionality of core components."""

    def test_rmsnorm_creation(self):
        """Test creating RMSNorm layer."""
        from src.model.normalization import RMSNorm

        norm = RMSNorm(dim=512)
        assert norm.weight.shape == (512,)

    def test_rmsnorm_forward(self):
        """Test RMSNorm forward pass."""
        from src.model.normalization import RMSNorm

        norm = RMSNorm(dim=512)
        x = torch.randn(2, 10, 512)
        output = norm(x)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_rope_creation(self):
        """Test creating RoPE."""
        from src.model.embeddings import RotaryPositionEmbedding

        rope = RotaryPositionEmbedding(dim=64, max_seq_len=100)
        assert rope.dim == 64
        assert rope.max_seq_len == 100

    def test_rope_forward(self):
        """Test RoPE forward pass."""
        from src.model.embeddings import RotaryPositionEmbedding

        rope = RotaryPositionEmbedding(dim=64, max_seq_len=100)
        cos, sin = rope(seq_len=50)
        # RoPE returns full dim, not dim//2
        assert cos.shape == (50, 64)
        assert sin.shape == (50, 64)
        assert not torch.isnan(cos).any()
        assert not torch.isnan(sin).any()

    def test_lora_layer_creation(self):
        """Test creating LoRA layer."""
        from src.lora.lora_layer import LoRALayer

        lora = LoRALayer(in_features=512, out_features=512, r=8)
        assert lora.r == 8
        assert lora.lora_A.shape == (8, 512)
        assert lora.lora_B.shape == (512, 8)

    def test_lora_layer_forward(self):
        """Test LoRA layer forward pass."""
        from src.lora.lora_layer import LoRALayer

        lora = LoRALayer(in_features=512, out_features=512, r=8)
        x = torch.randn(2, 10, 512)
        output = lora(x)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_experience_replay_buffer(self):
        """Test experience replay buffer."""
        from src.continual.experience_replay import ExperienceReplayBuffer, Experience

        buffer = ExperienceReplayBuffer(max_size=100)

        # Add some experiences
        for i in range(10):
            exp = Experience(
                input_ids=torch.randint(0, 1000, (10,)),
                labels=torch.randint(0, 1000, (10,)),
                importance=1.0,
                domain="test"
            )
            buffer.add(exp)

        assert buffer.size == 10
        assert len(buffer.sample(5)) == 5

    def test_evaluation_metrics(self):
        """Test evaluation metrics."""
        from src.evaluation.metrics import compute_forgetting_rate

        previous = {'task_A': 0.9, 'task_B': 0.85}
        current = {'task_A': 0.87, 'task_B': 0.82}

        metrics = compute_forgetting_rate(current, previous)
        assert metrics.forgetting_rate >= 0
        assert hasattr(metrics, 'backward_transfer')
        assert hasattr(metrics, 'forward_transfer')

    def test_data_pipeline(self):
        """Test data pipeline document class."""
        from src.data.data_pipeline import Document

        doc = Document(
            content="Test content",
            source="test.txt",
            domain="test"
        )
        assert doc.content == "Test content"
        assert doc.source == "test.txt"
        assert len(doc) == len("Test content")


class TestModelCreation:
    """Test creating full models."""

    def test_create_small_llm(self):
        """Test creating a small LLM."""
        from src.model.llm import ContinualLLM, ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_query_heads=4,
            num_kv_heads=2,
            d_ff=512,
            max_seq_len=128
        )

        model = ContinualLLM(config)
        assert model.config.d_model == 128
        assert len(model.layers) == 2

    def test_llm_forward_pass(self):
        """Test LLM forward pass."""
        from src.model.llm import ContinualLLM, ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_query_heads=4,
            num_kv_heads=2,
            d_ff=512,
            max_seq_len=128
        )

        model = ContinualLLM(config)
        input_ids = torch.randint(0, 1000, (2, 10))

        logits, cache, _ = model(input_ids)

        assert logits.shape == (2, 10, 1000)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()


class TestSystemHealth:
    """Test overall system health."""

    def test_torch_available(self):
        """Test that PyTorch is available."""
        import torch
        assert torch.__version__ >= "2.0.0"

    def test_mps_or_cuda_available(self):
        """Test that GPU acceleration is available."""
        import torch
        has_acceleration = (
            torch.cuda.is_available() or
            torch.backends.mps.is_available()
        )
        # Just log, don't fail if no GPU
        if has_acceleration:
            if torch.cuda.is_available():
                print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
            if torch.backends.mps.is_available():
                print("\n✓ MPS (Apple Silicon) available")
        else:
            print("\n⚠ No GPU acceleration available (CPU only)")

    def test_project_structure(self):
        """Test that key directories exist."""
        key_dirs = [
            "src/model",
            "src/lora",
            "src/continual",
            "src/tokenization",
            "src/evaluation",
            "src/data",
            "scripts",
            "configs",
            "tests"
        ]

        for dir_path in key_dirs:
            assert Path(dir_path).exists(), f"Missing directory: {dir_path}"

    def test_config_files_exist(self):
        """Test that config files exist."""
        configs = ["configs/small.yaml", "configs/medium.yaml", "configs/large.yaml"]

        for config_path in configs:
            assert Path(config_path).exists(), f"Missing config: {config_path}"

    def test_scripts_exist(self):
        """Test that training scripts exist."""
        scripts = [
            "scripts/train_tokenizer.py",
            "scripts/pretrain.py",
            "scripts/continual_learn.py",
            "scripts/generate.py"
        ]

        for script_path in scripts:
            assert Path(script_path).exists(), f"Missing script: {script_path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
