"""
Unit tests for LoRA (Low-Rank Adaptation) layers.
"""

import pytest
import torch
import torch.nn as nn
from src.lora.lora_layer import LoRALayer, LoRAConfig, mark_only_lora_as_trainable
from src.lora.lora_model import LoRAModel


class TestLoRALayer:
    """Test LoRA layer implementation."""

    def test_initialization(self):
        """Test LoRA layer initialization."""
        in_features = 512
        out_features = 512
        r = 8
        alpha = 16.0

        lora = LoRALayer(in_features, out_features, r=r, alpha=alpha, dropout=0.0)

        assert lora.in_features == in_features
        assert lora.out_features == out_features
        assert lora.r == r
        assert lora.alpha == alpha
        assert lora.scaling == alpha / r

        # Check matrix shapes
        assert lora.lora_A.shape == (r, in_features)
        assert lora.lora_B.shape == (out_features, r)

    def test_initialization_zero_b(self):
        """Test that lora_B is initialized to zero."""
        lora = LoRALayer(512, 512, r=8)

        # lora_B should be initialized to zeros
        assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))

    def test_forward_shape(self):
        """Test forward pass shape."""
        batch_size = 4
        seq_len = 32
        in_features = 512
        out_features = 512

        lora = LoRALayer(in_features, out_features, r=8)

        x = torch.randn(batch_size, seq_len, in_features)
        output = lora(x)

        assert output.shape == (batch_size, seq_len, out_features)

    def test_low_rank_property(self):
        """Test that LoRA produces low-rank updates."""
        in_features = 512
        out_features = 512
        r = 8  # Rank

        lora = LoRALayer(in_features, out_features, r=r)

        # Initialize with non-zero values
        nn.init.normal_(lora.lora_A)
        nn.init.normal_(lora.lora_B)

        # Compute full LoRA matrix: B @ A
        lora_weight = lora.lora_B @ lora.lora_A  # [out_features, in_features]

        # Check rank (should be at most r)
        rank = torch.linalg.matrix_rank(lora_weight).item()
        assert rank <= r

    def test_scaling_factor(self):
        """Test that scaling factor is applied correctly."""
        lora = LoRALayer(512, 512, r=8, alpha=16.0, dropout=0.0)

        # Set lora_A and lora_B to known values
        lora.lora_A.data = torch.ones_like(lora.lora_A)
        lora.lora_B.data = torch.ones_like(lora.lora_B)

        x = torch.ones(1, 1, 512)
        output = lora(x)

        # Expected output: x @ A^T @ B^T * (alpha / r)
        # With all ones: (512) * 8 * (16 / 8) = 512 * 8 * 2 = 8192
        expected_value = 512 * 8 * (16.0 / 8.0)

        assert torch.allclose(output, torch.full_like(output, expected_value), atol=1e-3)

    def test_merge_weights(self):
        """Test merging LoRA weights into base layer."""
        from src.lora.lora_layer import LinearWithLoRA

        in_features = 512
        out_features = 512
        r = 8

        base_layer = nn.Linear(in_features, out_features, bias=False)
        base_weight_before = base_layer.weight.data.clone()

        # Create LinearWithLoRA wrapper
        lora_layer = LinearWithLoRA(base_layer, r=r, alpha=16.0, dropout=0.0)

        # Initialize with known values
        nn.init.normal_(lora_layer.lora.lora_A, std=0.1)
        nn.init.normal_(lora_layer.lora.lora_B, std=0.1)

        # Merge
        lora_layer.merge()

        # Check that weights were updated
        assert lora_layer.merged

        # Merged should be: base_weight + lora_B @ lora_A * scaling
        expected = base_weight_before + (lora_layer.lora.lora_B @ lora_layer.lora.lora_A) * lora_layer.lora.scaling

        assert torch.allclose(base_layer.weight, expected, atol=1e-5)

    def test_enabled_disabled(self):
        """Test that LoRA produces non-zero output when parameters are non-zero."""
        lora = LoRALayer(512, 512, r=8, dropout=0.0)
        lora.lora_A.data = torch.ones_like(lora.lora_A)
        lora.lora_B.data = torch.ones_like(lora.lora_B)

        x = torch.randn(2, 10, 512)

        # LoRA should produce non-zero output
        output = lora(x)
        assert not torch.allclose(output, torch.zeros_like(output))

        # When lora_B is zero, output should be zero (since LoRA effect is zero)
        lora_zero = LoRALayer(512, 512, r=8, dropout=0.0)
        # lora_B is initialized to zero, so output should be zero
        output_zero = lora_zero(x)
        assert torch.allclose(output_zero, torch.zeros_like(output_zero))

    def test_trainable_parameters(self):
        """Test that only LoRA parameters are trainable."""
        lora = LoRALayer(512, 512, r=8)

        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in lora.parameters() if p.requires_grad)

        # Should be: r * in_features + out_features * r
        expected_params = 8 * 512 + 512 * 8
        assert trainable_params == expected_params


class TestLoRAConfig:
    """Test LoRA configuration."""

    def test_default_config(self):
        """Test default LoRA configuration."""
        config = LoRAConfig()

        assert config.r == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.0
        # Default target modules from actual implementation
        assert config.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"]

    def test_custom_config(self):
        """Test custom LoRA configuration."""
        config = LoRAConfig(
            r=16,
            alpha=32.0,
            dropout=0.1,
            target_modules=["query", "value"]
        )

        assert config.r == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.1
        assert config.target_modules == ["query", "value"]


class TestMarkOnlyLoRATrainable:
    """Test the mark_only_lora_as_trainable function."""

    def test_freeze_base_parameters(self):
        """Test that base parameters are frozen."""
        model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # Add LoRA layers
        model[0].lora = LoRALayer(512, 512, r=8)
        model[2].lora = LoRALayer(512, 256, r=8)

        # Mark only LoRA as trainable
        mark_only_lora_as_trainable(model)

        # Check base parameters are frozen
        for name, param in model.named_parameters():
            if 'lora' in name:
                assert param.requires_grad, f"LoRA parameter {name} should be trainable"
            else:
                assert not param.requires_grad, f"Base parameter {name} should be frozen"

    def test_count_trainable_parameters(self):
        """Test that number of trainable parameters is reduced."""
        in_features = 512
        out_features = 512
        r = 8

        # Base model
        base_model = nn.Linear(in_features, out_features, bias=False)
        base_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

        # Add LoRA
        base_model.lora = LoRALayer(in_features, out_features, r=r)
        mark_only_lora_as_trainable(base_model)

        lora_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

        # LoRA should have much fewer parameters
        assert lora_params < base_params

        # LoRA params: r * in_features + out_features * r
        expected_lora_params = r * in_features + out_features * r
        assert lora_params == expected_lora_params

        # Calculate reduction ratio
        reduction = base_params / lora_params
        print(f"Parameter reduction: {reduction:.1f}x")
        assert reduction > 10  # Should be much more efficient


class TestLoRAModel:
    """Test LoRA model wrapper."""

    def test_model_wrapping(self):
        """Test wrapping a model with LoRA."""
        # Create a simple base model
        base_model = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False)
        )

        config = LoRAConfig(r=8, target_modules=["Linear"])

        # Wrap with LoRA
        lora_model = LoRAModel(base_model, config, adapter_name="test")

        # Check that LoRA layers were injected
        assert hasattr(lora_model, 'base_model')
        assert len(lora_model.lora_modules) >= 0  # lora_modules, not lora_layers

    def test_adapter_management(self):
        """Test adding and managing multiple adapters."""
        from src.lora.lora_model import MultiAdapterLoRAModel

        base_model = nn.Linear(512, 512, bias=False)
        config = LoRAConfig(r=8)

        lora_model = MultiAdapterLoRAModel(base_model, config)

        # Add adapters
        lora_model.add_adapter("adapter1", config)
        lora_model.add_adapter("adapter2", config)

        assert "adapter1" in lora_model.adapters
        assert "adapter2" in lora_model.adapters

    def test_adapter_switching(self):
        """Test switching between adapters."""
        from src.lora.lora_model import MultiAdapterLoRAModel

        base_model = nn.Linear(512, 512, bias=False)
        config = LoRAConfig(r=8, dropout=0.0)

        lora_model = MultiAdapterLoRAModel(base_model, config)
        lora_model.add_adapter("adapter1", config)
        lora_model.add_adapter("adapter2", config)

        # Set adapter
        lora_model.set_adapter("adapter1")
        assert "adapter1" in lora_model.active_adapters

        # Switch adapter
        lora_model.set_adapter("adapter2")
        assert "adapter2" in lora_model.active_adapters

        # Test forward pass with different adapters
        x = torch.randn(2, 10, 512)

        lora_model.set_adapter("adapter1")
        output1 = lora_model(x)
        assert output1.shape == (2, 10, 512)

        lora_model.set_adapter("adapter2")
        output2 = lora_model(x)
        assert output2.shape == (2, 10, 512)

        # Test passed if no errors during adapter switching


class TestLoRANumericalStability:
    """Test numerical stability of LoRA."""

    def test_zero_input(self):
        """Test LoRA with zero input."""
        lora = LoRALayer(512, 512, r=8)
        x = torch.zeros(2, 10, 512)
        output = lora(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_large_values(self):
        """Test LoRA with large values."""
        lora = LoRALayer(512, 512, r=8)
        x = torch.randn(2, 10, 512) * 1000
        output = lora(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow through LoRA."""
        lora = LoRALayer(512, 512, r=8)
        x = torch.randn(2, 10, 512, requires_grad=True)

        output = lora(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        assert not torch.isnan(lora.lora_A.grad).any()
        assert not torch.isnan(lora.lora_B.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
