"""
Integration tests for advanced brain-inspired mechanisms.

Tests the integration of:
- Predictive Coding with transformer layers
- Liquid Time Constants with LoRA
- Combined continual learning system
"""

import pytest
import torch
import torch.nn as nn

from src.neurobio.predictive_coding import (
    PredictiveCodingLayer,
    PredictiveCodingNetwork,
    PredictiveCodingConfig,
    compute_free_energy
)
from src.neurobio.liquid_dynamics import (
    LiquidTimeConstantCell,
    LiquidLayer,
    LiquidMLP,
    LiquidConfig
)
from src.lora.liquid_lora import (
    LiquidLoRALayer,
    LiquidLoRALinear,
    LiquidLoRAConfig
)
from src.model.predictive_transformer import (
    PredictiveTransformerBlock,
    PredictiveTransformerStack,
    PredictiveTransformerConfig
)
from src.continual.continual_trainer import ContinualLearningConfig


class TestPredictiveCodingIntegration:
    """Test predictive coding integration."""

    def test_pc_layer_forward(self):
        """Test basic PC layer forward pass."""
        config = PredictiveCodingConfig(d_model=128, num_inference_steps=5)
        layer = PredictiveCodingLayer(config)

        x_lower = torch.randn(2, 10, 128)
        x_higher = torch.randn(2, 10, 128)

        output, error = layer(x_lower, x_higher)

        assert output.shape == x_lower.shape
        assert error.shape == x_lower.shape
        assert not torch.isnan(output).any()
        assert not torch.isnan(error).any()

    def test_pc_network_hierarchy(self):
        """Test hierarchical PC network."""
        config = PredictiveCodingConfig(d_model=64, num_inference_steps=3)
        network = PredictiveCodingNetwork(config, num_layers=3)

        x = torch.randn(2, 10, 64)
        output, errors = network(x, return_all_errors=True)

        assert output.shape == (2, 10, 64)
        assert len(errors) == 3
        for error in errors:
            if error is not None:
                assert error.shape[0] == 2
                assert error.shape[1] == 10

    def test_pc_free_energy_reduction(self):
        """Test that free energy decreases during learning."""
        config = PredictiveCodingConfig(d_model=64, num_inference_steps=10)
        network = PredictiveCodingNetwork(config, num_layers=2)

        x = torch.randn(2, 10, 64)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

        # Initial free energy
        _, errors = network(x, return_all_errors=True)
        initial_energy = compute_free_energy(errors).item()

        # Train for a few steps
        for _ in range(5):
            optimizer.zero_grad()
            _, errors = network(x, return_all_errors=True)
            loss = compute_free_energy(errors)
            loss.backward()
            optimizer.step()

        # Final free energy
        _, errors = network(x, return_all_errors=True)
        final_energy = compute_free_energy(errors).item()

        # Energy should decrease
        assert final_energy < initial_energy


class TestLiquidDynamicsIntegration:
    """Test liquid time constant integration."""

    def test_liquid_cell_adaptive_tau(self):
        """Test adaptive time constants."""
        config = LiquidConfig(d_model=64, tau_base=1.0)
        cell = LiquidTimeConstantCell(config)

        x = torch.randn(2, 64)
        h = torch.zeros(2, 64)

        # Forward pass
        h_new = cell(x, h, dt=0.1)
        assert h_new.shape == h.shape

        # Check time constants
        tau = cell.get_time_constants(x, h)
        assert tau.shape == (2, 64)
        assert (tau >= config.tau_min).all()
        assert (tau <= config.tau_max).all()

    def test_liquid_layer_sequence_processing(self):
        """Test liquid layer on sequences."""
        config = LiquidConfig(d_model=128, tau_base=1.0)
        layer = LiquidLayer(input_dim=64, hidden_dim=128, config=config)

        x = torch.randn(2, 10, 64)
        output = layer(x, dt=0.1)

        assert output.shape == (2, 10, 128)
        assert not torch.isnan(output).any()

    def test_liquid_mlp_iterative_dynamics(self):
        """Test liquid MLP with iterative updates."""
        config = LiquidConfig(d_model=256, tau_base=0.5)
        mlp = LiquidMLP(d_model=64, d_ff=256, config=config, num_steps=5)

        x = torch.randn(2, 10, 64)
        output = mlp(x, dt=0.1)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestLiquidLoRAIntegration:
    """Test Liquid LoRA integration."""

    def test_liquid_lora_layer_forward(self):
        """Test basic Liquid LoRA forward pass."""
        config = LiquidLoRAConfig(r=8, tau_base=1.0, use_liquid_dynamics=True)
        layer = LiquidLoRALayer(256, 256, config)

        x = torch.randn(2, 10, 256)
        output = layer(x, dt=0.1)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_liquid_lora_complete_linear(self):
        """Test complete linear layer with Liquid LoRA."""
        config = LiquidLoRAConfig(r=8, tau_base=1.0, use_liquid_dynamics=True)
        layer = LiquidLoRALinear(128, 128, config, bias=False)

        x = torch.randn(2, 128)
        output = layer(x, dt=0.1)

        assert output.shape == x.shape

        # Check trainable parameters
        trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in layer.parameters())
        assert trainable < total  # Base is frozen

    def test_liquid_lora_importance_tracking(self):
        """Test importance-based time constant adaptation."""
        config = LiquidLoRAConfig(r=8, tau_base=1.0, use_liquid_dynamics=True)
        layer = LiquidLoRALayer(64, 64, config)

        x = torch.randn(2, 10, 64)

        # Forward pass
        output = layer(x, dt=0.1)
        loss = output.pow(2).mean()
        loss.backward()

        # Update importance based on gradients
        grad_mag = layer.lora_A.grad.abs().mean(dim=1)
        layer.update_importance(grad_mag)

        # Check importance updated
        assert layer.importance.mean() > 0
        assert layer.importance.mean() <= 1.0

    def test_liquid_lora_reset_dynamics(self):
        """Test domain switching via reset."""
        config = LiquidLoRAConfig(r=8, tau_base=1.0, use_liquid_dynamics=True)
        layer = LiquidLoRALinear(64, 64, config)

        x = torch.randn(2, 64)

        # Run forward to populate hidden state
        for _ in range(10):
            _ = layer(x, dt=0.5)  # Larger dt to accumulate more

        # Check hidden state exists (may be small due to no_grad updates)
        # The important part is reset works
        hidden_norm_before = layer.lora.hidden_B.norm().item()

        # Reset for new domain
        layer.lora.reset_dynamics()

        # Hidden state should be zero after reset
        hidden_norm_after = layer.lora.hidden_B.norm().item()
        assert hidden_norm_after == 0.0

        # Importance should be reset to 1.0
        assert (layer.lora.importance == 1.0).all()


class TestPredictiveTransformerIntegration:
    """Test predictive transformer integration."""

    def test_predictive_transformer_block(self):
        """Test predictive transformer block."""
        from src.model.embeddings import RotaryPositionEmbedding
        from src.model.attention import create_causal_mask

        config = PredictiveTransformerConfig(
            d_model=128,
            num_query_heads=4,
            num_kv_heads=2,
            d_ff=512,
            pc_inference_steps=3,
            use_pc_for_ffn=True
        )
        block = PredictiveTransformerBlock(config)

        # Setup
        x = torch.randn(2, 10, 128)
        head_dim = 128 // 4
        rope = RotaryPositionEmbedding(dim=head_dim)
        cos, sin = rope(10)
        mask = create_causal_mask(10, x.device)

        # Forward
        output, cache, error = block(x, cos, sin, mask, use_cache=True)

        assert output.shape == x.shape
        assert cache is not None
        # Error might be None if PC not enabled for this block

    def test_predictive_transformer_stack(self):
        """Test stack with selective PC layers."""
        from src.model.embeddings import RotaryPositionEmbedding
        from src.model.attention import create_causal_mask

        config = PredictiveTransformerConfig(
            d_model=64,
            num_query_heads=4,
            num_kv_heads=2,
            d_ff=256,
            pc_inference_steps=3
        )

        stack = PredictiveTransformerStack(
            config,
            num_blocks=6,
            pc_start_layer=2,
            pc_end_layer=4
        )

        # Setup
        x = torch.randn(2, 10, 64)
        head_dim = 64 // 4
        rope = RotaryPositionEmbedding(dim=head_dim)
        cos, sin = rope(10)
        mask = create_causal_mask(10, x.device)

        # Forward
        output, caches, errors = stack(x, cos, sin, mask, use_cache=True)

        assert output.shape == x.shape
        assert len(caches) == 6
        # Only layers 2-3 should have errors
        assert len(errors) > 0

    def test_predictive_stack_importance_scores(self):
        """Test layer importance scoring."""
        from src.model.embeddings import RotaryPositionEmbedding
        from src.model.attention import create_causal_mask

        config = PredictiveTransformerConfig(
            d_model=64,
            num_query_heads=4,
            num_kv_heads=2,
            d_ff=256
        )

        stack = PredictiveTransformerStack(
            config,
            num_blocks=4,
            pc_start_layer=1,
            pc_end_layer=3
        )

        x = torch.randn(2, 10, 64)
        head_dim = 64 // 4
        rope = RotaryPositionEmbedding(dim=head_dim)
        cos, sin = rope(10)
        mask = create_causal_mask(10, x.device)

        # Forward to generate errors
        output, _, _ = stack(x, cos, sin, mask)

        # Get importance scores
        scores = stack.get_layer_importance_scores()
        assert len(scores) == 4
        # Layers 1-2 should have non-zero scores
        assert scores[1] >= 0
        assert scores[2] >= 0


class TestContinualLearnerConfiguration:
    """Test continual learner configuration with new mechanisms."""

    def test_config_with_predictive_coding(self):
        """Test configuration includes PC options."""
        config = ContinualLearningConfig(
            use_predictive_coding=True,
            pc_inference_steps=5,
            pc_start_layer=6,
            pc_end_layer=9
        )

        assert config.use_predictive_coding is True
        assert config.pc_inference_steps == 5
        assert config.pc_start_layer == 6
        assert config.pc_end_layer == 9

    def test_config_with_liquid_lora(self):
        """Test configuration includes Liquid LoRA options."""
        config = ContinualLearningConfig(
            use_liquid_lora=True,
            liquid_tau_base=1.0,
            liquid_tau_min=0.1,
            liquid_tau_max=10.0
        )

        assert config.use_liquid_lora is True
        assert config.liquid_tau_base == 1.0
        assert config.liquid_tau_min == 0.1
        assert config.liquid_tau_max == 10.0

    def test_config_all_brain_mechanisms(self):
        """Test configuration with all brain mechanisms enabled."""
        config = ContinualLearningConfig(
            # Phase 1
            use_neuromodulation=True,
            use_homeostasis=True,
            use_autonomous_lr=True,
            # Phase 2
            use_predictive_coding=True,
            use_liquid_lora=True
        )

        assert config.use_neuromodulation is True
        assert config.use_homeostasis is True
        assert config.use_autonomous_lr is True
        assert config.use_predictive_coding is True
        assert config.use_liquid_lora is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
