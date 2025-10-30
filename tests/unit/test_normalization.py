"""
Unit tests for normalization layers (RMSNorm, LayerNorm).
"""

import pytest
import torch
import torch.nn as nn
from src.model.normalization import RMSNorm, LayerNorm, create_norm_layer


class TestRMSNorm:
    """Test RMSNorm implementation."""

    def test_initialization(self):
        """Test RMSNorm initialization."""
        norm = RMSNorm(dim=512, eps=1e-6)
        assert norm.eps == 1e-6
        assert norm.weight.shape == (512,)
        assert torch.allclose(norm.weight, torch.ones(512))

    def test_forward_shape(self):
        """Test that output shape matches input shape."""
        norm = RMSNorm(dim=512)
        x = torch.randn(4, 32, 512)  # [batch, seq_len, d_model]
        output = norm(x)
        assert output.shape == x.shape

    def test_normalization(self):
        """Test that normalization works correctly."""
        norm = RMSNorm(dim=512, eps=1e-8)
        x = torch.randn(2, 10, 512)
        output = norm(x)

        # Check that RMS is approximately 1 (scaled by weight)
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        # Weight is initialized to 1, so RMS should be close to 1
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_scale_invariance(self):
        """Test that RMSNorm is scale-invariant along feature dim."""
        norm = RMSNorm(dim=512)
        x = torch.randn(2, 10, 512)

        # Scale input by constant
        scaled_x = x * 5.0

        # Outputs should have same RMS (up to the learned weight)
        output1 = norm(x)
        output2 = norm(scaled_x)

        # The normalized outputs should be similar in magnitude
        rms1 = torch.sqrt(torch.mean(output1 ** 2, dim=-1))
        rms2 = torch.sqrt(torch.mean(output2 ** 2, dim=-1))

        assert torch.allclose(rms1, rms2, atol=0.01)

    def test_trainable_weight(self):
        """Test that weight is trainable."""
        norm = RMSNorm(dim=512)
        assert norm.weight.requires_grad

    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        norm = RMSNorm(dim=512)

        # Test float32
        x_fp32 = torch.randn(2, 10, 512, dtype=torch.float32)
        output_fp32 = norm(x_fp32)
        assert output_fp32.dtype == torch.float32

        # Test float16
        x_fp16 = torch.randn(2, 10, 512, dtype=torch.float16)
        output_fp16 = norm(x_fp16)
        assert output_fp16.dtype == torch.float16


class TestLayerNormWrapper:
    """Test our LayerNorm wrapper."""

    def test_initialization(self):
        """Test LayerNorm initialization."""
        norm = create_norm_layer(dim=512, norm_type="layernorm")
        assert isinstance(norm, LayerNorm)

    def test_forward_shape(self):
        """Test that output shape matches input shape."""
        norm = create_norm_layer(dim=512, norm_type="layernorm")
        x = torch.randn(4, 32, 512)
        output = norm(x)
        assert output.shape == x.shape


class TestNormFactory:
    """Test norm layer factory function."""

    def test_rmsnorm_creation(self):
        """Test creating RMSNorm via factory."""
        norm = create_norm_layer(dim=512, norm_type="rmsnorm", eps=1e-6)
        assert isinstance(norm, RMSNorm)
        assert norm.weight.shape == (512,)

    def test_layernorm_creation(self):
        """Test creating LayerNorm via factory."""
        norm = create_norm_layer(dim=512, norm_type="layernorm")
        assert isinstance(norm, LayerNorm)

    def test_invalid_norm_type(self):
        """Test that invalid norm type raises error."""
        with pytest.raises(ValueError):
            create_norm_layer(dim=512, norm_type="invalid_norm")


class TestNormalizationNumericalStability:
    """Test numerical stability of normalization."""

    def test_zero_input(self):
        """Test handling of zero input."""
        norm = RMSNorm(dim=512, eps=1e-8)
        x = torch.zeros(2, 10, 512)
        output = norm(x)

        # Should not produce NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_very_large_input(self):
        """Test handling of very large values."""
        norm = RMSNorm(dim=512)
        x = torch.randn(2, 10, 512) * 1e6
        output = norm(x)

        # Should not produce NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_very_small_input(self):
        """Test handling of very small values."""
        norm = RMSNorm(dim=512)
        x = torch.randn(2, 10, 512) * 1e-6
        output = norm(x)

        # Should not produce NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
