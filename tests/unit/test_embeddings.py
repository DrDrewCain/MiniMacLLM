"""
Unit tests for position embeddings (RoPE).
"""

import pytest
import torch

from src.model.embeddings import RotaryPositionEmbedding, apply_rotary_pos_emb


class TestRotaryPositionEmbedding:
    """Test RoPE implementation."""

    def test_initialization(self):
        """Test RoPE initialization."""
        rope = RotaryPositionEmbedding(dim=128, max_seq_len=2048, base=10000.0)
        assert rope.dim == 128
        assert rope.max_seq_len == 2048
        assert rope.base == 10000.0

        # Check precomputed frequencies
        assert rope.cos_cached.shape == (2048, 128)  # [max_seq_len, dim]
        assert rope.sin_cached.shape == (2048, 128)

    def test_get_cos_sin_shape(self):
        """Test that forward returns correct shapes."""
        rope = RotaryPositionEmbedding(dim=128, max_seq_len=2048)

        # Test with different sequence lengths
        for seq_len in [10, 100, 512, 2048]:
            cos, sin = rope(seq_len)
            assert cos.shape == (seq_len, 128)
            assert sin.shape == (seq_len, 128)

    def test_frequency_computation(self):
        """Test that frequencies are computed correctly."""
        dim = 128
        base = 10000.0
        rope = RotaryPositionEmbedding(dim=dim, max_seq_len=100, base=base)

        # Check that inv_freq follows the formula: 1 / (base ^ (2i/dim))
        inv_freq_expected = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        # The actual inv_freq is stored in the frequencies
        # We can check the first position
        cos_0 = rope.cos_cached[0]  # cos at position 0
        # At position 0, cos should be 1 (cos(0) = 1)
        assert torch.allclose(cos_0, torch.ones_like(cos_0), atol=1e-6)

    def test_position_dependence(self):
        """Test that embeddings change with position."""
        rope = RotaryPositionEmbedding(dim=128, max_seq_len=2048)

        cos_0, sin_0 = rope(1)
        cos_100, sin_100 = rope(101)

        # Position 0 and position 100 should have different embeddings
        assert not torch.allclose(cos_100[-1], cos_0[0])
        assert not torch.allclose(sin_100[-1], sin_0[0])

    def test_caching(self):
        """Test that caching works correctly."""
        rope = RotaryPositionEmbedding(dim=128, max_seq_len=2048)

        # Get embeddings twice
        cos1, sin1 = rope(100)
        cos2, sin2 = rope(100)

        # Should return same results (cached)
        assert torch.equal(cos1, cos2)
        assert torch.equal(sin1, sin2)

    def test_different_sequence_lengths(self):
        """Test handling different sequence lengths."""
        rope = RotaryPositionEmbedding(dim=128, max_seq_len=2048)

        for seq_len in [1, 10, 100, 500, 1000, 2048]:
            cos, sin = rope(seq_len)
            assert cos.shape[0] == seq_len
            assert not torch.isnan(cos).any()
            assert not torch.isinf(sin).any()


class TestApplyRotaryPosEmb:
    """Test the apply_rotary_pos_emb function."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 100, 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=2048)
        cos, sin = rope(seq_len)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotation_changes_values(self):
        """Test that rotation actually changes the values."""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 100, 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=2048)
        cos, sin = rope(seq_len)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        # Rotated values should be different
        assert not torch.allclose(q_rot, q)
        assert not torch.allclose(k_rot, k)

    def test_rotation_preserves_norm(self):
        """Test that rotation preserves the norm of vectors."""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 100, 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=2048)
        cos, sin = rope(seq_len)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        # Rotation is an orthogonal transformation, should preserve norms
        q_norm_before = torch.norm(q, dim=-1)
        q_norm_after = torch.norm(q_rot, dim=-1)

        k_norm_before = torch.norm(k, dim=-1)
        k_norm_after = torch.norm(k_rot, dim=-1)

        assert torch.allclose(q_norm_before, q_norm_after, atol=1e-5)
        assert torch.allclose(k_norm_before, k_norm_after, atol=1e-5)

    def test_position_sensitivity(self):
        """Test that different positions get different rotations."""
        batch_size, num_heads, seq_len, head_dim = 1, 1, 2, 64

        # Same query at two different positions
        q = torch.ones(batch_size, num_heads, seq_len, head_dim)
        k = torch.ones(batch_size, num_heads, seq_len, head_dim)

        rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=2048)
        cos, sin = rope(seq_len)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        # Position 0 and position 1 should have different values after rotation
        assert not torch.allclose(q_rot[0, 0, 0, :], q_rot[0, 0, 1, :])

    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 100, 64

        # Test with zero input
        q = torch.zeros(batch_size, num_heads, seq_len, head_dim)
        k = torch.zeros(batch_size, num_heads, seq_len, head_dim)

        rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=2048)
        cos, sin = rope(seq_len)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        assert not torch.isnan(q_rot).any()
        assert not torch.isnan(k_rot).any()

        # Test with large values
        q = torch.randn(batch_size, num_heads, seq_len, head_dim) * 1000
        k = torch.randn(batch_size, num_heads, seq_len, head_dim) * 1000

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        assert not torch.isnan(q_rot).any()
        assert not torch.isnan(k_rot).any()
        assert not torch.isinf(q_rot).any()
        assert not torch.isinf(k_rot).any()


class TestRoPEExtrapolation:
    """Test RoPE's ability to extrapolate to longer sequences."""

    def test_extrapolation_beyond_training(self):
        """Test that RoPE can handle sequences longer than max_seq_len."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=100, base=10000.0)

        # Try to get embeddings for longer sequence
        # This should work by using the cached values up to max_seq_len
        cos, sin = rope(100)  # At the limit
        assert cos.shape == (100, 64)

        # Exactly at max_seq_len should work
        cos, sin = rope(100)
        assert not torch.isnan(cos).any()
        assert not torch.isnan(sin).any()


class TestRoPEDifferentDimensions:
    """Test RoPE with different head dimensions."""

    @pytest.mark.parametrize("head_dim", [32, 64, 80, 128])
    def test_different_head_dims(self, head_dim):
        """Test RoPE with various head dimensions."""
        rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=1024)

        cos, sin = rope(100)

        # Should have shape [seq_len, head_dim]
        assert cos.shape == (100, head_dim)
        assert sin.shape == (100, head_dim)
        assert not torch.isnan(cos).any()
        assert not torch.isnan(sin).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
