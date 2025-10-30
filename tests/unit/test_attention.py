"""
Unit tests for attention mechanisms (GQA).
"""

import pytest
import torch
from src.model.attention import GroupedQueryAttention


class TestGroupedQueryAttention:
    """Test GQA implementation."""

    def test_initialization(self):
        """Test GQA initialization."""
        d_model = 512
        num_query_heads = 8
        num_kv_heads = 2

        attn = GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dropout=0.0
        )

        assert attn.d_model == d_model
        assert attn.num_query_heads == num_query_heads
        assert attn.num_kv_heads == num_kv_heads
        assert attn.num_queries_per_kv == num_query_heads // num_kv_heads
        assert attn.head_dim == d_model // num_query_heads

    def test_forward_shape_without_cache(self):
        """Test forward pass shape without KV cache."""
        d_model = 512
        num_query_heads = 8
        num_kv_heads = 2
        batch_size = 2
        seq_len = 32

        attn = GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads
        )

        x = torch.randn(batch_size, seq_len, d_model)
        head_dim = d_model // num_query_heads
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)

        output, cache = attn(x, cos, sin, mask=None, cache=None, use_cache=False)

        assert output.shape == (batch_size, seq_len, d_model)
        assert cache is None

    def test_forward_with_cache(self):
        """Test forward pass with KV caching."""
        d_model = 512
        num_query_heads = 8
        num_kv_heads = 2
        batch_size = 2
        seq_len = 32

        attn = GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads
        )

        x = torch.randn(batch_size, seq_len, d_model)
        head_dim = d_model // num_query_heads
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)

        output, cache = attn(x, cos, sin, mask=None, cache=None, use_cache=True)

        assert output.shape == (batch_size, seq_len, d_model)
        assert cache is not None
        assert hasattr(cache, 'key') and hasattr(cache, 'value')

        # Check cache shapes
        head_dim = d_model // num_query_heads
        assert cache.key.shape == (batch_size, num_kv_heads, seq_len, head_dim)
        assert cache.value.shape == (batch_size, num_kv_heads, seq_len, head_dim)

    def test_incremental_generation_with_cache(self):
        """Test incremental generation using KV cache."""
        d_model = 512
        num_query_heads = 8
        num_kv_heads = 2
        batch_size = 1

        attn = GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads
        )

        # First forward pass with full sequence
        seq_len_1 = 10
        x1 = torch.randn(batch_size, seq_len_1, d_model)
        head_dim = d_model // num_query_heads
        cos1 = torch.randn(seq_len_1, head_dim)
        sin1 = torch.randn(seq_len_1, head_dim)

        output1, cache = attn(x1, cos1, sin1, mask=None, cache=None, use_cache=True)

        # Second forward pass with one new token
        seq_len_2 = 1
        x2 = torch.randn(batch_size, seq_len_2, d_model)
        # When using cache, cos/sin should match the new token length, not total length
        cos2 = torch.randn(seq_len_2, head_dim)
        sin2 = torch.randn(seq_len_2, head_dim)

        output2, cache = attn(x2, cos2, sin2, mask=None, cache=cache, use_cache=True)

        # Output should have shape for the new token
        assert output2.shape == (batch_size, seq_len_2, d_model)

        # Cache should now contain seq_len_1 + seq_len_2 tokens
        assert cache.key.shape == (batch_size, num_kv_heads, seq_len_1 + seq_len_2, head_dim)

    def test_attention_mask(self):
        """Test attention masking."""
        d_model = 512
        num_query_heads = 8
        num_kv_heads = 2
        batch_size = 2
        seq_len = 32

        attn = GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads
        )

        x = torch.randn(batch_size, seq_len, d_model)
        head_dim = d_model // num_query_heads
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        output, _ = attn(x, cos, sin, mask=mask, cache=None, use_cache=False)

        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()

    def test_kv_head_sharing(self):
        """Test that KV heads are shared across query heads."""
        d_model = 512
        num_query_heads = 8
        num_kv_heads = 2  # Each KV head shared by 4 query heads

        attn = GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads
        )

        assert attn.num_queries_per_kv == 4

        # Check that K and V projections have correct output dimension
        head_dim = d_model // num_query_heads
        assert attn.k_proj.out_features == num_kv_heads * head_dim
        assert attn.v_proj.out_features == num_kv_heads * head_dim
        assert attn.q_proj.out_features == num_query_heads * head_dim

    def test_different_gqa_ratios(self):
        """Test GQA with different query-to-kv ratios."""
        d_model = 768
        test_cases = [
            (12, 12),  # MHA (no grouping)
            (12, 3),   # 4:1 ratio
            (12, 2),   # 6:1 ratio
            (12, 1),   # MQA (maximum grouping)
        ]

        for num_query_heads, num_kv_heads in test_cases:
            attn = GroupedQueryAttention(
                d_model=d_model,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads
            )

            x = torch.randn(2, 16, d_model)
            head_dim = d_model // num_query_heads
            cos = torch.randn(16, head_dim)
            sin = torch.randn(16, head_dim)

            output, _ = attn(x, cos, sin)

            assert output.shape == (2, 16, d_model)
            assert not torch.isnan(output).any()

    def test_dropout_in_training(self):
        """Test that dropout is applied during training."""
        d_model = 512
        num_query_heads = 8
        num_kv_heads = 2

        attn = GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dropout=0.5  # High dropout for testing
        )

        attn.train()  # Set to training mode

        x = torch.randn(2, 32, d_model)
        head_dim = d_model // num_query_heads
        cos = torch.randn(32, head_dim)
        sin = torch.randn(32, head_dim)

        # Run multiple times, should get different results due to dropout
        output1, _ = attn(x, cos, sin)
        output2, _ = attn(x, cos, sin)

        # Outputs should be different due to dropout
        assert not torch.allclose(output1, output2)

    def test_no_dropout_in_eval(self):
        """Test that dropout is not applied during evaluation."""
        d_model = 512
        num_query_heads = 8
        num_kv_heads = 2

        attn = GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dropout=0.5
        )

        attn.eval()  # Set to eval mode

        x = torch.randn(2, 32, d_model)
        head_dim = d_model // num_query_heads
        cos = torch.randn(32, head_dim)
        sin = torch.randn(32, head_dim)

        # Run multiple times, should get same results (no dropout)
        output1, _ = attn(x, cos, sin)
        output2, _ = attn(x, cos, sin)

        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2)


class TestAttentionNumericalStability:
    """Test numerical stability of attention."""

    def test_zero_input(self):
        """Test handling of zero input."""
        attn = GroupedQueryAttention(d_model=512, num_query_heads=8, num_kv_heads=2)

        x = torch.zeros(2, 32, 512)
        head_dim = 512 // 8
        cos = torch.zeros(32, head_dim)
        sin = torch.zeros(32, head_dim)

        output, _ = attn(x, cos, sin)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_large_values(self):
        """Test handling of large values."""
        attn = GroupedQueryAttention(d_model=512, num_query_heads=8, num_kv_heads=2)

        x = torch.randn(2, 32, 512) * 100
        head_dim = 512 // 8
        cos = torch.randn(32, head_dim)
        sin = torch.randn(32, head_dim)

        output, _ = attn(x, cos, sin)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_long_sequence(self):
        """Test with long sequences."""
        attn = GroupedQueryAttention(d_model=512, num_query_heads=8, num_kv_heads=2)

        seq_len = 1024  # Long sequence
        x = torch.randn(1, seq_len, 512)
        head_dim = 512 // 8
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)

        output, _ = attn(x, cos, sin)

        assert output.shape == (1, seq_len, 512)
        assert not torch.isnan(output).any()


class TestAttentionMemoryEfficiency:
    """Test memory efficiency of GQA vs MHA."""

    def test_gqa_reduces_kv_cache_size(self):
        """Test that GQA reduces KV cache size compared to MHA."""
        d_model = 768
        num_query_heads = 12
        batch_size = 2
        seq_len = 100

        # MHA: num_kv_heads = num_query_heads
        mha = GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_query_heads
        )

        # GQA: num_kv_heads < num_query_heads
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=3  # 4x reduction
        )

        x = torch.randn(batch_size, seq_len, d_model)
        head_dim = d_model // num_query_heads
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)

        # Get caches
        _, mha_cache = mha(x, cos, sin, use_cache=True)
        _, gqa_cache = gqa(x, cos, sin, use_cache=True)

        # GQA cache should be 4x smaller
        mha_cache_size = mha_cache.key.numel() + mha_cache.value.numel()
        gqa_cache_size = gqa_cache.key.numel() + gqa_cache.value.numel()

        assert gqa_cache_size < mha_cache_size
        assert mha_cache_size / gqa_cache_size == pytest.approx(4.0, rel=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
