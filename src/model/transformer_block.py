"""
Transformer block implementation.

Implements modern transformer decoder block with:
- Pre-normalization (norm before attention/FFN)
- Grouped Query Attention (GQA)
- SwiGLU feed-forward
- RMSNorm
- Residual connections

This architecture follows current best practices for efficient
transformer-based language modeling.

References:
- Xiong et al., 2020. "On Layer Normalization in the Transformer Architecture" ICML 2020
- Vaswani et al., 2017. "Attention Is All You Need" NeurIPS 2017
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import GroupedQueryAttention, AttentionCache
from .feedforward import create_feedforward
from .normalization import create_norm_layer


class TransformerBlock(nn.Module):
    """
    Modern transformer decoder block.

    Architecture (pre-norm):
        x = x + Attention(Norm(x))
        x = x + FeedForward(Norm(x))

    This differs from original Transformer (post-norm):
        x = Norm(x + Attention(x))
        x = Norm(x + FeedForward(x))

    Pre-norm provides better gradient flow and training stability.

    Args:
        d_model: Model dimension
        num_query_heads: Number of query attention heads
        num_kv_heads: Number of key-value attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        norm_type: Type of normalization ("rmsnorm" or "layernorm")
        ff_type: Type of feed-forward ("swiglu", "geglu", "gelu")
        norm_eps: Epsilon for numerical stability in normalization
        bias: Whether to use bias in linear layers

    References:
        - Xiong et al., 2020. "On Layer Normalization in the Transformer Architecture" ICML 2020
        - Pre-normalization enables stable training of deep transformers
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        ff_type: str = "swiglu",
        norm_eps: float = 1e-6,
        bias: bool = False,
        max_seq_len: int = 2048
    ):
        super().__init__()

        # Attention layer with GQA
        self.attention = GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            bias=bias,
            max_seq_len=max_seq_len
        )

        # Feed-forward network (SwiGLU or variant)
        self.feed_forward = create_feedforward(
            d_model=d_model,
            d_ff=d_ff,
            ff_type=ff_type,
            bias=bias,
            dropout=dropout
        )

        # Pre-normalization layers
        self.attention_norm = create_norm_layer(
            dim=d_model,
            norm_type=norm_type,
            eps=norm_eps
        )

        self.ffn_norm = create_norm_layer(
            dim=d_model,
            norm_type=norm_type,
            eps=norm_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[AttentionCache] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[AttentionCache]]:
        """
        Forward pass of transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            cos: RoPE cosine embeddings
            sin: RoPE sine embeddings
            mask: Attention mask
            cache: Previous key-value cache
            use_cache: Whether to return updated cache

        Returns:
            - Output tensor of shape (batch, seq_len, d_model)
            - Updated cache (if use_cache=True), else None
        """
        # Pre-norm attention with residual connection
        # x = x + Attention(Norm(x))
        attn_output, new_cache = self.attention(
            self.attention_norm(x),
            cos=cos,
            sin=sin,
            mask=mask,
            cache=cache,
            use_cache=use_cache
        )
        x = x + attn_output

        # Pre-norm feed-forward with residual connection
        # x = x + FFN(Norm(x))
        ffn_output = self.feed_forward(self.ffn_norm(x))
        x = x + ffn_output

        return x, new_cache


class ParallelTransformerBlock(nn.Module):
    """
    Parallel transformer block (experimental).

    Instead of sequential attention → FFN, applies them in parallel.
    This can be faster but may affect model quality.

    Architecture:
        x = x + Attention(Norm(x)) + FeedForward(Norm(x))

    This can improve throughput but may affect model quality.

    Args:
        Same as TransformerBlock
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        ff_type: str = "swiglu",
        norm_eps: float = 1e-6,
        bias: bool = False,
        max_seq_len: int = 2048
    ):
        super().__init__()

        self.attention = GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            bias=bias,
            max_seq_len=max_seq_len
        )

        self.feed_forward = create_feedforward(
            d_model=d_model,
            d_ff=d_ff,
            ff_type=ff_type,
            bias=bias,
            dropout=dropout
        )

        # Single normalization layer for both paths
        self.norm = create_norm_layer(
            dim=d_model,
            norm_type=norm_type,
            eps=norm_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[AttentionCache] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[AttentionCache]]:
        """Forward pass with parallel attention and FFN."""
        # Normalize once
        x_norm = self.norm(x)

        # Compute attention and FFN in parallel
        attn_output, new_cache = self.attention(
            x_norm, cos, sin, mask, cache, use_cache
        )
        ffn_output = self.feed_forward(x_norm)

        # Combine with residual
        x = x + attn_output + ffn_output

        return x, new_cache


if __name__ == "__main__":
    # Test TransformerBlock
    print("Testing TransformerBlock...")

    from .embeddings import RotaryPositionEmbedding
    from .attention import create_causal_mask

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_query_heads = 8
    num_kv_heads = 2
    d_ff = 2048

    # Create transformer block
    block = TransformerBlock(
        d_model=d_model,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        d_ff=d_ff,
        dropout=0.1,
        norm_type="rmsnorm",
        ff_type="swiglu"
    )

    print(f"TransformerBlock configuration:")
    print(f"  d_model: {d_model}")
    print(f"  num_query_heads: {num_query_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  d_ff: {d_ff}")

    # Count parameters
    total_params = sum(p.numel() for p in block.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Create input
    x = torch.randn(batch_size, seq_len, d_model)

    # Create RoPE embeddings
    head_dim = d_model // num_query_heads
    rope = RotaryPositionEmbedding(dim=head_dim)
    cos, sin = rope(seq_len)

    # Create causal mask
    mask = create_causal_mask(seq_len, x.device)

    # Forward pass
    print("\nForward pass...")
    output, cache = block(x, cos, sin, mask, use_cache=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Cache key shape: {cache.key.shape}")

    # Test with cache (next token)
    print("\nTesting cached generation...")
    new_token = torch.randn(batch_size, 1, d_model)
    cos_new, sin_new = rope(1)

    output_new, cache_new = block(
        new_token, cos_new, sin_new,
        cache=cache, use_cache=True
    )

    print(f"New token output shape: {output_new.shape}")
    print(f"Updated cache size: {cache_new.key.shape[2]} tokens")

    # Test parallel block
    print("\n\nTesting ParallelTransformerBlock...")
    parallel_block = ParallelTransformerBlock(
        d_model=d_model,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        d_ff=d_ff
    )

    parallel_params = sum(p.numel() for p in parallel_block.parameters())
    print(f"Parallel block parameters: {parallel_params:,}")
    print(f"Difference from sequential: {parallel_params - total_params:,}")
    print("(Parallel has fewer params due to shared norm)")

    output_parallel, _ = parallel_block(x, cos, sin, mask)
    print(f"Parallel block output shape: {output_parallel.shape}")

    print("\n✓ TransformerBlock implementation complete!")
