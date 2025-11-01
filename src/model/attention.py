"""
Attention mechanisms for transformer models.

Implements:
1. Grouped Query Attention (GQA) - Memory-efficient attention with shared KV heads
2. Multi-Head Attention (MHA) - Traditional approach (for comparison)
3. Multi-Query Attention (MQA) - Extreme efficiency variant

GQA provides the best balance of quality and efficiency for modern transformers.

References:
- Ainslie et al., 2023. "GQA: Training Generalized Multi-Query Transformer Models
  from Multi-Head Checkpoints" arXiv:2305.13245
- Shazeer, 2019. "Fast Transformer Decoding: One Write-Head is All You Need"
  arXiv:1911.02150 (Multi-Query Attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass

from .embeddings import apply_rotary_pos_emb


@dataclass
class AttentionCache:
    """
    Cache for storing key-value pairs during inference.

    Enables fast autoregressive generation by avoiding recomputation
    of keys and values for previous tokens.
    """
    key: torch.Tensor  # (batch, num_kv_heads, seq_len, head_dim)
    value: torch.Tensor  # (batch, num_kv_heads, seq_len, head_dim)

    def update(self, new_key: torch.Tensor, new_value: torch.Tensor) -> 'AttentionCache':
        """
        Append new keys and values to the cache.

        Args:
            new_key: New keys of shape (batch, num_kv_heads, new_seq_len, head_dim)
            new_value: New values of same shape

        Returns:
            Updated cache
        """
        self.key = torch.cat([self.key, new_key], dim=2)
        self.value = torch.cat([self.value, new_value], dim=2)
        return self


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    Key innovation: Share key-value heads across multiple query heads.
    - Q heads: num_query_heads (e.g., 32)
    - KV heads: num_kv_heads (e.g., 8)
    - Each KV head is shared by num_query_heads // num_kv_heads query heads

    Benefits:
    - Reduces KV cache size by factor of (num_query_heads / num_kv_heads)
    - Minimal quality degradation compared to MHA
    - Faster inference while maintaining training speed

    Special cases:
    - num_kv_heads == num_query_heads: Multi-Head Attention (MHA)
    - num_kv_heads == 1: Multi-Query Attention (MQA)

    Args:
        d_model: Model dimension
        num_query_heads: Number of query heads
        num_kv_heads: Number of key-value heads (must divide num_query_heads)
        dropout: Dropout probability
        bias: Whether to use bias in linear projections

    References:
        - Ainslie et al., 2023. "GQA: Training Generalized Multi-Query Transformer Models
          from Multi-Head Checkpoints" arXiv:2305.13245
        - Reduces KV cache memory by 4-8x with minimal quality loss
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 2048
    ):
        super().__init__()

        # Validate configuration
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads ({num_query_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if d_model % num_query_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by "
                f"num_query_heads ({num_query_heads})"
            )

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.max_seq_len = max_seq_len

        # Q projection has all heads
        self.q_proj = nn.Linear(d_model, num_query_heads * self.head_dim, bias=bias)

        # K, V projections have fewer heads
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)

        # Output projection
        self.o_proj = nn.Linear(num_query_heads * self.head_dim, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)

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
        Forward pass of GQA.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            cos: RoPE cosine embeddings of shape (seq_len, head_dim)
            sin: RoPE sine embeddings of shape (seq_len, head_dim)
            mask: Attention mask of shape (batch, 1, seq_len, seq_len) or None
            cache: Previous key-value cache for fast generation
            use_cache: Whether to return updated cache

        Returns:
            - Output tensor of shape (batch, seq_len, d_model)
            - Updated cache (if use_cache=True), else None
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, num_query_heads * head_dim)
        k = self.k_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)

        # Reshape to separate heads
        # Q: (batch, seq_len, num_query_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_query_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose for attention computation
        # (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Update cache if provided
        if cache is not None:
            k = torch.cat([cache.key, k], dim=2)
            v = torch.cat([cache.value, v], dim=2)

        # Create/update cache if requested
        new_cache = None
        if use_cache:
            new_cache = AttentionCache(key=k, value=v)

        # Expand K, V to match number of Q heads (GQA grouping)
        # Repeat each KV head num_queries_per_kv times
        if self.num_queries_per_kv > 1:
            k = self._repeat_kv(k, self.num_queries_per_kv)
            v = self._repeat_kv(v, self.num_queries_per_kv)

        # Compute attention scores
        # Q: (batch, num_query_heads, seq_len, head_dim)
        # K: (batch, num_query_heads, kv_seq_len, head_dim)
        # scores: (batch, num_query_heads, seq_len, kv_seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores + mask  # Mask should contain -inf for masked positions

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # attn_weights: (batch, num_query_heads, seq_len, kv_seq_len)
        # v: (batch, num_query_heads, kv_seq_len, head_dim)
        # attn_output: (batch, num_query_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back to (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.o_proj(attn_output)

        return output, new_cache

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat key or value tensor to match number of query heads.

        Args:
            x: KV tensor of shape (batch, num_kv_heads, seq_len, head_dim)
            n_rep: Number of times to repeat each KV head

        Returns:
            Expanded tensor of shape (batch, num_kv_heads * n_rep, seq_len, head_dim)

        Example:
            If we have 8 KV heads and need 32 Q heads (n_rep=4):
            [kv0, kv1, ..., kv7] -> [kv0, kv0, kv0, kv0, kv1, kv1, kv1, kv1, ..., kv7, kv7, kv7, kv7]
        """
        if n_rep == 1:
            return x

        batch, num_kv_heads, seq_len, head_dim = x.shape

        # Repeat along the head dimension
        # (batch, num_kv_heads, 1, seq_len, head_dim) -> (batch, num_kv_heads, n_rep, seq_len, head_dim)
        x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)

        # Reshape to combine repeated heads
        # (batch, num_kv_heads * n_rep, seq_len, head_dim)
        return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"d_model={self.d_model}, "
            f"num_query_heads={self.num_query_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"queries_per_kv={self.num_queries_per_kv}"
        )


class MultiHeadAttention(nn.Module):
    """
    Traditional Multi-Head Attention (MHA).

    Kept for comparison and backwards compatibility.
    In modern LLMs, GQA is preferred for efficiency.

    This is equivalent to GQA with num_kv_heads == num_query_heads.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 2048
    ):
        super().__init__()
        # MHA is just GQA with equal number of Q and KV heads
        self.gqa = GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=num_heads,
            num_kv_heads=num_heads,  # Same as Q heads
            dropout=dropout,
            bias=bias,
            max_seq_len=max_seq_len
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
        """Forward pass - delegates to GQA."""
        return self.gqa(x, cos, sin, mask, cache, use_cache)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal attention mask to prevent attending to future positions.

    The mask contains 0 for allowed positions and -inf for masked positions.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Mask of shape (1, 1, seq_len, seq_len)

    Example:
        For seq_len=4:
        [[0, -inf, -inf, -inf],
         [0,    0, -inf, -inf],
         [0,    0,    0, -inf],
         [0,    0,    0,    0]]
    """
    # Create lower triangular matrix
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)

    # Convert to attention mask format
    mask = mask.masked_fill(mask == 1, float('-inf'))

    # Add batch and head dimensions
    return mask.view(1, 1, seq_len, seq_len)


if __name__ == "__main__":
    # Test GQA implementation
    print("Testing Grouped Query Attention...")

    from .embeddings import RotaryPositionEmbedding

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_query_heads = 8
    num_kv_heads = 2  # 4 queries per KV head

    # Create GQA layer
    gqa = GroupedQueryAttention(
        d_model=d_model,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.1
    )

    print(f"GQA Configuration:")
    print(f"  Queries per KV head: {gqa.num_queries_per_kv}")
    print(f"  Head dimension: {gqa.head_dim}")
    print(f"  KV cache reduction: {num_query_heads / num_kv_heads}x")

    # Create input
    x = torch.randn(batch_size, seq_len, d_model)

    # Create RoPE embeddings
    rope = RotaryPositionEmbedding(dim=gqa.head_dim)
    cos, sin = rope(seq_len)

    # Create causal mask
    mask = create_causal_mask(seq_len, x.device)

    # Forward pass
    output, cache = gqa(x, cos, sin, mask, use_cache=True)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Cache key shape: {cache.key.shape}")
    print(f"Cache value shape: {cache.value.shape}")

    # Test with cache (simulating next token generation)
    print("\nTesting with cache...")
    new_token = torch.randn(batch_size, 1, d_model)
    cos_new, sin_new = rope(1)
    output_new, cache_new = gqa(new_token, cos_new, sin_new, cache=cache, use_cache=True)

    print(f"New token input shape: {new_token.shape}")
    print(f"New token output shape: {output_new.shape}")
    print(f"Updated cache key shape: {cache_new.key.shape}")

    print("\n✓ GQA implementation complete!")
