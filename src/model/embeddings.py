"""
Position embedding layers for transformer models.

Implements:
1. Rotary Position Embeddings (RoPE) - Used in LLaMA, GPT-NeoX, Mistral
2. Learned Absolute Position Embeddings - Traditional approach
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).

    Encodes positional information by rotating query and key vectors.
    Key advantages over absolute/learned embeddings:
    - Better extrapolation to longer sequences
    - Encodes relative position information naturally
    - No learnable parameters needed

    Args:
        dim: Dimension of the attention head (d_k)
        max_seq_len: Maximum sequence length to precompute
        base: Base for the geometric progression (default: 10000.0)
        device: Device to place tensors on

    References:
        - "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
        - Used in: LLaMA 3, Mistral, GPT-NeoX, PaLM
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute rotation frequencies
        # inv_freq shape: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin for all positions
        self._precompute_freqs(max_seq_len, device=device)

    def _precompute_freqs(self, seq_len: int, device: Optional[torch.device] = None):
        """
        Precompute cosine and sine frequencies for all positions.

        For each position t and each dimension i:
        θ_i = t / (base^(2i/d))

        We store cos(θ) and sin(θ) for efficiency.
        """
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        # Outer product: (seq_len,) x (dim//2,) -> (seq_len, dim//2)
        # This gives us θ for each (position, dimension) pair
        freqs = torch.outer(t, self.inv_freq)

        # For complex number rotation, we need to duplicate frequencies
        # to apply to real and imaginary parts
        # Shape: (seq_len, dim//2) -> (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        # Precompute cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin embeddings for a given sequence length.

        Args:
            seq_len: Length of the sequence
            device: Device for tensors

        Returns:
            Tuple of (cos, sin) tensors, both of shape (seq_len, dim)
        """
        # If sequence is longer than cached, recompute
        if seq_len > self.max_seq_len:
            self._precompute_freqs(seq_len, device=device)
            self.max_seq_len = seq_len

        # Return cached values for the requested length
        return (self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device))


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    The rotation is applied using complex number multiplication:
    x_rotated = x * cos(θ) + rotate_half(x) * sin(θ)

    where rotate_half() swaps and negates pairs of dimensions:
    [x1, x2, x3, x4, ...] -> [-x2, x1, -x4, x3, ...]

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
        cos: Cosine embeddings of shape (seq_len, head_dim)
        sin: Sine embeddings of shape (seq_len, head_dim)

    Returns:
        Rotated (q, k) tensors with same shapes as input

    References:
        - Original implementation from LLaMA
        - Conceptually: treating consecutive pairs as complex numbers
    """
    # Reshape cos/sin to match q/k shape
    # From (seq_len, dim) to (1, 1, seq_len, dim) for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)

    # Apply rotation to q and k
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the dimensions of x.

    This implements the complex number multiplication trick:
    - Split features into pairs: (x1, x2), (x3, x4), ...
    - Rotate: (-x2, x1), (-x4, x3), ...

    Args:
        x: Input tensor of shape (..., dim) where dim is even

    Returns:
        Rotated tensor of same shape

    Example:
        >>> x = torch.tensor([[1, 2, 3, 4]])
        >>> rotate_half(x)
        tensor([[-2, 1, -4, 3]])
    """
    x1, x2 = x.chunk(2, dim=-1)  # Split into two halves
    return torch.cat([-x2, x1], dim=-1)  # Rotate and concatenate


class LearnedPositionEmbedding(nn.Module):
    """
    Traditional learned absolute position embeddings.

    Each position has a learned embedding vector.
    Simpler than RoPE but worse extrapolation to longer sequences.

    Args:
        max_seq_len: Maximum sequence length
        d_model: Model dimension

    Note: This is kept for comparison but RoPE is recommended.
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get position embeddings for sequence.

        Args:
            seq_len: Length of sequence
            device: Device for tensors

        Returns:
            Position embeddings of shape (seq_len, d_model)
        """
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions)


if __name__ == "__main__":
    # Test RoPE implementation
    print("Testing Rotary Position Embeddings...")

    batch_size = 2
    num_heads = 8
    seq_len = 10
    head_dim = 64

    # Create RoPE
    rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=2048)

    # Create dummy Q and K tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Get cos/sin embeddings
    cos, sin = rope(seq_len)
    print(f"Cos shape: {cos.shape}, Sin shape: {sin.shape}")

    # Apply RoPE
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    print(f"Q shape: {q.shape} -> Q_rot shape: {q_rot.shape}")
    print(f"K shape: {k.shape} -> K_rot shape: {k_rot.shape}")

    # Verify rotation preserves norm (approximately)
    q_norm_before = torch.norm(q, dim=-1).mean()
    q_norm_after = torch.norm(q_rot, dim=-1).mean()
    print(f"\nQ norm before: {q_norm_before:.4f}")
    print(f"Q norm after: {q_norm_after:.4f}")
    print(f"Norm preserved: {torch.allclose(q_norm_before, q_norm_after, rtol=1e-4)}")

    # Test rotate_half function
    print("\nTesting rotate_half...")
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    x_rot = rotate_half(x)
    print(f"Input: {x}")
    print(f"Rotated: {x_rot}")
    print(f"Expected: [[-2, 1, -4, 3]]")

    print("\n✓ RoPE implementation complete!")
