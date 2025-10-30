"""
Feed-forward network layers for transformer models.

Implements various GLU (Gated Linear Unit) variants:
1. SwiGLU - Used in LLaMA 3, PaLM (SOTA as of 2024-2025)
2. GeGLU - Used in some vision transformers
3. Standard FFN with GELU - Traditional approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU: Swish-Gated Linear Unit.

    Combines Swish activation (also known as SiLU) with gating mechanism.
    This is the state-of-the-art feed-forward architecture as of 2024-2025.

    Formula:
        SwiGLU(x, W, V, W2) = (Swish(xW) ⊙ xV) W2
        where:
        - Swish(x) = x * sigmoid(x) = x * σ(x)
        - ⊙ is element-wise multiplication
        - W, V, W2 are linear transformations

    Note: SwiGLU requires ~2/3 more parameters than standard FFN for the same
    hidden dimension, but empirically performs better.

    Args:
        d_model: Model dimension (input/output size)
        d_ff: Feed-forward hidden dimension
        bias: Whether to include bias in linear layers
        dropout: Dropout probability

    References:
        - "GLU Variants Improve Transformer" (Shazeer, 2020)
        - Used in: LLaMA 3, PaLM, Mistral, Granite
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # SwiGLU uses three projections instead of two
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)  # Gate projection
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)  # Down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=bias)  # Up projection

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Output tensor of shape (..., d_model)
        """
        # Apply SwiGLU formula
        # F.silu is the Swish/SiLU activation: x * sigmoid(x)
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"d_model={self.d_model}, d_ff={self.d_ff}"


class GeGLU(nn.Module):
    """
    GeGLU: GELU-Gated Linear Unit.

    Similar to SwiGLU but uses GELU activation instead of Swish.

    Formula:
        GeGLU(x, W, V, W2) = (GELU(xW) ⊙ xV) W2

    Args:
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension
        bias: Whether to include bias
        dropout: Dropout probability

    References:
        - "GLU Variants Improve Transformer" (Shazeer, 2020)
        - Used in some vision transformers
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.w3 = nn.Linear(d_model, d_ff, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using GELU activation."""
        return self.w2(self.dropout(F.gelu(self.w1(x)) * self.w3(x)))


class FeedForward(nn.Module):
    """
    Standard feed-forward network with GELU activation.

    This is the traditional approach used in original Transformer and BERT.
    Kept for comparison, but SwiGLU is recommended for modern LLMs.

    Formula:
        FFN(x) = W2(Dropout(GELU(W1(x))))

    Args:
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension
        bias: Whether to include bias
        dropout: Dropout probability
        activation: Activation function ("gelu", "relu", "silu")
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        bias: bool = False,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # Select activation function
        activations = {
            "gelu": F.gelu,
            "relu": F.relu,
            "silu": F.silu,
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation}")

        self.activation = activations[activation.lower()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with chosen activation."""
        return self.w2(self.dropout(self.activation(self.w1(x))))


def create_feedforward(
    d_model: int,
    d_ff: int,
    ff_type: str = "swiglu",
    bias: bool = False,
    dropout: float = 0.0,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create feed-forward layers.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension
        ff_type: Type of feed-forward ("swiglu", "geglu", "gelu", "relu")
        bias: Whether to use bias
        dropout: Dropout probability
        **kwargs: Additional arguments for specific types

    Returns:
        Feed-forward module

    Example:
        >>> ff = create_feedforward(512, 2048, "swiglu")
        >>> x = torch.randn(2, 10, 512)
        >>> y = ff(x)
        >>> y.shape
        torch.Size([2, 10, 512])
    """
    ff_type = ff_type.lower()

    if ff_type == "swiglu":
        return SwiGLU(d_model, d_ff, bias, dropout)
    elif ff_type == "geglu":
        return GeGLU(d_model, d_ff, bias, dropout)
    elif ff_type in ["gelu", "relu", "silu"]:
        return FeedForward(d_model, d_ff, bias, dropout, activation=ff_type)
    else:
        raise ValueError(
            f"Unknown ff_type: {ff_type}. " f"Use 'swiglu', 'geglu', 'gelu', 'relu', or 'silu'"
        )


if __name__ == "__main__":
    # Test SwiGLU
    print("Testing SwiGLU...")

    batch_size, seq_len, d_model = 2, 10, 512
    d_ff = 2048

    x = torch.randn(batch_size, seq_len, d_model)

    # Test SwiGLU
    swiglu = SwiGLU(d_model, d_ff, dropout=0.1)
    y_swiglu = swiglu(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_swiglu.shape}")

    # Count parameters
    swiglu_params = sum(p.numel() for p in swiglu.parameters())
    print(f"SwiGLU parameters: {swiglu_params:,}")

    # Compare with standard FFN
    ffn = FeedForward(d_model, d_ff, dropout=0.1)
    y_ffn = ffn(x)
    ffn_params = sum(p.numel() for p in ffn.parameters())

    print(f"\nStandard FFN parameters: {ffn_params:,}")
    print(f"SwiGLU has {swiglu_params / ffn_params:.2f}x more parameters")
    print("(This is expected - SwiGLU uses 3 projections vs 2)")

    # Test all variants
    print("\nTesting all variants...")
    variants = ["swiglu", "geglu", "gelu", "relu"]

    for variant in variants:
        ff = create_feedforward(d_model, d_ff, ff_type=variant)
        y = ff(x)
        params = sum(p.numel() for p in ff.parameters())
        print(f"{variant:8s}: output shape {y.shape}, parameters: {params:,}")

    print("\n✓ Feed-forward implementation complete!")
