"""
Normalization layers for transformer models.

Implements RMSNorm (Root Mean Square Layer Normalization) as used in:
- LLaMA 3 (Meta)
- T5 (Google)
- Mistral

RMSNorm is simpler and faster than LayerNorm while maintaining similar performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler than LayerNorm: only normalizes by RMS, no mean subtraction.
    Formula: y = (x / RMS(x)) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        dim: Dimension of the input features
        eps: Small constant for numerical stability (default: 1e-6)

    References:
        - "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
        - Used in LLaMA, T5, and other modern LLMs
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter (same as LayerNorm's weight)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RMS normalization.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS: sqrt(mean(x^2))
        # keepdim=True maintains the dimension for broadcasting
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized and scaled tensor of shape (..., dim)
        """
        # Convert to float32 for stability, then convert back
        output_dtype = x.dtype
        x = x.float()

        # Normalize and scale
        x = self._norm(x)
        x = x * self.weight

        return x.to(output_dtype)

    def extra_repr(self) -> str:
        """String representation for print/debug."""
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


class LayerNorm(nn.Module):
    """
    Standard Layer Normalization (for comparison/fallback).

    Formula: y = ((x - mean(x)) / std(x)) * weight + bias

    Args:
        dim: Dimension of the input features
        eps: Small constant for numerical stability
        bias: Whether to include learnable bias term
    """

    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using PyTorch's native LayerNorm."""
        return F.layer_norm(
            x,
            self.weight.shape,
            self.weight,
            self.bias,
            self.eps
        )


def create_norm_layer(
    dim: int,
    norm_type: str = "rmsnorm",
    eps: float = 1e-6,
    **kwargs
) -> nn.Module:
    """
    Factory function to create normalization layers.

    Args:
        dim: Feature dimension
        norm_type: Type of normalization ("rmsnorm" or "layernorm")
        eps: Epsilon for numerical stability
        **kwargs: Additional arguments for specific norm types

    Returns:
        Normalization layer module

    Example:
        >>> norm = create_norm_layer(512, "rmsnorm")
        >>> x = torch.randn(2, 10, 512)
        >>> y = norm(x)
        >>> y.shape
        torch.Size([2, 10, 512])
    """
    norm_type = norm_type.lower()

    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps)
    elif norm_type == "layernorm":
        return LayerNorm(dim, eps, bias=kwargs.get("bias", False))
    else:
        raise ValueError(f"Unknown norm type: {norm_type}. Use 'rmsnorm' or 'layernorm'")


if __name__ == "__main__":
    # Test RMSNorm
    print("Testing RMSNorm...")

    batch_size, seq_len, d_model = 2, 10, 512
    x = torch.randn(batch_size, seq_len, d_model)

    # Create and test RMSNorm
    rms_norm = RMSNorm(d_model)
    y_rms = rms_norm(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_rms.shape}")
    print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"Output mean: {y_rms.mean():.4f}, std: {y_rms.std():.4f}")

    # Compare with LayerNorm
    layer_norm = LayerNorm(d_model)
    y_ln = layer_norm(x)

    print(f"\nLayerNorm output mean: {y_ln.mean():.4f}, std: {y_ln.std():.4f}")
    print(f"Difference between RMSNorm and LayerNorm: {(y_rms - y_ln).abs().mean():.6f}")

    print("\n✓ RMSNorm implementation complete!")
