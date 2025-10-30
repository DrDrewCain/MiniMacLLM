"""
Low-Rank Adaptation (LoRA) layer implementation.

LoRA adds trainable low-rank decomposition matrices to frozen pretrained weights.
This enables efficient fine-tuning with minimal parameters.

Key innovation: Instead of updating W directly, add ΔW = BA where:
- B is (d × r) - down projection
- A is (r × k) - up projection
- r << min(d, k) - rank is much smaller than dimensions

Forward pass: h = W₀x + BAx
where W₀ is frozen, only B and A are trainable.

References:
    - "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
    - Used for efficient fine-tuning of LLMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
import math


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA layers.

    Args:
        r: Rank of the low-rank decomposition (typically 8-64)
        alpha: Scaling factor for LoRA weights (typically 2*r)
        dropout: Dropout probability for LoRA layers
        merge_weights: Whether to merge LoRA weights into base weights
        target_modules: List of module names to apply LoRA to
    """
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    merge_weights: bool = False
    target_modules: list = None

    def __post_init__(self):
        """Set default target modules if not provided."""
        if self.target_modules is None:
            # Default: apply to all attention projections + FFN
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "w1", "w2", "w3"  # SwiGLU feed-forward
            ]

    @property
    def scaling(self) -> float:
        """Scaling factor applied to LoRA outputs."""
        return self.alpha / self.r


class LoRALayer(nn.Module):
    """
    LoRA layer that wraps a linear layer.

    Adds low-rank adaptation: h = Wx + (α/r)·BAx
    where W is frozen, B and A are trainable.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        r: Rank of low-rank matrices
        alpha: Scaling factor
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.in_features = in_features
        self.out_features = out_features

        # LoRA parameters
        # A: (r, in_features) - initialize with Kaiming uniform
        # B: (out_features, r) - initialize with zeros
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Initialize A with Kaiming uniform (like default linear layer)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B starts at zero so LoRA starts with no effect

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LoRA layer.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            LoRA output of shape (..., out_features)
        """
        # LoRA path: x @ A.T @ B.T = (x @ A.T) @ B.T
        # This is equivalent to: B @ A @ x
        result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result * self.scaling

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"r={self.r}, "
            f"alpha={self.alpha}, "
            f"scaling={self.scaling:.4f}"
        )


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Combines a frozen pretrained linear layer with trainable LoRA parameters.

    Forward pass: h = Wx + ΔWx = Wx + (α/r)·BAx

    Args:
        base_layer: Pretrained linear layer (will be frozen)
        r: Rank of LoRA
        alpha: Scaling factor
        dropout: Dropout probability
        merged: Whether LoRA weights are merged into base
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()

        # Store base layer (frozen)
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Add LoRA layer
        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout
        )

        # Track if LoRA is merged into base weights
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        if self.merged:
            # LoRA is merged into base weights
            return self.base_layer(x)
        else:
            # LoRA is separate: base + lora
            return self.base_layer(x) + self.lora(x)

    def merge(self):
        """
        Merge LoRA weights into base layer for faster inference.

        After merging:
        W_new = W_base + (α/r)·BA

        This is useful for deployment when you don't need to switch adapters.
        """
        if not self.merged:
            # Compute ΔW = (α/r)·BA
            delta_w = (self.lora.scaling *
                      self.lora.lora_B @ self.lora.lora_A)

            # Merge into base weights
            self.base_layer.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """
        Unmerge LoRA weights from base layer.

        After unmerging:
        W_new = W_base - (α/r)·BA

        This restores the original base weights.
        """
        if self.merged:
            # Compute ΔW = (α/r)·BA
            delta_w = (self.lora.scaling *
                      self.lora.lora_B @ self.lora.lora_A)

            # Subtract from base weights
            self.base_layer.weight.data -= delta_w
            self.merged = False

    def get_lora_parameters(self):
        """Get only LoRA parameters (for optimization)."""
        return [self.lora.lora_A, self.lora.lora_B]

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"r={self.lora.r}, "
            f"merged={self.merged}"
        )


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """
    Freeze all parameters except LoRA parameters.

    Args:
        model: Model with LoRA layers
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Then, unfreeze LoRA parameters
    for module in model.modules():
        if isinstance(module, (LoRALayer, LinearWithLoRA)):
            if isinstance(module, LinearWithLoRA):
                module = module.lora
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True


def get_lora_parameters(model: nn.Module) -> list:
    """
    Get all LoRA parameters from a model.

    Args:
        model: Model with LoRA layers

    Returns:
        List of LoRA parameter tensors
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, (LoRALayer, LinearWithLoRA)):
            if isinstance(module, LinearWithLoRA):
                lora_params.extend(module.get_lora_parameters())
            else:
                lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def count_lora_parameters(model: nn.Module) -> dict:
    """
    Count LoRA parameters in a model.

    Args:
        model: Model with LoRA layers

    Returns:
        Dictionary with parameter counts
    """
    lora_params = get_lora_parameters(model)

    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for p in lora_params)
    base_params_count = total_params - lora_params_count

    return {
        'total': total_params,
        'lora': lora_params_count,
        'base': base_params_count,
        'lora_percentage': 100.0 * lora_params_count / total_params if total_params > 0 else 0
    }


if __name__ == "__main__":
    # Test LoRA implementation
    print("Testing LoRA layers...")

    # Create a base linear layer
    in_features = 512
    out_features = 512
    base_linear = nn.Linear(in_features, out_features, bias=False)

    # Wrap with LoRA
    r = 16
    alpha = 32
    lora_linear = LinearWithLoRA(base_linear, r=r, alpha=alpha, dropout=0.1)

    print(f"\nBase layer: {in_features} → {out_features}")
    print(f"LoRA rank: {r}")
    print(f"LoRA alpha: {alpha}")
    print(f"LoRA scaling: {alpha / r}")

    # Count parameters
    base_params = base_linear.weight.numel()
    lora_params = lora_linear.lora.lora_A.numel() + lora_linear.lora.lora_B.numel()

    print(f"\nParameter counts:")
    print(f"  Base: {base_params:,}")
    print(f"  LoRA: {lora_params:,}")
    print(f"  Reduction: {100.0 * lora_params / base_params:.2f}%")

    # Test forward pass
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, in_features)

    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")

    # Before LoRA (just base)
    with torch.no_grad():
        y_base = base_linear(x)
    print(f"  Base output shape: {y_base.shape}")

    # With LoRA
    y_lora = lora_linear(x)
    print(f"  LoRA output shape: {y_lora.shape}")

    # Test merge/unmerge
    print(f"\nTesting merge/unmerge:")
    print(f"  Merged before: {lora_linear.merged}")

    lora_linear.merge()
    print(f"  Merged after merge(): {lora_linear.merged}")

    y_merged = lora_linear(x)
    print(f"  Output matches: {torch.allclose(y_lora, y_merged, atol=1e-5)}")

    lora_linear.unmerge()
    print(f"  Merged after unmerge(): {lora_linear.merged}")

    # Test parameter marking
    print(f"\nTesting parameter freezing:")
    trainable_before = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
    print(f"  Trainable params before: {trainable_before:,}")

    mark_only_lora_as_trainable(lora_linear)
    trainable_after = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
    print(f"  Trainable params after: {trainable_after:,}")
    print(f"  Expected: {lora_params:,}")
    print(f"  Match: {trainable_after == lora_params}")

    print("\n✓ LoRA layer implementation complete!")
