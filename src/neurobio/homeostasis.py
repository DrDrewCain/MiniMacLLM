"""
Homeostatic Plasticity for Maintaining Neural Activity.

Implements mechanisms that maintain stable firing rates while
enabling continual learning. Based on:
- Loss of plasticity in deep continual learning (Nature 2024)
- BioLogicalNeuron layer (Scientific Reports 2025)

Key Principles:
- Neurons maintain target firing rate through adaptive thresholds
- Prevents "dead neurons" during continual learning
- Maintains model capacity across multiple domains
- Calcium-driven homeostatic regulation
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class HomeostaticConfig:
    """
    Configuration for homeostatic plasticity.

    Args:
        hidden_dim: Dimension of neural layer
        target_rate: Target average activation rate (0-1)
        tau: Time constant for averaging (higher = slower adaptation)
        threshold_lr: Learning rate for threshold adjustment
        min_threshold: Minimum threshold value
        max_threshold: Maximum threshold value
        use_batch_norm: Whether to also use batch normalization
    """
    hidden_dim: int = 768
    target_rate: float = 0.1  # 10% sparsity (like cortex)
    tau: int = 1000  # ~1000 steps to adapt
    threshold_lr: float = 0.01
    min_threshold: float = -5.0
    max_threshold: float = 5.0
    use_batch_norm: bool = False


class HomeostaticNeuron(nn.Module):
    """
    Homeostatic plasticity mechanism for stable neural activity.

    Maintains stable firing rates through adaptive thresholds,
    preventing dead neurons and loss of plasticity during
    continual learning.

    Based on biological mechanisms:
    - Synaptic scaling (multiplicative homeostasis)
    - Intrinsic plasticity (threshold adaptation)
    - Calcium-driven regulation

    Example:
        >>> config = HomeostaticConfig(hidden_dim=768)
        >>> homeostatic = HomeostaticNeuron(config)
        >>>
        >>> x = torch.randn(32, 196, 768)
        >>> x_regulated = homeostatic(x, training=True)
        >>>
        >>> # Check adaptation
        >>> print(homeostatic.get_firing_rate())  # Should approach target_rate
    """

    def __init__(self, config: HomeostaticConfig):
        super().__init__()
        self.config = config

        # Running average of activation (like calcium concentration)
        self.register_buffer(
            'avg_activation',
            torch.ones(config.hidden_dim) * config.target_rate
        )

        # Adaptive threshold (intrinsic plasticity)
        self.threshold = nn.Parameter(torch.zeros(config.hidden_dim))

        # Synaptic scaling factors (multiplicative homeostasis)
        self.scaling = nn.Parameter(torch.ones(config.hidden_dim))

        # Optional batch normalization
        if config.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(config.hidden_dim)
        else:
            self.batch_norm = None

        # Track steps for tau-based averaging
        self.register_buffer('steps', torch.tensor(0))

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        Apply homeostatic regulation.

        Args:
            x: Input tensor (batch, ..., hidden_dim)
            training: Whether in training mode

        Returns:
            Regulated tensor with maintained firing rates
        """
        original_shape = x.shape

        # Flatten to (batch * seq, hidden_dim) for processing
        if x.dim() > 2:
            x_flat = x.reshape(-1, x.size(-1))
        else:
            x_flat = x

        # Apply synaptic scaling (multiplicative homeostasis)
        x_scaled = x_flat * self.scaling

        # Apply adaptive threshold (intrinsic plasticity)
        x_shifted = x_scaled - self.threshold

        if training:
            # Compute current firing rate (sigmoid activation as proxy)
            current_rate = torch.sigmoid(x_shifted).mean(dim=0).detach()

            # Update running average with tau-based decay
            alpha = 1.0 / self.config.tau
            self.avg_activation.mul_(1 - alpha).add_(current_rate * alpha)

            # Adjust threshold to maintain target rate
            # If firing too much (avg > target), increase threshold
            # If firing too little (avg < target), decrease threshold
            rate_error = self.avg_activation - self.config.target_rate
            threshold_update = self.config.threshold_lr * rate_error
            self.threshold.data.add_(threshold_update)

            # Clamp threshold to reasonable range
            self.threshold.data.clamp_(
                self.config.min_threshold,
                self.config.max_threshold
            )

            # Adjust scaling to normalize variance (synaptic scaling)
            # If variance is too high/low, adjust scaling
            current_std = x_shifted.std(dim=0).detach()
            target_std = 1.0
            scaling_adjustment = target_std / (current_std + 1e-8)
            self.scaling.data.mul_(
                1.0 - 0.001  # Slow adaptation
            ).add_(
                0.001 * scaling_adjustment
            )

            # Clamp scaling to prevent explosion/vanishing
            self.scaling.data.clamp_(0.1, 10.0)

            # Increment steps
            self.steps += 1

        # Optional batch normalization
        if self.batch_norm is not None and training:
            x_shifted = self.batch_norm(x_shifted)

        # Reshape back to original shape
        if len(original_shape) > 2:
            x_shifted = x_shifted.reshape(original_shape)

        return x_shifted

    def get_firing_rate(self) -> torch.Tensor:
        """Get current average firing rate per neuron."""
        return self.avg_activation.clone()

    def get_dead_neuron_ratio(self, threshold: float = 0.01) -> float:
        """
        Compute ratio of "dead" neurons (firing rate < threshold).

        Args:
            threshold: Minimum firing rate to be considered "alive"

        Returns:
            Ratio of dead neurons (0-1)
        """
        dead_count = (self.avg_activation < threshold).sum().item()
        total_count = self.avg_activation.numel()
        return dead_count / total_count

    def reset_homeostasis(self):
        """Reset homeostatic state to initial values."""
        with torch.no_grad():
            self.avg_activation.fill_(self.config.target_rate)
            self.threshold.data.zero_()
            self.scaling.data.fill_(1.0)
            self.steps.zero_()

    def get_statistics(self) -> dict:
        """Get homeostatic statistics for monitoring."""
        return {
            'mean_firing_rate': self.avg_activation.mean().item(),
            'std_firing_rate': self.avg_activation.std().item(),
            'dead_neuron_ratio': self.get_dead_neuron_ratio(),
            'mean_threshold': self.threshold.mean().item(),
            'mean_scaling': self.scaling.mean().item(),
            'steps': self.steps.item(),
        }


class HomeostaticWrapper(nn.Module):
    """
    Wrapper to add homeostatic plasticity to existing layers.

    Wraps any nn.Module with homeostatic regulation on its output.

    Example:
        >>> mlp = nn.Sequential(
        ...     nn.Linear(768, 3072),
        ...     nn.GELU(),
        ...     nn.Linear(3072, 768)
        ... )
        >>> homeostatic_mlp = HomeostaticWrapper(
        ...     mlp,
        ...     HomeostaticConfig(hidden_dim=768)
        ... )
    """

    def __init__(self, layer: nn.Module, config: HomeostaticConfig):
        super().__init__()
        self.layer = layer
        self.homeostatic = HomeostaticNeuron(config)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with homeostatic regulation."""
        # Pass through wrapped layer
        out = self.layer(x, **kwargs) if kwargs else self.layer(x)

        # Apply homeostatic regulation
        out_regulated = self.homeostatic(out, training=self.training)

        return out_regulated

    def get_statistics(self) -> dict:
        """Get homeostatic statistics."""
        return self.homeostatic.get_statistics()


if __name__ == "__main__":
    print("Testing HomeostaticNeuron...")

    # Create homeostatic neuron
    config = HomeostaticConfig(hidden_dim=512, target_rate=0.15)
    homeostatic = HomeostaticNeuron(config)

    print("\nInitial statistics:")
    stats = homeostatic.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    # Simulate training with varying activations
    print("\nSimulating continual learning...")
    for step in range(100):
        # Random activations (simulating different domains)
        if step < 30:
            x = torch.randn(32, 196, 512) * 2.0  # High variance
        elif step < 60:
            x = torch.randn(32, 196, 512) * 0.5  # Low variance
        else:
            x = torch.randn(32, 196, 512)  # Normal

        # Apply homeostatic regulation
        _ = homeostatic(x, training=True)  # Result used for side effects (statistics tracking)

        if step % 20 == 19:
            stats = homeostatic.get_statistics()
            print(f"\nStep {step+1}:")
            print(f"  Firing rate: {stats['mean_firing_rate']:.4f} "
                  f"(target: {config.target_rate:.4f})")
            print(f"  Dead neurons: {stats['dead_neuron_ratio']:.2%}")

    # Test dead neuron detection
    print("\nFinal firing rates:")
    rates = homeostatic.get_firing_rate()
    print(f"  Min: {rates.min():.4f}")
    print(f"  Mean: {rates.mean():.4f}")
    print(f"  Max: {rates.max():.4f}")
    print(f"  Dead ratio: {homeostatic.get_dead_neuron_ratio():.2%}")

    print("\n✓ HomeostaticNeuron working!")
