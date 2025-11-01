"""
Dendritic Computation for Parameter-Efficient Learning.

Implements compartmentalized dendritic processing inspired by
biological neurons. Based on:
- Dendritic ANNs (Nature Communications 2025)
- Feedforward Tree Networks (FFTN, 2024)

Key Principles:
- Multiple dendritic branches perform semi-independent computation
- Context-dependent gating (like NMDA receptors)
- Somatic integration of dendritic signals
- "Network-in-a-neuron" capability

Benefits:
- Parameter efficiency (fewer params, same performance)
- Better generalization and robustness
- Biological plausibility
"""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass


@dataclass
class DendriticConfig:
    """
    Configuration for dendritic computation layer.

    Args:
        d_model: Hidden dimension
        num_dendrites: Number of dendritic branches (2-8 typical)
        use_context_gating: Whether to use context-dependent gating
        dropout: Dropout probability
        nonlinearity: Activation function ('silu', 'gelu', 'relu')
    """
    d_model: int = 768
    num_dendrites: int = 4
    use_context_gating: bool = True
    dropout: float = 0.0
    nonlinearity: str = 'silu'


class DendriticLayer(nn.Module):
    """
    Dendritic computation layer with compartmentalized processing.

    Each neuron has multiple dendritic branches that process inputs
    semi-independently, with context-dependent gating controlling
    information flow.

    Biological inspiration:
    - Dendrites compute nonlinear functions locally
    - NMDA receptors provide context-dependent amplification
    - Soma integrates dendritic signals

    Example:
        >>> config = DendriticConfig(d_model=768, num_dendrites=4)
        >>> dendritic = DendriticLayer(config)
        >>>
        >>> x = torch.randn(32, 196, 768)
        >>> context = torch.randn(32, 196, 768)  # Optional
        >>> output = dendritic(x, context=context)
    """

    def __init__(self, config: DendriticConfig):
        super().__init__()
        self.config = config

        # Each dendrite processes input semi-independently
        dendrite_dim = config.d_model // config.num_dendrites

        self.dendrites = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, dendrite_dim),
                self._get_activation(config.nonlinearity),
            )
            for _ in range(config.num_dendrites)
        ])

        # Context-dependent gating (like NMDA receptors)
        if config.use_context_gating:
            self.context_gate = nn.Sequential(
                nn.Linear(config.d_model, config.num_dendrites),
                nn.Sigmoid()
            )
        else:
            self.context_gate = None

        # Soma integrates dendritic signals
        self.soma = nn.Linear(config.d_model, config.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
        }
        return activations.get(name, nn.SiLU())

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Dendritic computation with context-dependent gating.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            context: Optional context for gating (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Dendritic processing
        dendritic_outputs = []
        for dendrite in self.dendrites:
            dendritic_out = dendrite(x)
            dendritic_outputs.append(dendritic_out)

        # Concatenate dendritic outputs
        dendritic_concat = torch.cat(dendritic_outputs, dim=-1)

        # Context-dependent gating
        if self.context_gate is not None and context is not None:
            # Compute gate values from context
            gates = self.context_gate(context)  # (batch, seq, num_dendrites)

            # Expand gates to match dendritic dimensions
            # gates: (batch, seq, num_dendrites) -> (batch, seq, d_model)
            dendrite_dim = self.config.d_model // self.config.num_dendrites
            gates_expanded = gates.unsqueeze(-1).expand(
                -1, -1, -1, dendrite_dim
            ).reshape(gates.size(0), gates.size(1), self.config.d_model)

            # Apply gating
            dendritic_concat = dendritic_concat * gates_expanded

        # Somatic integration
        output = self.soma(dendritic_concat)

        # Dropout
        output = self.dropout(output)

        return output

    def get_dendrite_activations(
        self,
        x: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Get individual dendrite activations for analysis.

        Args:
            x: Input tensor

        Returns:
            List of dendrite outputs
        """
        activations = []
        for dendrite in self.dendrites:
            activations.append(dendrite(x).detach())
        return activations


class DendriticMLP(nn.Module):
    """
    MLP with dendritic computation (replacement for standard MLP).

    Two-layer MLP where first layer uses dendritic processing.

    Example:
        >>> mlp = DendriticMLP(
        ...     d_model=768,
        ...     mlp_ratio=4.0,
        ...     num_dendrites=4
        ... )
        >>> x = torch.randn(32, 196, 768)
        >>> output = mlp(x)
    """

    def __init__(
        self,
        d_model: int,
        mlp_ratio: float = 4.0,
        num_dendrites: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()

        mlp_hidden_dim = int(d_model * mlp_ratio)

        # First layer: expand with dendritic processing
        dendritic_config = DendriticConfig(
            d_model=d_model,
            num_dendrites=num_dendrites,
            dropout=dropout
        )

        # Override output dimension
        self.fc1_dendritic = DendriticLayer(dendritic_config)
        self.fc1_expand = nn.Linear(d_model, mlp_hidden_dim)

        # Activation
        self.activation = nn.GELU()

        # Second layer: project back
        self.fc2 = nn.Linear(mlp_hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dendritic MLP."""
        # Dendritic processing + expansion
        x = self.fc1_dendritic(x)
        x = self.fc1_expand(x)

        # Activation
        x = self.activation(x)
        x = self.dropout(x)

        # Project back
        x = self.fc2(x)
        x = self.dropout(x)

        return x


if __name__ == "__main__":
    print("Testing DendriticLayer...")

    # Create dendritic layer
    config = DendriticConfig(d_model=512, num_dendrites=4)
    dendritic = DendriticLayer(config)

    # Count parameters
    num_params = sum(p.numel() for p in dendritic.parameters())
    print(f"\nDendritic layer parameters: {num_params:,}")

    # Compare with standard linear layer
    standard = nn.Linear(512, 512)
    standard_params = sum(p.numel() for p in standard.parameters())
    print(f"Standard linear parameters: {standard_params:,}")
    print(f"Parameter ratio: {num_params/standard_params:.2f}x")

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 10, 512)
    context = torch.randn(2, 10, 512)

    output = dendritic(x, context=context)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    # Test dendrite activations
    activations = dendritic.get_dendrite_activations(x)
    print(f"\nIndividual dendrite activations:")
    for i, act in enumerate(activations):
        print(f"  Dendrite {i}: {act.shape}, mean={act.mean():.3f}")

    # Test dendritic MLP
    print("\nTesting DendriticMLP...")
    mlp = DendriticMLP(d_model=512, mlp_ratio=4.0, num_dendrites=4)
    mlp_params = sum(p.numel() for p in mlp.parameters())
    print(f"  Parameters: {mlp_params:,}")

    output = mlp(x)
    print(f"  Output shape: {output.shape}")

    print("\n✓ DendriticLayer working!")
