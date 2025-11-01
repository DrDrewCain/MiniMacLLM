"""
Liquid Time Constants - Adaptive Temporal Dynamics for Neural Networks.

Implements continuous-time neural networks with adaptive time constants based on:
- Liquid Time-Constant Networks (LTCs)
- Adaptive temporal processing
- Bounded, stable dynamics (guaranteed convergence)
- Compact parameter usage

Key principles:
- Neurons have adaptive time constants (not fixed)
- Continuous-time ODEs (not discrete time steps)
- Closed-form solution (no iterative ODE solvers)
- Bounded activations (stability guarantees)

References:
- Hasani et al., 2021: Liquid Time-Constant Networks (arXiv:2006.04439)
- Lechner et al., 2022: Closed-form continuous-time (Nature MI)
- Hasani et al., 2022: Liquid Structural State-Space Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class LiquidConfig:
    """
    Configuration for liquid time-constant networks.

    Args:
        d_model: Dimensionality of hidden state
        tau_base: Base time constant (controls adaptation speed)
        tau_min: Minimum time constant (fast adaptation)
        tau_max: Maximum time constant (slow consolidation)
        use_closed_form: Use closed-form solution (vs Euler)
        bounded_output: Enforce output bounds for stability
        output_min: Minimum output value
        output_max: Maximum output value
    """
    d_model: int = 768
    tau_base: float = 1.0
    tau_min: float = 0.1
    tau_max: float = 10.0
    use_closed_form: bool = True
    bounded_output: bool = True
    output_min: float = -1.0
    output_max: float = 1.0


class LiquidTimeConstantCell(nn.Module):
    """
    Single liquid time-constant neuron with adaptive dynamics.

    Implements the core LTC equations:
        dx/dt = -[1/τ + f(x,I,t)] · x + f(x,I,t) · A

    Where:
        - τ: base time constant (learnable)
        - f(...): nonlinear gating function (context-dependent)
        - A: asymptotic stable point

    The effective time constant is:
        τ_sys = τ / [1 + τ · f(x,I,t)]

    Bounded between τ/(1+τW) and τ for stability.

    Example:
        >>> config = LiquidConfig(d_model=128)
        >>> cell = LiquidTimeConstantCell(config)
        >>>
        >>> x = torch.randn(2, 128)
        >>> h = torch.zeros(2, 128)
        >>> new_h = cell(x, h, dt=0.1)
    """

    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.config = config

        # Base time constant (learnable)
        self.tau = nn.Parameter(
            torch.ones(config.d_model) * config.tau_base
        )

        # Gating function f(x, h, t)
        self.gate_net = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Tanh(),
            nn.Linear(config.d_model, config.d_model),
            nn.Sigmoid()  # Ensures f >= 0
        )

        # Asymptotic stable point A
        self.stable_point = nn.Parameter(
            torch.zeros(config.d_model)
        )

    def compute_gate(
        self,
        x: torch.Tensor,
        h: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gating function f(x, h).

        Args:
            x: Input (batch, d_model)
            h: Hidden state (batch, d_model)

        Returns:
            Gate values (batch, d_model)
        """
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=-1)
        gate = self.gate_net(combined)
        return gate

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt: float = 0.1
    ) -> torch.Tensor:
        """
        Forward pass with liquid dynamics.

        Args:
            x: Input tensor (batch, d_model)
            h: Previous hidden state (batch, d_model)
            dt: Time step size

        Returns:
            New hidden state (batch, d_model)
        """
        # Compute gating function
        f = self.compute_gate(x, h)

        # Effective time constant
        # τ_sys = τ / (1 + τ · f)
        tau_clamp = torch.clamp(self.tau, self.config.tau_min,
                                self.config.tau_max)
        tau_effective = tau_clamp / (1.0 + tau_clamp * f)

        if self.config.use_closed_form:
            # Closed-form solution (stable for stiff ODEs)
            # h(t+dt) = [h(t) + dt·f·A] / [1 + dt(1/τ + f)]
            numerator = h + dt * f * self.stable_point
            denominator = 1.0 + dt * (1.0 / tau_effective + f)
            h_new = numerator / denominator
        else:
            # Forward Euler (less stable but faster)
            # dh/dt = -(1/τ + f)·h + f·A
            dhdt = -(1.0 / tau_effective + f) * h + f * self.stable_point
            h_new = h + dt * dhdt

        # Optional: Enforce bounds for stability
        if self.config.bounded_output:
            h_new = torch.clamp(
                h_new,
                self.config.output_min,
                self.config.output_max
            )

        return h_new

    def get_time_constants(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Get current effective time constants.

        Useful for:
        - Visualizing adaptation rates
        - Identifying fast vs slow neurons
        - Debugging temporal dynamics

        Args:
            x: Input
            h: Hidden state

        Returns:
            Effective time constants (batch, d_model)
        """
        f = self.compute_gate(x, h)
        tau_clamp = torch.clamp(self.tau, self.config.tau_min,
                                self.config.tau_max)
        tau_effective = tau_clamp / (1.0 + tau_clamp * f)
        return tau_effective


class LiquidLayer(nn.Module):
    """
    Liquid neural network layer.

    Stacks multiple liquid cells with input/output projections.
    Can be used as drop-in replacement for LSTM/GRU in sequence models.

    Example:
        >>> config = LiquidConfig(d_model=256)
        >>> layer = LiquidLayer(input_dim=128, hidden_dim=256, config=config)
        >>>
        >>> x = torch.randn(2, 10, 128)  # (batch, seq_len, input_dim)
        >>> h = layer(x, dt=0.1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        config: LiquidConfig
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Liquid cell
        cell_config = LiquidConfig(
            d_model=hidden_dim,
            tau_base=config.tau_base,
            tau_min=config.tau_min,
            tau_max=config.tau_max,
            use_closed_form=config.use_closed_form,
            bounded_output=config.bounded_output,
            output_min=config.output_min,
            output_max=config.output_max
        )
        self.cell = LiquidTimeConstantCell(cell_config)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        h_init: Optional[torch.Tensor] = None,
        dt: float = 0.1
    ) -> torch.Tensor:
        """
        Forward through sequence with liquid dynamics.

        Args:
            x: Input sequence (batch, seq_len, input_dim)
            h_init: Initial hidden state (batch, hidden_dim)
            dt: Time step size

        Returns:
            Hidden states (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden state
        if h_init is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            h = h_init

        outputs = []

        # Process sequence
        for t in range(seq_len):
            # Project input
            x_t = self.input_proj(x[:, t])

            # Liquid dynamics
            h = self.cell(x_t, h, dt=dt)

            # Project output
            out = self.output_proj(h)
            outputs.append(out)

        # Stack outputs
        return torch.stack(outputs, dim=1)

    def get_adaptation_rates(
        self,
        x: torch.Tensor,
        h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get adaptation rates for visualization.

        Returns:
            Tuple of (fast_neurons, slow_neurons) indices
        """
        tau_eff = self.cell.get_time_constants(x, h)

        # Mean across batch
        tau_mean = tau_eff.mean(dim=0)

        # Identify fast vs slow
        threshold = tau_mean.median()
        fast_neurons = (tau_mean < threshold).nonzero(as_tuple=True)[0]
        slow_neurons = (tau_mean >= threshold).nonzero(as_tuple=True)[0]

        return fast_neurons, slow_neurons


class LiquidMLP(nn.Module):
    """
    Multi-layer liquid network for non-temporal processing.

    Like standard MLP but with liquid dynamics between layers.
    Useful for adaptive feed-forward processing in transformers.

    Example:
        >>> config = LiquidConfig(d_model=256, tau_base=0.5)
        >>> mlp = LiquidMLP(d_model=256, d_ff=1024, config=config)
        >>>
        >>> x = torch.randn(2, 10, 256)
        >>> out = mlp(x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        config: LiquidConfig,
        num_steps: int = 5
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_steps = num_steps

        # Expand
        self.expand = nn.Linear(d_model, d_ff)

        # Liquid dynamics in middle
        cell_config = LiquidConfig(
            d_model=d_ff,
            tau_base=config.tau_base,
            tau_min=config.tau_min,
            tau_max=config.tau_max,
            use_closed_form=config.use_closed_form,
            bounded_output=config.bounded_output
        )
        self.liquid_cell = LiquidTimeConstantCell(cell_config)

        # Contract
        self.contract = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Forward with iterative liquid dynamics.

        Args:
            x: Input (batch, seq_len, d_model)
            dt: Time step size

        Returns:
            Output (batch, seq_len, d_model)
        """
        # Expand to FFN dimension
        h = self.expand(x)
        h = F.gelu(h)

        # Iterative liquid dynamics
        for _ in range(self.num_steps):
            h = self.liquid_cell(h, h, dt=dt)

        # Contract back to model dimension
        out = self.contract(h)

        return out


if __name__ == "__main__":
    print("Testing Liquid Time-Constant Networks...")

    # Configuration
    config = LiquidConfig(
        d_model=128,
        tau_base=1.0,
        tau_min=0.1,
        tau_max=10.0,
        use_closed_form=True
    )

    print(f"\nConfiguration:")
    print(f"  Hidden dimension: {config.d_model}")
    print(f"  Base time constant: {config.tau_base}")
    print(f"  Time constant range: [{config.tau_min}, {config.tau_max}]")
    print(f"  Closed-form solution: {config.use_closed_form}")

    # Test cell
    print("\nTesting Liquid Cell...")
    cell = LiquidTimeConstantCell(config)

    x = torch.randn(2, config.d_model)
    h = torch.zeros(2, config.d_model)

    print(f"  Input shape: {x.shape}")
    print(f"  Hidden shape: {h.shape}")

    # Forward pass
    for step in range(5):
        h = cell(x, h, dt=0.1)
        if step % 2 == 0:
            tau_eff = cell.get_time_constants(x, h)
            print(f"  Step {step}: τ_mean = {tau_eff.mean().item():.3f}, "
                  f"h_mean = {h.mean().item():.3f}")

    # Test layer
    print("\nTesting Liquid Layer...")
    layer = LiquidLayer(
        input_dim=64,
        hidden_dim=128,
        config=config
    )

    layer_params = sum(p.numel() for p in layer.parameters())
    print(f"  Parameters: {layer_params:,}")

    # Sequence processing
    x_seq = torch.randn(2, 10, 64)
    h_seq = layer(x_seq, dt=0.1)

    print(f"  Input sequence: {x_seq.shape}")
    print(f"  Output sequence: {h_seq.shape}")

    # Adaptation rates
    fast, slow = layer.get_adaptation_rates(
        layer.input_proj(x_seq[:, 0]),
        h_seq[:, 0]
    )
    print(f"  Fast neurons: {len(fast)} ({len(fast)/config.d_model*100:.1f}%)")
    print(f"  Slow neurons: {len(slow)} ({len(slow)/config.d_model*100:.1f}%)")

    # Test MLP
    print("\nTesting Liquid MLP...")
    mlp = LiquidMLP(
        d_model=128,
        d_ff=512,
        config=config,
        num_steps=5
    )

    mlp_params = sum(p.numel() for p in mlp.parameters())
    print(f"  Parameters: {mlp_params:,}")

    x_mlp = torch.randn(2, 10, 128)
    out_mlp = mlp(x_mlp, dt=0.1)

    print(f"  Input shape: {x_mlp.shape}")
    print(f"  Output shape: {out_mlp.shape}")

    # Test learning
    print("\nTesting gradient flow...")
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)

    for step in range(5):
        optimizer.zero_grad()
        out = mlp(x_mlp, dt=0.1)
        loss = out.pow(2).mean()
        loss.backward()
        optimizer.step()

        if step % 2 == 0:
            print(f"  Step {step}: Loss = {loss.item():.4f}")

    print("\n✓ Liquid Time-Constant Networks operational!")
    print("Implements: adaptive τ, closed-form dynamics, bounded stability")
