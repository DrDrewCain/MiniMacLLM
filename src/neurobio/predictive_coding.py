"""
Predictive Coding - Hierarchical Prediction Errors for Continual Learning.

Implements brain-inspired predictive coding based on:
- Top-down predictions from higher layers
- Bottom-up error propagation
- Iterative inference to minimize free energy
- Local weight updates

Key principles:
- Brain is a hierarchical prediction machine
- Learning minimizes prediction errors at all levels
- Errors propagate bottom-up, predictions top-down
- Natural credit assignment through prediction errors

References:
- Rao & Ballard, 1999: Predictive coding in visual cortex
- Friston, 2005: Free energy principle
- Whittington & Bogacz, 2017: Approximation of backprop
- Song et al., 2024: PRECO library (arXiv:2407.04117)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class PredictiveCodingConfig:
    """
    Configuration for predictive coding networks.

    Args:
        d_model: Dimensionality of representations
        num_inference_steps: Number of iterative inference steps
        inference_lr: Learning rate for inference (updating hidden states)
        energy_tolerance: Convergence threshold for free energy
        use_gain_modulation: Apply gain modulation to errors (f' ⊙ ε)
        detach_predictions: Detach predictions during inference
    """
    d_model: int = 768
    num_inference_steps: int = 5
    inference_lr: float = 0.1
    energy_tolerance: float = 1e-4
    use_gain_modulation: bool = True
    detach_predictions: bool = True


class PredictiveCodingLayer(nn.Module):
    """
    Single predictive coding layer with top-down predictions.

    Implements the core PC dynamics:
    1. Compute prediction from higher layer: x̂ = f(W·x_higher)
    2. Compute prediction error: ε = x - x̂
    3. Update hidden state to minimize error
    4. Propagate error upward for learning

    Example:
        >>> config = PredictiveCodingConfig(d_model=768)
        >>> pc_layer = PredictiveCodingLayer(config)
        >>>
        >>> # Forward pass with inference
        >>> x_lower = torch.randn(2, 10, 768)
        >>> x_higher = torch.randn(2, 10, 768)
        >>> output, error = pc_layer(x_lower, x_higher)
    """

    def __init__(self, config: PredictiveCodingConfig):
        super().__init__()
        self.config = config

        # Top-down prediction pathway
        self.prediction_net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )

        # Error neurons (separate from value neurons)
        self.register_buffer(
            'error_state',
            torch.zeros(1, 1, config.d_model)
        )

        # Gain modulation (like f' in PC equations)
        if config.use_gain_modulation:
            self.gain_net = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.Sigmoid()
            )
        else:
            self.gain_net = None

    def predict(self, x_higher: torch.Tensor) -> torch.Tensor:
        """
        Top-down prediction from higher layer.

        Args:
            x_higher: Higher layer representation (batch, seq_len, d_model)

        Returns:
            Prediction of current layer (batch, seq_len, d_model)
        """
        return self.prediction_net(x_higher)

    def compute_error(
        self,
        x_current: torch.Tensor,
        x_higher: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute prediction error.

        Args:
            x_current: Current layer representation
            x_higher: Higher layer representation (optional)

        Returns:
            Prediction error (batch, seq_len, d_model)
        """
        if x_higher is None:
            # No higher layer (top of hierarchy)
            return torch.zeros_like(x_current)

        # Top-down prediction
        if self.config.detach_predictions:
            prediction = self.predict(x_higher.detach())
        else:
            prediction = self.predict(x_higher)

        # Prediction error
        error = x_current - prediction

        # Optional: Apply gain modulation
        if self.gain_net is not None:
            gain = self.gain_net(x_current)
            error = error * gain

        return error

    def forward(
        self,
        x_lower: torch.Tensor,
        x_higher: Optional[torch.Tensor] = None,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with iterative inference.

        Args:
            x_lower: Bottom-up input from lower layer
            x_higher: Top-down input from higher layer (optional)
            return_trajectory: Return full inference trajectory

        Returns:
            Tuple of (converged_state, final_error)
            If return_trajectory: (states_trajectory, errors_trajectory)
        """
        batch_size, seq_len, d_model = x_lower.shape

        # Initialize hidden state with bottom-up input
        x_current = x_lower.clone()

        if return_trajectory:
            trajectory = [x_current.clone()]
            error_trajectory = []

        # Iterative inference to minimize prediction error
        for step in range(self.config.num_inference_steps):
            # Compute prediction error
            error = self.compute_error(x_current, x_higher)

            if return_trajectory:
                error_trajectory.append(error.clone())

            # Update hidden state to reduce error (gradient descent on free energy)
            if x_higher is not None:
                x_current = x_current - self.config.inference_lr * error

            # Check convergence
            if error.abs().mean() < self.config.energy_tolerance:
                break

            if return_trajectory:
                trajectory.append(x_current.clone())

        # Final error for learning
        final_error = self.compute_error(x_current, x_higher)

        if return_trajectory:
            return torch.stack(trajectory), torch.stack(error_trajectory)
        else:
            return x_current, final_error


class PredictiveCodingNetwork(nn.Module):
    """
    Multi-layer predictive coding network.

    Implements full hierarchical predictive coding:
    - Multiple layers with top-down predictions
    - Bidirectional error propagation
    - Iterative inference across all layers
    - Can be used as drop-in replacement for feedforward layers

    Example:
        >>> config = PredictiveCodingConfig(d_model=768)
        >>> pcn = PredictiveCodingNetwork(config, num_layers=3)
        >>>
        >>> # Process input through hierarchy
        >>> x = torch.randn(2, 10, 768)
        >>> output, errors = pcn(x)
        >>>
        >>> # Total free energy
        >>> free_energy = sum(e.pow(2).sum() for e in errors)
    """

    def __init__(
        self,
        config: PredictiveCodingConfig,
        num_layers: int = 3
    ):
        super().__init__()
        self.config = config
        self.num_layers = num_layers

        # Stack of predictive coding layers
        self.layers = nn.ModuleList([
            PredictiveCodingLayer(config)
            for _ in range(num_layers)
        ])

        # Bottom-up processing (value neurons)
        self.bottom_up = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        return_all_errors: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through predictive coding hierarchy.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            return_all_errors: Return errors from all layers

        Returns:
            Tuple of (final_output, prediction_errors)
        """
        # Initialize hidden states for all layers
        hidden_states = [x]
        for layer_idx in range(self.num_layers):
            h = self.bottom_up[layer_idx](hidden_states[-1])
            hidden_states.append(F.gelu(h))

        # Iterative inference across all layers
        errors = [None] * self.num_layers

        for step in range(self.config.num_inference_steps):
            new_hidden_states = [hidden_states[0]]  # Keep input fixed

            # Update each layer based on prediction error
            for layer_idx in range(self.num_layers):
                x_lower = new_hidden_states[layer_idx]
                x_higher = hidden_states[layer_idx + 1] if layer_idx < self.num_layers - 1 else None

                # PC layer updates hidden state
                h_updated, error = self.layers[layer_idx](
                    self.bottom_up[layer_idx](x_lower),
                    x_higher
                )

                new_hidden_states.append(F.gelu(h_updated))
                errors[layer_idx] = error

            hidden_states = new_hidden_states

            # Check convergence (total free energy)
            total_error = sum(e.abs().mean() for e in errors if e is not None)
            if total_error < self.config.energy_tolerance:
                break

        # Return final output and errors
        output = hidden_states[-1]

        if return_all_errors:
            return output, errors
        else:
            # Return only non-None errors
            valid_errors = [e for e in errors if e is not None]
            return output, valid_errors


def compute_free_energy(
    prediction_errors: List[torch.Tensor],
    complexity_weight: float = 0.01
) -> torch.Tensor:
    """
    Compute total free energy from prediction errors.

    Free Energy = Σ ||ε_i||² + λ·Complexity

    Args:
        prediction_errors: List of prediction errors per layer
        complexity_weight: Weight for model complexity term

    Returns:
        Total free energy (scalar)
    """
    # Accuracy term: sum of squared errors
    accuracy = sum(e.pow(2).sum() for e in prediction_errors)

    # Complexity term (can be extended to include KL divergence)
    complexity = torch.tensor(0.0, device=accuracy.device)

    return accuracy + complexity_weight * complexity


if __name__ == "__main__":
    print("Testing Predictive Coding System...")

    # Create predictive coding network
    config = PredictiveCodingConfig(
        d_model=128,
        num_inference_steps=10,
        inference_lr=0.1
    )
    pcn = PredictiveCodingNetwork(config, num_layers=3)

    print(f"\nConfiguration:")
    print(f"  Layers: {pcn.num_layers}")
    print(f"  Dimensionality: {config.d_model}")
    print(f"  Inference steps: {config.num_inference_steps}")
    print(f"  Inference LR: {config.inference_lr}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.d_model)

    output, errors = pcn(x, return_all_errors=True)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of error signals: {len(errors)}")

    # Compute free energy
    free_energy = compute_free_energy(errors)
    print(f"\nFree Energy: {free_energy.item():.4f}")

    # Show error magnitudes per layer
    print("\nPrediction errors per layer:")
    for i, error in enumerate(errors):
        if error is not None:
            print(f"  Layer {i}: {error.abs().mean().item():.6f}")

    # Test learning (backprop through PC)
    print("\nTesting backpropagation...")
    optimizer = torch.optim.Adam(pcn.parameters(), lr=0.001)

    initial_energy = free_energy.item()

    for step in range(5):
        optimizer.zero_grad()
        output, errors = pcn(x, return_all_errors=True)
        loss = compute_free_energy(errors)
        loss.backward()
        optimizer.step()

        if step % 2 == 0:
            print(f"  Step {step}: Free energy = {loss.item():.4f}")

    final_energy = loss.item()
    print(f"\nEnergy reduction: {initial_energy:.4f} → {final_energy:.4f}")
    print(f"Improvement: {(1 - final_energy/initial_energy)*100:.1f}%")

    print("\n✓ Predictive Coding system operational!")
    print("Implements: hierarchical predictions, error minimization, iterative inference")
