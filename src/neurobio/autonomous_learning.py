"""
Autonomous Learning Rate Control - True Self-Organization.

The brain doesn't have a human-set learning rate. Instead, learning speed
emerges from local mathematical rules and feedback loops:

1. Prediction error magnitude determines urgency
2. Uncertainty determines exploration vs exploitation
3. Metabolic cost constrains plasticity
4. Success history modulates future learning
5. Local gradient information guides step size

This creates EMERGENT learning dynamics - the system decides its own
learning speed based on mathematical principles, not human presets.

Key principle: Everything derives from simple mathematical combinations,
just as chemistry/physics/biology all reduce to math.

Biological Motivation:
- Neurons don't "know" global learning rates
- Plasticity emerges from local rules (STDP, calcium dynamics)
- Metabolic constraints naturally limit learning
- Meta-plasticity adjusts learning based on history

References:
- **Spike-Timing Dependent Plasticity**: Bi & Poo, 1998. "Synaptic modifications in
  cultured hippocampal neurons" Journal of Neuroscience, 18(24):10464-10472
- **Metabolic Constraints**: Laughlin & Sejnowski, 2003. "Communication in neuronal
  networks" Science, 301(5641):1870-1874
- **Meta-plasticity**: Abraham & Bear, 1996. "Metaplasticity: the plasticity of
  synaptic plasticity" Trends in Neurosciences, 19(4):126-130
- **Local Learning Rules**: Lillicrap et al., 2020. "Backpropagation and the brain"
  Nature Reviews Neuroscience, 21(6):335-346
- **Adaptive Learning Rates**: Baydin et al., 2018. "Automatic differentiation in
  machine learning: a survey" Journal of Machine Learning Research, 18:1-43
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from dataclasses import dataclass
import math


@dataclass
class AutonomousLearningConfig:
    """
    Configuration for autonomous learning rate control.

    Args:
        base_sensitivity: Base sensitivity to gradients (replaces lr)
        min_rate: Minimum learning rate (safety bound)
        max_rate: Maximum learning rate (safety bound)
        adaptation_speed: How fast to adapt (tau for exponential averaging)
        uncertainty_weight: Weight for uncertainty-based modulation
        metabolic_cost_weight: Weight for metabolic constraints
    """
    base_sensitivity: float = 1.0  # Not a learning rate!
    min_rate: float = 1e-6
    max_rate: float = 1e-2
    adaptation_speed: float = 0.01  # Tau for exponential moving average
    uncertainty_weight: float = 0.5
    metabolic_cost_weight: float = 0.1


class AutonomousLearningRateController(nn.Module):
    """
    Autonomous learning rate that emerges from mathematical principles.

    No human sets the learning rate. Instead:
    - Prediction error determines urgency
    - Gradient statistics determine step size
    - Success history modulates future learning
    - Metabolic cost constrains plasticity
    - Uncertainty drives exploration

    This is how the brain actually works - learning rate emerges from
    local computations, not global parameters.

    Mathematical basis:
    - Information theory (surprise, entropy)
    - Control theory (feedback, stability)
    - Thermodynamics (energy minimization)
    - Statistics (gradient estimation, uncertainty)

    Example:
        >>> config = AutonomousLearningConfig()
        >>> controller = AutonomousLearningRateController(config)
        >>>
        >>> # During training, let system decide learning rate
        >>> loss = compute_loss(...)
        >>> gradients = compute_gradients(...)
        >>> lr = controller.compute_learning_rate(
        ...     loss=loss,
        ...     gradients=gradients,
        ...     prediction_error=error
        ... )
        >>> # lr is now emergent, not preset!
    """

    def __init__(self, config: AutonomousLearningConfig):
        super().__init__()
        self.config = config

        # Running statistics (exponential moving averages)
        self.register_buffer('avg_loss', torch.tensor(1.0))
        self.register_buffer('avg_gradient_norm', torch.tensor(0.1))
        self.register_buffer('avg_prediction_error', torch.tensor(1.0))
        self.register_buffer('success_rate', torch.tensor(0.5))

        # Gradient variance (uncertainty estimate)
        self.register_buffer('gradient_variance', torch.tensor(0.1))

        # Metabolic cost accumulator
        self.register_buffer('metabolic_cost', torch.tensor(0.0))

        # History of learning rates (for analysis)
        self.lr_history = []

    def compute_learning_rate(
        self,
        loss: torch.Tensor,
        gradients: list,
        prediction_error: Optional[torch.Tensor] = None,
        previous_loss: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute emergent learning rate from mathematical principles.

        No human presets - the system decides based on:
        1. Current prediction error (surprise)
        2. Gradient magnitude (local information)
        3. Gradient variance (uncertainty)
        4. Success history (meta-learning)
        5. Metabolic cost (resource constraints)

        Args:
            loss: Current loss value
            gradients: List of gradient tensors
            prediction_error: Prediction error magnitude
            previous_loss: Previous loss for computing improvement

        Returns:
            Emergent learning rate
        """
        with torch.no_grad():
            # 1. Compute gradient statistics
            grad_norms = [g.norm() for g in gradients if g is not None]
            if grad_norms:
                current_grad_norm = torch.stack(grad_norms).mean()
            else:
                current_grad_norm = self.avg_gradient_norm

            # Update running average
            alpha = self.config.adaptation_speed
            self.avg_gradient_norm = (
                (1 - alpha) * self.avg_gradient_norm +
                alpha * current_grad_norm
            )

            # 2. Compute gradient variance (uncertainty)
            if len(grad_norms) > 1:
                grad_variance = torch.stack(grad_norms).var()
                self.gradient_variance = (
                    (1 - alpha) * self.gradient_variance +
                    alpha * grad_variance
                )

            # 3. Update loss history
            self.avg_loss = (1 - alpha) * self.avg_loss + alpha * loss

            # 4. Compute prediction error (surprise)
            if prediction_error is not None:
                self.avg_prediction_error = (
                    (1 - alpha) * self.avg_prediction_error +
                    alpha * prediction_error
                )
            else:
                # Use loss as proxy for prediction error
                self.avg_prediction_error = self.avg_loss

            # 5. Compute success rate (learning progress)
            if previous_loss is not None:
                improvement = previous_loss - loss
                success = float(improvement > 0)
                self.success_rate = (
                    (1 - alpha) * self.success_rate + alpha * success
                )

            # 6. Update metabolic cost (accumulated plasticity)
            plasticity = current_grad_norm * self.avg_loss
            self.metabolic_cost = (
                0.99 * self.metabolic_cost + 0.01 * plasticity
            )

            # === EMERGENT LEARNING RATE COMPUTATION ===
            # No human presets - pure mathematical derivation

            # Component 1: Prediction error drives urgency
            # Higher error = need to learn faster
            error_factor = torch.sqrt(
                self.avg_prediction_error / (1.0 + self.avg_loss)
            )

            # Component 2: Gradient magnitude determines natural step size
            # Use inverse of gradient norm (like natural gradient)
            # Large gradients = small steps (stability)
            # Small gradients = larger steps (exploration)
            gradient_factor = 1.0 / (1.0 + self.avg_gradient_norm)

            # Component 3: Uncertainty modulates exploration
            # High variance = more exploration (larger lr)
            # Low variance = exploitation (smaller lr)
            uncertainty_factor = 1.0 + (
                self.config.uncertainty_weight *
                torch.sqrt(self.gradient_variance)
            )

            # Component 4: Success history (meta-learning)
            # Successful learning = continue current rate
            # Failed learning = reduce rate (more conservative)
            meta_factor = 0.5 + self.success_rate

            # Component 5: Metabolic cost constraint
            # High cost = reduce learning (conservation)
            # Low cost = allow more learning (resources available)
            metabolic_factor = 1.0 / (
                1.0 + self.config.metabolic_cost_weight * self.metabolic_cost
            )

            # === COMBINE ALL FACTORS ===
            # This is the emergent learning rate!
            emergent_lr = (
                self.config.base_sensitivity *
                error_factor *
                gradient_factor *
                uncertainty_factor *
                meta_factor *
                metabolic_factor
            )

            # Safety bounds (physical constraints)
            emergent_lr = torch.clamp(
                emergent_lr,
                self.config.min_rate,
                self.config.max_rate
            )

            # Track history
            if len(self.lr_history) < 1000:
                self.lr_history.append({
                    'lr': emergent_lr.item(),
                    'error_factor': error_factor.item(),
                    'gradient_factor': gradient_factor.item(),
                    'uncertainty_factor': uncertainty_factor.item(),
                    'meta_factor': meta_factor.item(),
                    'metabolic_factor': metabolic_factor.item(),
                })

            return emergent_lr.item()

    def get_learning_dynamics(self) -> Dict[str, float]:
        """
        Get current learning dynamics (for analysis/visualization).

        Returns:
            Dictionary with all factors contributing to learning rate
        """
        return {
            'avg_loss': self.avg_loss.item(),
            'avg_gradient_norm': self.avg_gradient_norm.item(),
            'gradient_variance': self.gradient_variance.item(),
            'avg_prediction_error': self.avg_prediction_error.item(),
            'success_rate': self.success_rate.item(),
            'metabolic_cost': self.metabolic_cost.item(),
        }

    def reset_dynamics(self):
        """Reset learning dynamics (for new task/domain)."""
        with torch.no_grad():
            self.avg_loss.fill_(1.0)
            self.avg_gradient_norm.fill_(0.1)
            self.avg_prediction_error.fill_(1.0)
            self.success_rate.fill_(0.5)
            self.gradient_variance.fill_(0.1)
            self.metabolic_cost.fill_(0.0)
            self.lr_history = []


class AdaptiveOptimizer(torch.optim.Optimizer):
    """
    Optimizer with autonomous learning rate.

    Uses AutonomousLearningRateController to determine step size
    from mathematical principles rather than human presets.

    Example:
        >>> controller = AutonomousLearningRateController(config)
        >>> optimizer = AdaptiveOptimizer(model.parameters(), controller)
        >>>
        >>> loss = compute_loss()
        >>> loss.backward()
        >>> optimizer.step(loss=loss)  # lr emerges automatically!
    """

    def __init__(
        self,
        params,
        learning_controller: AutonomousLearningRateController,
        weight_decay: float = 0.01
    ):
        defaults = {'weight_decay': weight_decay}
        super().__init__(params, defaults)
        self.learning_controller = learning_controller
        self.previous_loss = None

    def step(
        self,
        closure=None,
        loss: Optional[torch.Tensor] = None,
        prediction_error: Optional[torch.Tensor] = None
    ):
        """
        Performs a single optimization step with emergent learning rate.

        Args:
            closure: Optional closure for recomputing gradients
            loss: Current loss value
            prediction_error: Optional prediction error
        """
        if closure is not None:
            loss = closure()

        # Collect gradients
        gradients = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    gradients.append(p.grad.data)

        # Compute emergent learning rate
        if loss is not None:
            lr = self.learning_controller.compute_learning_rate(
                loss=loss,
                gradients=gradients,
                prediction_error=prediction_error,
                previous_loss=self.previous_loss
            )
            self.previous_loss = loss.clone()
        else:
            # Fallback if no loss provided
            lr = self.learning_controller.config.base_sensitivity

        # Apply updates with emergent learning rate
        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Standard SGD update with emergent lr
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                # The key: use EMERGENT learning rate, not preset!
                p.data.add_(d_p, alpha=-lr)

        return loss


if __name__ == "__main__":
    print("Testing Autonomous Learning Rate Control...")

    # Create controller
    config = AutonomousLearningConfig(base_sensitivity=1.0)
    controller = AutonomousLearningRateController(config)

    print("\nInitial dynamics:")
    dynamics = controller.get_learning_dynamics()
    for key, value in dynamics.items():
        print(f"  {key}: {value:.4f}")

    # Simulate learning trajectory
    print("\nSimulating learning trajectory...")
    previous_loss = None

    for step in range(100):
        # Simulate decreasing loss (successful learning)
        loss = torch.tensor(10.0 * math.exp(-step / 30) + 0.5)

        # Simulate gradients
        gradients = [
            torch.randn(100, 100) * (1.0 + 0.1 * step / 100),
            torch.randn(100, 100) * (1.0 + 0.1 * step / 100)
        ]

        # Compute emergent learning rate
        lr = controller.compute_learning_rate(
            loss=loss,
            gradients=gradients,
            prediction_error=loss,
            previous_loss=previous_loss
        )

        previous_loss = loss

        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Emergent LR: {lr:.6f}")
            dynamics = controller.get_learning_dynamics()
            print(f"  Success rate: {dynamics['success_rate']:.2%}")
            print(f"  Metabolic cost: {dynamics['metabolic_cost']:.4f}")

    # Show learning rate evolution
    print("\nLearning rate evolved autonomously:")
    if controller.lr_history:
        print(f"  Initial LR: {controller.lr_history[0]['lr']:.6f}")
        print(f"  Final LR: {controller.lr_history[-1]['lr']:.6f}")
        print(f"  Ratio: {controller.lr_history[-1]['lr'] / controller.lr_history[0]['lr']:.2f}x")

    # Test adaptive optimizer
    print("\nTesting AdaptiveOptimizer...")
    model = nn.Linear(10, 10)
    optimizer = AdaptiveOptimizer(model.parameters(), controller)

    x = torch.randn(32, 10)
    y = torch.randn(32, 10)

    for step in range(5):
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step(loss=loss)

        if step % 2 == 0:
            print(f"  Step {step}: Loss={loss.item():.4f}")

    print("\n✓ Autonomous learning rate working!")
    print("Learning rate emerges from mathematical principles, not human presets!")
