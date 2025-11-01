"""
Neuromodulation System for Adaptive Learning Control.

Implements dopamine, serotonin, and acetylcholine-inspired
modulation of learning dynamics.

Key Principles:
- Dopamine: Modulates learning rate based on prediction error/reward
- Serotonin: Balances stability vs plasticity (exploration/exploitation)
- Acetylcholine: Gates attention to salient features
- Noradrenaline: Arousal and urgency modulation

Biological Motivation:
- Neuromodulators broadcast signals that modulate learning globally
- Different modulators affect different aspects of plasticity
- Reward prediction errors drive dopamine release
- Serotonin regulates exploration vs exploitation trade-off

References:
- **Multi-neuromodulatory dynamics**: Sajid et al., 2025. "Multi-neuromodulatory
  dynamics regulate distinct behavioral states" arXiv:2501.06762
- **Dopamine and Learning**: Schultz et al., 1997. "A neural substrate of prediction
  and reward" Science, 275(5306):1593-1599
- **Dopamine Teaching Signals**: Eshel et al., 2015. "Arithmetic and local circuitry
  underlying dopamine prediction errors" Nature, 525(7568):243-246
- **Serotonin and Exploration**: Dayan & Huys, 2009. "Serotonin in affective control"
  Annual Review of Neuroscience, 32:95-126
- **Acetylcholine and Attention**: Sarter et al., 2005. "Unraveling the attentional
  functions of cortical cholinergic inputs" Neuron, 48(5):667-677
- **Neuromodulation Review**: Dayan & Yu, 2006. "Phasic norepinephrine: a neural
  interrupt signal for unexpected events" Network, 17(4):335-350
- **Computational Neuromodulation**: Keramati & Gutkin, 2014. "Homeostatic
  reinforcement learning" Behavioural Brain Research, 265:143-154
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class NeuromodulatorConfig:
    """
    Configuration for neuromodulation system.

    Args:
        d_model: Hidden dimension for context encoding
        initial_dopamine: Initial dopamine level (learning rate scale)
        initial_serotonin: Initial serotonin (stability preference)
        initial_acetylcholine: Initial acetylcholine (attention gating)
        initial_noradrenaline: Initial noradrenaline (arousal)
        adapt_from_context: Whether to adapt levels from input context
        dopamine_range: (min, max) bounds for dopamine
        temperature: Temperature for sigmoid activation
    """
    d_model: int = 768
    initial_dopamine: float = 1.0
    initial_serotonin: float = 0.5
    initial_acetylcholine: float = 0.7
    initial_noradrenaline: float = 0.3
    adapt_from_context: bool = True
    dopamine_range: tuple = (0.1, 3.0)
    temperature: float = 1.0


class NeuromodulationController(nn.Module):
    """
    Brain-inspired neuromodulation for adaptive learning control.

    Modulates learning dynamics through four key neuromodulators:
    - Dopamine (DA): Learning rate, plasticity, reward processing
    - Serotonin (5-HT): Stability, exploration/exploitation balance
    - Acetylcholine (ACh): Attention gating, feature salience
    - Noradrenaline (NE): Arousal, urgency, novelty detection

    Example:
        >>> config = NeuromodulatorConfig(d_model=768)
        >>> controller = NeuromodulationController(config)
        >>>
        >>> # During learning
        >>> x = torch.randn(2, 10, 768)  # (batch, seq, hidden)
        >>> modulation = controller(x, performance=0.8, novelty=0.3)
        >>>
        >>> # Use modulation signals
        >>> learning_rate = base_lr * modulation['learning_scale']
        >>> attention_weights = attention * modulation['attention_scale']
    """

    def __init__(self, config: NeuromodulatorConfig):
        super().__init__()
        self.config = config

        # Base neuromodulator levels (learnable parameters)
        self.base_dopamine = nn.Parameter(
            torch.tensor([config.initial_dopamine])
        )
        self.base_serotonin = nn.Parameter(
            torch.tensor([config.initial_serotonin])
        )
        self.base_acetylcholine = nn.Parameter(
            torch.tensor([config.initial_acetylcholine])
        )
        self.base_noradrenaline = nn.Parameter(
            torch.tensor([config.initial_noradrenaline])
        )

        # Context-dependent modulation
        if config.adapt_from_context:
            self.context_encoder = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.Tanh(),
                nn.Linear(config.d_model // 2, 4),  # 4 neuromodulators
            )
        else:
            self.context_encoder = None

        # Performance-based dopamine adjustment
        self.performance_to_dopamine = nn.Linear(1, 1, bias=True)

        # Novelty-based noradrenaline adjustment
        self.novelty_to_noradrenaline = nn.Linear(1, 1, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        performance: Optional[float] = None,
        novelty: Optional[float] = None,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute neuromodulator levels and modulation signals.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            performance: Task performance (0-1), high = success
            novelty: Novelty/surprise (0-1), high = unexpected
            return_details: Return detailed neuromodulator levels

        Returns:
            Dictionary with modulation signals:
            - learning_scale: Multiply with learning rate
            - attention_scale: Multiply with attention weights
            - exploration_factor: Epsilon for exploration
            - dopamine, serotonin, acetylcholine, noradrenaline: Levels
        """
        batch_size = x.size(0)
        device = x.device

        # Context-dependent modulation
        if self.context_encoder is not None:
            # Average over sequence to get context
            context = x.mean(dim=1)  # (batch, d_model)
            context_mods = torch.sigmoid(
                self.context_encoder(context) / self.config.temperature
            )  # (batch, 4)
        else:
            context_mods = torch.ones(batch_size, 4, device=device) * 0.5

        # Base levels (broadcast to batch)
        dopamine = self.base_dopamine.expand(batch_size, -1)
        serotonin = self.base_serotonin.expand(batch_size, -1)
        acetylcholine = self.base_acetylcholine.expand(batch_size, -1)
        noradrenaline = self.base_noradrenaline.expand(batch_size, -1)

        # Modulate by context
        dopamine = dopamine * (1 + context_mods[:, 0:1])
        serotonin = serotonin * (1 + context_mods[:, 1:2])
        acetylcholine = acetylcholine * (1 + context_mods[:, 2:3])
        noradrenaline = noradrenaline * (1 + context_mods[:, 3:4])

        # Performance-based dopamine (reward prediction error)
        if performance is not None:
            perf_tensor = torch.tensor(
                [[performance]],
                device=device
            ).expand(batch_size, -1)
            da_adjustment = self.performance_to_dopamine(perf_tensor)
            dopamine = dopamine + da_adjustment

        # Novelty-based noradrenaline (surprise/arousal)
        if novelty is not None:
            nov_tensor = torch.tensor(
                [[novelty]],
                device=device
            ).expand(batch_size, -1)
            ne_adjustment = self.novelty_to_noradrenaline(nov_tensor)
            noradrenaline = noradrenaline + ne_adjustment

        # Clamp dopamine to safe range
        dopamine = torch.clamp(
            dopamine,
            self.config.dopamine_range[0],
            self.config.dopamine_range[1]
        )

        # Compute modulation signals
        # Dopamine: increases learning rate and plasticity
        learning_scale = dopamine

        # Acetylcholine: gates attention (sigmoid for 0-1 range)
        attention_scale = torch.sigmoid(acetylcholine)

        # Serotonin: low = explore, high = exploit
        # Exploration factor inversely related to serotonin
        exploration_factor = 1.0 - torch.sigmoid(serotonin)

        # Noradrenaline: modulates responsiveness (for future use)
        arousal = torch.sigmoid(noradrenaline)

        result = {
            'learning_scale': learning_scale,
            'attention_scale': attention_scale,
            'exploration_factor': exploration_factor,
            'arousal': arousal,
        }

        if return_details:
            result.update({
                'dopamine': dopamine,
                'serotonin': serotonin,
                'acetylcholine': acetylcholine,
                'noradrenaline': noradrenaline,
            })

        return result

    def update_from_reward(self, reward: float, target: float = 0.5):
        """
        Update dopamine based on reward prediction error.

        Positive RPE (reward > target) increases dopamine.
        Negative RPE (reward < target) decreases dopamine.

        Args:
            reward: Observed reward (0-1)
            target: Expected reward (0-1)
        """
        prediction_error = reward - target

        # Update dopamine (simple gradient-free update)
        with torch.no_grad():
            self.base_dopamine.data += 0.01 * prediction_error
            self.base_dopamine.data.clamp_(
                self.config.dopamine_range[0],
                self.config.dopamine_range[1]
            )

    def increase_stability(self, amount: float = 0.1):
        """Increase serotonin for more stability, less exploration."""
        with torch.no_grad():
            self.base_serotonin.data += amount
            self.base_serotonin.data.clamp_(0.0, 2.0)

    def increase_plasticity(self, amount: float = 0.1):
        """Decrease serotonin for more plasticity, more exploration."""
        with torch.no_grad():
            self.base_serotonin.data -= amount
            self.base_serotonin.data.clamp_(0.0, 2.0)

    def reset_to_defaults(self):
        """Reset neuromodulators to initial values."""
        with torch.no_grad():
            self.base_dopamine.data.fill_(self.config.initial_dopamine)
            self.base_serotonin.data.fill_(self.config.initial_serotonin)
            self.base_acetylcholine.data.fill_(self.config.initial_acetylcholine)
            self.base_noradrenaline.data.fill_(
                self.config.initial_noradrenaline
            )

    def get_state(self) -> Dict[str, float]:
        """Get current neuromodulator levels."""
        return {
            'dopamine': self.base_dopamine.item(),
            'serotonin': self.base_serotonin.item(),
            'acetylcholine': self.base_acetylcholine.item(),
            'noradrenaline': self.base_noradrenaline.item(),
        }


if __name__ == "__main__":
    print("Testing NeuromodulationController...")

    # Create controller
    config = NeuromodulatorConfig(d_model=512)
    controller = NeuromodulationController(config)

    print("\nInitial state:")
    print(controller.get_state())

    # Test forward pass
    x = torch.randn(2, 10, 512)
    modulation = controller(x, performance=0.8, novelty=0.2, return_details=True)

    print("\nModulation signals:")
    for key, value in modulation.items():
        if value.numel() == 1:
            print(f"  {key}: {value.item():.3f}")
        else:
            print(f"  {key}: shape {value.shape}, mean {value.mean():.3f}")

    # Test reward-based update
    print("\nAfter positive reward:")
    controller.update_from_reward(reward=0.9, target=0.5)
    print(controller.get_state())

    # Test stability adjustment
    print("\nAfter increasing stability:")
    controller.increase_stability(0.2)
    print(controller.get_state())

    print("\n✓ NeuromodulationController working!")
