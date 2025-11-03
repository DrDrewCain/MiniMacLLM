"""
Eligibility Propagation (e-prop) Optimizer for Continual Learning.

Based on research:
- Bellec et al. (2020): "A solution to the learning dilemma for recurrent networks"
- Three-factor learning rule: eligibility trace × learning signal × neuromodulator

Perfect for continual learning because:
1. Local updates (no full backprop needed)
2. Credit assignment via eligibility traces
3. Compatible with frozen base models + adapters
4. Biologically plausible

Reference: https://www.nature.com/articles/s41467-020-17236-y
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Dict, List


class EPropOptimizer(Optimizer):
    """
    Eligibility Propagation optimizer for continual learning.

    Implements 3-factor learning rule:
        Δw = η × e(t) × L(t) × M(t)
    where:
        η = learning rate (from autonomous controller)
        e(t) = eligibility trace (local synaptic memory)
        L(t) = learning signal (error)
        M(t) = neuromodulator (global feedback)

    This enables:
    - Local learning without full backprop
    - Credit assignment over time
    - Continual learning without catastrophic forgetting

    Args:
        params: Model parameters to optimize
        learning_controller: Autonomous learning rate controller
        trace_decay: Eligibility trace decay (τ), default 0.95
        trace_mode: "post" (postsynaptic) or "pre" (presynaptic)
        beta1: Exponential decay for learning signal, default 0.9
        beta2: Exponential decay for neuromodulator, default 0.999
        eps: Numerical stability, default 1e-8
        weight_decay: L2 regularization, default 0.01
    """

    def __init__(
        self,
        params,
        learning_controller,
        trace_decay: float = 0.95,
        trace_mode: str = "post",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        if not 0.0 <= trace_decay < 1.0:
            raise ValueError(f"Invalid trace_decay: {trace_decay}")
        if trace_mode not in ["post", "pre"]:
            raise ValueError(f"trace_mode must be 'post' or 'pre', got {trace_mode}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")

        defaults = {
            'trace_decay': trace_decay,
            'trace_mode': trace_mode,
            'beta1': beta1,
            'beta2': beta2,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)

        self.learning_controller = learning_controller
        self.previous_loss = None
        self.global_step = 0

    def _init_state(self, param):
        """Initialize optimizer state for a parameter."""
        state = self.state[param]

        # Eligibility trace (synaptic memory)
        state['eligibility'] = torch.zeros_like(param.data)

        # Learning signal (error momentum)
        state['m'] = torch.zeros_like(param.data)

        # Neuromodulator (adaptive scaling)
        state['v'] = torch.zeros_like(param.data)

        # For debugging/analysis
        state['trace_magnitude'] = 0.0

    def step(
        self,
        closure=None,
        loss: Optional[torch.Tensor] = None,
        prediction_error: Optional[torch.Tensor] = None,
        neuromodulator: Optional[torch.Tensor] = None
    ):
        """
        Perform eligibility propagation update.

        Args:
            closure: Optional closure for recomputing loss
            loss: Current loss (required for autonomous LR)
            prediction_error: Optional prediction error (neuromodulator signal)
            neuromodulator: Optional explicit neuromodulator (e.g., reward)

        Returns:
            loss value
        """
        if closure is not None:
            loss = closure()

        # Collect gradients for autonomous LR
        gradients = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    gradients.append(p.grad.data)

        # Get emergent learning rate from autonomous controller
        if loss is not None:
            # On first step, use current loss as previous to bootstrap
            prev_loss = self.previous_loss if self.previous_loss is not None else loss

            base_lr = self.learning_controller.compute_learning_rate(
                loss=loss,
                gradients=gradients,
                prediction_error=prediction_error,
                previous_loss=prev_loss
            )
        else:
            # Fallback to base sensitivity if no loss provided
            base_lr = self.learning_controller.config.base_sensitivity

        # Compute global neuromodulator if not provided
        # IMPORTANT: Do this BEFORE updating self.previous_loss to get correct improvement
        if neuromodulator is None and loss is not None and self.previous_loss is not None:
            # Use loss improvement as neuromodulator (reward signal)
            improvement = self.previous_loss - loss
            neuromodulator = torch.sigmoid(improvement)  # Normalize to [0,1]
        elif neuromodulator is None:
            neuromodulator = torch.tensor(1.0)

        # Update previous loss AFTER computing neuromodulator
        if loss is not None:
            self.previous_loss = loss.detach()

        self.global_step += 1

        # Apply e-prop update to each parameter
        for group in self.param_groups:
            trace_decay = group['trace_decay']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Initialize state if needed
                if len(self.state[p]) == 0:
                    self._init_state(p)

                state = self.state[p]
                grad = p.grad.data

                # 1. Update eligibility trace (Factor 1: e(t))
                # This is the "synaptic memory" that tracks recent activity
                state['eligibility'] = (
                    trace_decay * state['eligibility'] +
                    (1 - trace_decay) * grad
                )

                # Track trace magnitude for analysis
                state['trace_magnitude'] = state['eligibility'].abs().mean().item()

                # 2. Update learning signal (Factor 2: L(t))
                # Momentum-like accumulation of gradients
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad

                # Bias correction
                m_hat = state['m'] / (1 - beta1 ** self.global_step)

                # 3. Update neuromodulator scaling (Factor 3: M(t))
                # Adaptive per-parameter scaling (like Adam's second moment)
                state['v'] = beta2 * state['v'] + (1 - beta2) * (grad ** 2)

                # Bias correction
                v_hat = state['v'] / (1 - beta2 ** self.global_step)

                # Adaptive scaling factor
                modulation_scale = 1.0 / (torch.sqrt(v_hat) + eps)

                # 4. Three-factor learning rule
                # Δw = η × e(t) × L(t) × M(t)
                effective_update = (
                    state['eligibility'] *      # Factor 1: Eligibility trace
                    m_hat *                      # Factor 2: Learning signal
                    modulation_scale *           # Factor 3a: Adaptive scaling
                    neuromodulator              # Factor 3b: Global modulator
                )

                # Weight decay
                if weight_decay != 0:
                    effective_update = effective_update.add(p.data, alpha=weight_decay)

                # Apply update with emergent learning rate
                p.data.add_(effective_update, alpha=-base_lr)

        return loss

    def get_trace_statistics(self) -> Dict[str, float]:
        """
        Get statistics about eligibility traces for debugging.

        Returns:
            Dictionary with trace statistics
        """
        trace_mags = []
        for group in self.param_groups:
            for p in group['params']:
                if len(self.state[p]) > 0:
                    trace_mags.append(self.state[p]['trace_magnitude'])

        if not trace_mags:
            return {'mean_trace': 0.0, 'max_trace': 0.0, 'min_trace': 0.0}

        return {
            'mean_trace': sum(trace_mags) / len(trace_mags),
            'max_trace': max(trace_mags),
            'min_trace': min(trace_mags),
            'num_params': len(trace_mags)
        }

    def reset_traces(self):
        """Reset all eligibility traces (useful for new tasks in continual learning)."""
        for group in self.param_groups:
            for p in group['params']:
                if len(self.state[p]) > 0:
                    self.state[p]['eligibility'].zero_()
                    self.state[p]['trace_magnitude'] = 0.0


class ContinualEPropOptimizer(EPropOptimizer):
    """
    Extended e-prop optimizer specifically designed for continual learning.

    Adds features for:
    - Domain-specific trace management
    - Task boundary detection
    - Adaptive trace decay based on task novelty
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_domain = "default"
        self.domain_traces = {}  # Store traces per domain
        self.task_novelty = 1.0  # Higher = more novel task

    def set_domain(self, domain_name: str, reset_traces: bool = False):
        """
        Switch to a new domain/task.

        Args:
            domain_name: Name of the domain
            reset_traces: If True, reset eligibility traces for new task
        """
        old_domain = self.current_domain

        # Save traces from old domain
        if old_domain not in self.domain_traces:
            self.domain_traces[old_domain] = {}

        for group in self.param_groups:
            for p in group['params']:
                if len(self.state[p]) > 0:
                    param_id = id(p)
                    self.domain_traces[old_domain][param_id] = {
                        'eligibility': self.state[p]['eligibility'].clone(),
                        'trace_magnitude': self.state[p]['trace_magnitude']
                    }

        # Switch to new domain
        self.current_domain = domain_name

        # Load or initialize traces for new domain
        if domain_name in self.domain_traces:
            # Restore saved traces
            for group in self.param_groups:
                for p in group['params']:
                    param_id = id(p)
                    if param_id in self.domain_traces[domain_name]:
                        saved = self.domain_traces[domain_name][param_id]
                        self.state[p]['eligibility'] = saved['eligibility'].clone()
                        self.state[p]['trace_magnitude'] = saved['trace_magnitude']
        elif reset_traces:
            # New domain: reset traces
            self.reset_traces()

    def compute_task_novelty(self, current_loss: float, domain_history: List[float]) -> float:
        """
        Estimate how novel the current task is.

        Higher novelty → faster trace decay (forget old patterns)
        Lower novelty → slower trace decay (retain knowledge)

        Args:
            current_loss: Current loss value
            domain_history: List of past losses for this domain

        Returns:
            Novelty score [0, 1]
        """
        if not domain_history:
            return 1.0  # Completely novel

        avg_past_loss = sum(domain_history) / len(domain_history)

        # High current loss compared to past → novel task
        novelty = min(1.0, current_loss / (avg_past_loss + 1e-8))

        return novelty

    def adapt_trace_decay(self, base_decay: float) -> float:
        """
        Adapt trace decay based on task novelty.

        Novel tasks → faster decay (forget)
        Familiar tasks → slower decay (retain)
        """
        # More novel → higher decay (faster forgetting)
        adapted_decay = base_decay + (1 - base_decay) * self.task_novelty * 0.1
        return min(0.99, adapted_decay)