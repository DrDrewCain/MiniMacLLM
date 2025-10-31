"""
Elastic Weight Consolidation (EWC) for Continual Learning.

EWC prevents catastrophic forgetting by:
1. Computing Fisher Information Matrix (FIM) after learning a task
2. Penalizing changes to important weights when learning new tasks

Mathematical formula:
    L_EWC = L_task + (λ/2) * Σ F_i * (θ_i - θ_i*)²

Where:
- L_task: Loss on current task
- F_i: Fisher information (importance) of parameter i
- θ_i: Current parameter value
- θ_i*: Parameter value from previous task
- λ: Strength of regularization

References:
    - "Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al., 2017)
    - "Continual Learning Through Synaptic Intelligence" (Zenke et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class EWCConfig:
    """
    Configuration for EWC.

    Args:
        lambda_ewc: Strength of EWC regularization (higher = more protection)
        fisher_samples: Number of samples to use for Fisher estimation
        fisher_estimate_mode: How to estimate Fisher ("diagonal", "full", "empirical")
        normalize_fisher: Whether to normalize Fisher values
        online: Whether to use online EWC (accumulate Fisher over tasks)
    """

    lambda_ewc: float = 1000.0
    fisher_samples: int = 200
    fisher_estimate_mode: str = "diagonal"
    normalize_fisher: bool = True
    online: bool = False
    gamma: float = 1.0  # Decay factor for online EWC


class EWC:
    """
    Elastic Weight Consolidation for preventing catastrophic forgetting.

    Usage:
        1. Train model on task A
        2. Compute Fisher: ewc.compute_fisher_information(model, dataloader)
        3. Train on task B with EWC penalty: loss = task_loss + ewc.penalty(model)

    Args:
        model: Neural network model
        config: EWC configuration
        device: Device to compute on

    Example:
        >>> ewc = EWC(model, EWCConfig(lambda_ewc=1000))
        >>> # After training task A
        >>> ewc.compute_fisher_information(model, task_a_dataloader)
        >>> # Training task B
        >>> for batch in task_b_dataloader:
        ...     loss = compute_loss(model, batch)
        ...     loss = loss + ewc.penalty(model)
        ...     loss.backward()
    """

    def __init__(self, model: nn.Module, config: EWCConfig, device: str = "cpu"):
        self.config = config
        self.device = device

        # Store parameter names (only parameters with gradients)
        self.param_names = [name for name, param in model.named_parameters() if param.requires_grad]

        # Storage for Fisher information and old parameters
        self.fisher: Dict[str, torch.Tensor] = {}
        self.old_params: Dict[str, torch.Tensor] = {}

        # For online EWC (multiple tasks)
        self.task_count = 0

    def compute_fisher_information(
        self, model: nn.Module, dataloader, max_samples: Optional[int] = None
    ):
        """
        Compute Fisher Information Matrix.

        Uses the diagonal Fisher approximation:
        F_i ≈ E[(∂log p(y|x,θ) / ∂θ_i)²]

        This measures how much each parameter affects the loss.

        Args:
            model: Model to compute Fisher for
            dataloader: DataLoader with samples from the task
            max_samples: Maximum samples to use (None = use fisher_samples from config)
        """
        print("Computing Fisher Information Matrix...")

        model.eval()
        max_samples = max_samples or self.config.fisher_samples

        # Initialize Fisher dictionary
        fisher = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Sample counter
        num_samples = 0

        # Compute gradients over samples
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing Fisher")):
            if num_samples >= max_samples:
                break

            # Move batch to device
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
            elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                input_ids = batch[0].to(self.device)
                labels = batch[1].to(self.device)
            else:
                input_ids = batch.to(self.device)
                labels = input_ids

            batch_size = input_ids.size(0)

            # Forward pass
            model.zero_grad()

            if hasattr(model, "forward"):
                # Standard model
                if hasattr(model, "base_model"):
                    # LoRA wrapped model
                    outputs = model.base_model(input_ids)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                else:
                    outputs = model(input_ids)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
            else:
                raise ValueError("Model must have forward method")

            # Compute log likelihood
            # For language modeling: log p(y|x) = -CrossEntropy(logits, labels)
            if labels is not None:
                # Flatten for loss computation
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",  # Sum over batch for Fisher
                )
            else:
                # If no labels, use entropy of predictions
                probs = F.softmax(logits, dim=-1)
                loss = -(probs * torch.log(probs + 1e-10)).sum()

            # Backward to get gradients
            loss.backward()

            # Accumulate squared gradients (Fisher approximation)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += (param.grad.detach() ** 2) * batch_size

            num_samples += batch_size

        # Normalize by number of samples
        for name in fisher:
            fisher[name] /= num_samples

            # Optional: normalize Fisher values
            if self.config.normalize_fisher:
                fisher[name] = fisher[name] / (fisher[name].max() + 1e-8)

        # Store Fisher and current parameters
        if self.config.online and self.task_count > 0:
            # Online EWC: accumulate Fisher
            for name in fisher:
                if name in self.fisher:
                    self.fisher[name] = self.config.gamma * self.fisher[name] + fisher[name]
                else:
                    self.fisher[name] = fisher[name]
        else:
            # Standard EWC: replace Fisher
            self.fisher = fisher

        # Store current parameters as "old" parameters
        self.old_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        self.task_count += 1

        print(f"Fisher computed for task {self.task_count}")
        print(f"Number of parameters: {len(self.fisher)}")

        # Print statistics
        total_fisher = sum(f.sum().item() for f in self.fisher.values())
        print(f"Total Fisher mass: {total_fisher:.4f}")

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty term.

        L_EWC = (λ/2) * Σ F_i * (θ_i - θ_i*)²

        Args:
            model: Current model

        Returns:
            EWC penalty (scalar tensor)
        """
        if not self.fisher or not self.old_params:
            # No Fisher computed yet, no penalty
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)

        for name, param in model.named_parameters():
            if name in self.fisher and name in self.old_params:
                # Compute squared difference weighted by Fisher
                fisher = self.fisher[name]
                old_param = self.old_params[name]

                loss += (fisher * (param - old_param) ** 2).sum()

        # Apply lambda scaling
        loss = (self.config.lambda_ewc / 2.0) * loss

        return loss

    def save(self, path: str):
        """Save EWC state (Fisher and old parameters)."""
        state = {
            "fisher": self.fisher,
            "old_params": self.old_params,
            "task_count": self.task_count,
            "config": self.config.__dict__,
        }
        torch.save(state, path)
        print(f"Saved EWC state to {path}")

    def load(self, path: str):
        """Load EWC state."""
        state = torch.load(path, map_location=self.device)

        self.fisher = {k: v.to(self.device) for k, v in state["fisher"].items()}
        self.old_params = {k: v.to(self.device) for k, v in state["old_params"].items()}
        self.task_count = state["task_count"]

        print(f"Loaded EWC state from {path} (task {self.task_count})")

    def get_importance_stats(self) -> Dict:
        """Get statistics about parameter importance."""
        if not self.fisher:
            return {}

        fisher_values = torch.cat([f.flatten() for f in self.fisher.values()])

        return {
            "mean_importance": fisher_values.mean().item(),
            "std_importance": fisher_values.std().item(),
            "max_importance": fisher_values.max().item(),
            "min_importance": fisher_values.min().item(),
            "total_importance": fisher_values.sum().item(),
        }


class OnlineEWC(EWC):
    """
    Online EWC variant that accumulates Fisher over multiple tasks.

    Instead of storing Fisher for each task separately, maintains a
    running average of Fisher information.

    Formula:
        F_new = γ * F_old + F_current

    Where γ is a decay factor (typically 1.0).

    This is more memory efficient for many tasks.
    """

    def __init__(self, model: nn.Module, config: EWCConfig, device: str = "cpu"):
        # Force online mode
        config.online = True
        super().__init__(model, config, device)


def consolidate_model_knowledge(model: nn.Module, dataloader, ewc: EWC, device: str = "cpu"):
    """
    Convenience function to consolidate knowledge after learning a task.

    This computes Fisher and stores current parameters.

    Args:
        model: Trained model
        dataloader: Data from current task
        ewc: EWC instance
        device: Device to compute on
    """
    model.eval()
    model.to(device)

    ewc.compute_fisher_information(model, dataloader)

    return ewc


if __name__ == "__main__":
    # Test EWC implementation
    print("Testing Elastic Weight Consolidation...")

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleModel()
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dummy data
    class DummyDataset:
        def __init__(self, num_samples=100):
            self.data = [
                (torch.randn(2, 10), torch.randint(0, 10, (2, 10))) for _ in range(num_samples // 2)
            ]

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

    dataset = DummyDataset(num_samples=50)

    # Create EWC
    config = EWCConfig(lambda_ewc=1000.0, fisher_samples=20, normalize_fisher=True)

    ewc = EWC(model, config)

    # Compute Fisher
    print("\nComputing Fisher for task A...")
    ewc.compute_fisher_information(model, dataset, max_samples=20)

    # Print importance stats
    print("\nParameter importance statistics:")
    stats = ewc.get_importance_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")

    # Simulate training on task B
    print("\nSimulating training on task B...")

    # Compute penalty before update
    penalty_before = ewc.penalty(model)
    print(f"EWC penalty before update: {penalty_before:.4f}")

    # Simulate parameter update
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.01)

    # Compute penalty after update
    penalty_after = ewc.penalty(model)
    print(f"EWC penalty after update: {penalty_after:.4f}")
    print(f"Penalty increased: {penalty_after > penalty_before}")

    # Test save/load
    print("\nTesting save/load...")
    ewc.save("test_ewc.pt")

    new_ewc = EWC(model, config)
    new_ewc.load("test_ewc.pt")

    penalty_loaded = new_ewc.penalty(model)
    print(f"Penalty after load: {penalty_loaded:.4f}")
    print(f"Matches original: {torch.allclose(penalty_after, penalty_loaded)}")

    import os

    os.remove("test_ewc.pt")

    print("\n✓ EWC implementation complete!")
