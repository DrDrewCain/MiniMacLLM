"""
Experience Replay Buffer for Continual Learning.

Stores past examples and replays them during training to prevent catastrophic forgetting.

Key strategies:
1. Reservoir sampling - maintain fixed-size buffer with uniform distribution
2. Importance sampling - prioritize difficult/important examples
3. Diversity sampling - maximize coverage of data distribution

References:
    - "Continual Learning Through Synaptic Intelligence" (Zenke et al., 2017)
    - "Experience Replay for Continual Learning" (Rolnick et al., 2019)
    - Classic technique from reinforcement learning
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import random
import hashlib


@dataclass
class Experience:
    """
    Single experience (example) to store in replay buffer.

    Args:
        input_ids: Token IDs of shape (seq_len,)
        labels: Target token IDs (same shape as input_ids)
        importance: Importance score (higher = more important to remember)
        domain: Domain/task label (e.g., "math", "code", "general")
        metadata: Additional information (loss, timestamp, etc.)
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    importance: float = 1.0
    domain: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate experience."""
        assert (
            self.input_ids.shape == self.labels.shape
        ), "input_ids and labels must have same shape"

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "input_ids": self.input_ids,
            "labels": self.labels,
            "importance": self.importance,
            "domain": self.domain,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Experience":
        """Create experience from dictionary."""
        return cls(
            input_ids=data["input_ids"],
            labels=data["labels"],
            importance=data.get("importance", 1.0),
            domain=data.get("domain"),
            metadata=data.get("metadata", {}),
        )

    def get_hash(self) -> str:
        """Get hash of input for deduplication."""
        # Convert to bytes without numpy (for compatibility when numpy is unavailable)
        # Use tolist() and convert to string for hashing
        tensor_list = self.input_ids.cpu().tolist()
        tensor_str = str(tensor_list).encode("utf-8")
        return hashlib.md5(tensor_str).hexdigest()


class ExperienceReplayBuffer:
    """
    Experience replay buffer with multiple sampling strategies.

    Maintains a fixed-size buffer of past examples and provides methods to:
    - Add new experiences
    - Sample batches for training
    - Manage importance scores
    - Handle buffer overflow

    Args:
        max_size: Maximum number of experiences to store
        sampling_strategy: How to sample ("uniform", "importance", "reservoir")
        importance_decay: Decay factor for old experiences (1.0 = no decay)
        device: Device to store tensors on

    Example:
        >>> buffer = ExperienceReplayBuffer(max_size=10000)
        >>> # Add experiences
        >>> for input_ids, labels in dataset:
        ...     exp = Experience(input_ids, labels, importance=0.8)
        ...     buffer.add(exp)
        >>> # Sample for training
        >>> batch = buffer.sample(batch_size=32, new_data_ratio=0.5)
    """

    def __init__(
        self,
        max_size: int = 10000,
        sampling_strategy: str = "importance",
        importance_decay: float = 1.0,
        device: str = "cpu",
    ):
        self.max_size = max_size
        self.sampling_strategy = sampling_strategy
        self.importance_decay = importance_decay
        self.device = device

        # Storage
        self.buffer: List[Experience] = []
        self.size = 0

        # For reservoir sampling
        self.total_seen = 0

        # For deduplication
        self.seen_hashes: set = set()

        # Statistics
        self.stats = {"added": 0, "replaced": 0, "duplicates": 0, "samples_drawn": 0}

    def add(self, experience: Experience, allow_duplicates: bool = False) -> bool:
        """
        Add experience to buffer.

        Args:
            experience: Experience to add
            allow_duplicates: Whether to allow duplicate inputs

        Returns:
            True if added, False if rejected
        """
        self.total_seen += 1

        # Check for duplicates
        if not allow_duplicates:
            exp_hash = experience.get_hash()
            if exp_hash in self.seen_hashes:
                self.stats["duplicates"] += 1
                return False
            self.seen_hashes.add(exp_hash)

        # Move to device
        experience.input_ids = experience.input_ids.to(self.device)
        experience.labels = experience.labels.to(self.device)

        # Add or replace
        if self.size < self.max_size:
            # Buffer not full, just add
            self.buffer.append(experience)
            self.size += 1
            self.stats["added"] += 1
            return True
        else:
            # Buffer full, need replacement strategy
            return self._replace_experience(experience)

    def _replace_experience(self, new_experience: Experience) -> bool:
        """
        Replace an experience in full buffer.

        Uses different strategies based on sampling_strategy:
        - reservoir: Replace with probability 1/total_seen
        - importance: Replace least important
        - uniform: Replace random

        Args:
            new_experience: New experience to potentially add

        Returns:
            True if added, False if rejected
        """
        if self.sampling_strategy == "reservoir":
            # Reservoir sampling: replace with prob max_size/total_seen
            prob = self.max_size / self.total_seen
            if random.random() < prob:
                idx = random.randint(0, self.max_size - 1)
                # Remove old hash
                old_hash = self.buffer[idx].get_hash()
                self.seen_hashes.discard(old_hash)
                # Replace
                self.buffer[idx] = new_experience
                self.stats["replaced"] += 1
                return True
            return False

        elif self.sampling_strategy == "importance":
            # Find least important experience
            importances = [exp.importance for exp in self.buffer]
            min_importance = min(importances)

            # Only replace if new experience is more important
            if new_experience.importance > min_importance:
                idx = importances.index(min_importance)
                # Remove old hash
                old_hash = self.buffer[idx].get_hash()
                self.seen_hashes.discard(old_hash)
                # Replace
                self.buffer[idx] = new_experience
                self.stats["replaced"] += 1
                return True
            return False

        else:  # uniform
            # Replace random experience
            idx = random.randint(0, self.max_size - 1)
            old_hash = self.buffer[idx].get_hash()
            self.seen_hashes.discard(old_hash)
            self.buffer[idx] = new_experience
            self.stats["replaced"] += 1
            return True

    def sample(self, batch_size: int, importance_weighted: bool = None) -> List[Experience]:
        """
        Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample
            importance_weighted: Whether to use importance weighting
                                (None = use buffer's strategy)

        Returns:
            List of sampled experiences
        """
        if self.size == 0:
            return []

        batch_size = min(batch_size, self.size)

        # Determine sampling method
        use_importance = (
            importance_weighted
            if importance_weighted is not None
            else self.sampling_strategy == "importance"
        )

        if use_importance:
            # Importance-weighted sampling
            importances = np.array([exp.importance for exp in self.buffer[: self.size]])
            probabilities = importances / importances.sum()

            indices = np.random.choice(self.size, size=batch_size, replace=False, p=probabilities)
        else:
            # Uniform sampling
            indices = np.random.choice(self.size, size=batch_size, replace=False)

        self.stats["samples_drawn"] += batch_size

        return [self.buffer[i] for i in indices]

    def sample_by_domain(self, domain: str, batch_size: int) -> List[Experience]:
        """
        Sample experiences from a specific domain.

        Args:
            domain: Domain to sample from
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences from domain
        """
        domain_experiences = [exp for exp in self.buffer if exp.domain == domain]

        if not domain_experiences:
            return []

        batch_size = min(batch_size, len(domain_experiences))
        return random.sample(domain_experiences, batch_size)

    def update_importance(self, experience_idx: int, new_importance: float):
        """
        Update importance of an experience.

        Used to increase importance of difficult examples.

        Args:
            experience_idx: Index of experience
            new_importance: New importance score
        """
        if 0 <= experience_idx < self.size:
            self.buffer[experience_idx].importance = new_importance

    def apply_importance_decay(self):
        """Decay importance of all experiences (make older ones less important)."""
        if self.importance_decay < 1.0:
            for exp in self.buffer:
                exp.importance *= self.importance_decay

    def get_domain_distribution(self) -> Dict[str, int]:
        """Get count of experiences per domain."""
        distribution = {}
        for exp in self.buffer:
            domain = exp.domain or "unknown"
            distribution[domain] = distribution.get(domain, 0) + 1
        return distribution

    def clear(self):
        """Clear all experiences."""
        self.buffer.clear()
        self.seen_hashes.clear()
        self.size = 0
        self.total_seen = 0
        self.stats = {"added": 0, "replaced": 0, "duplicates": 0, "samples_drawn": 0}

    def save(self, path: str):
        """Save buffer to disk."""
        state = {
            "buffer": [exp.to_dict() for exp in self.buffer],
            "size": self.size,
            "total_seen": self.total_seen,
            "seen_hashes": list(self.seen_hashes),
            "stats": self.stats,
            "config": {
                "max_size": self.max_size,
                "sampling_strategy": self.sampling_strategy,
                "importance_decay": self.importance_decay,
            },
        }
        torch.save(state, path)
        print(f"Saved replay buffer to {path}")

    def load(self, path: str):
        """Load buffer from disk."""
        state = torch.load(path, map_location=self.device)

        self.buffer = [Experience.from_dict(exp_dict) for exp_dict in state["buffer"]]
        self.size = state["size"]
        self.total_seen = state["total_seen"]
        self.seen_hashes = set(state["seen_hashes"])
        self.stats = state["stats"]

        # Update config
        config = state["config"]
        self.max_size = config["max_size"]
        self.sampling_strategy = config["sampling_strategy"]
        self.importance_decay = config["importance_decay"]

        print(f"Loaded replay buffer from {path} ({self.size} experiences)")

    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        return {
            **self.stats,
            "current_size": self.size,
            "max_size": self.max_size,
            "total_seen": self.total_seen,
            "fill_percentage": 100.0 * self.size / self.max_size,
            "domain_distribution": self.get_domain_distribution(),
        }

    def __len__(self) -> int:
        """Get current buffer size."""
        return self.size

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ExperienceReplayBuffer("
            f"size={self.size}/{self.max_size}, "
            f"strategy={self.sampling_strategy}, "
            f"domains={len(self.get_domain_distribution())})"
        )


class StreamingReplayBuffer(ExperienceReplayBuffer):
    """
    Extended replay buffer for streaming data.

    Adds features for:
    - Batch addition of experiences
    - Automatic importance scoring based on loss
    - Recent experience tracking
    - Time-based prioritization

    Args:
        max_size: Maximum buffer size
        recent_window: Size of recent experience window
        **kwargs: Additional arguments for base buffer
    """

    def __init__(self, max_size: int = 10000, recent_window: int = 1000, **kwargs):
        super().__init__(max_size=max_size, **kwargs)
        self.recent_window = recent_window
        self.recent_experiences = deque(maxlen=recent_window)

    def add_batch(self, batch: List[Experience], auto_importance: bool = True) -> int:
        """
        Add multiple experiences at once.

        Args:
            batch: List of experiences
            auto_importance: Whether to automatically set importance from metadata

        Returns:
            Number of experiences successfully added
        """
        added_count = 0

        for exp in batch:
            # Auto-set importance from loss if available
            if auto_importance and "loss" in exp.metadata:
                # Higher loss = more important (harder example)
                loss = exp.metadata["loss"]
                exp.importance = min(loss, 10.0)  # Cap at 10

            if self.add(exp):
                added_count += 1
                self.recent_experiences.append(exp)

        return added_count

    def get_recent_batch(self, batch_size: int) -> List[Experience]:
        """Get batch from recent experiences."""
        if not self.recent_experiences:
            return []

        batch_size = min(batch_size, len(self.recent_experiences))
        return random.sample(list(self.recent_experiences), batch_size)

    def mixed_sample(self, batch_size: int, recent_ratio: float = 0.3) -> List[Experience]:
        """
        Sample mix of recent and historical experiences.

        Args:
            batch_size: Total batch size
            recent_ratio: Fraction of batch from recent experiences

        Returns:
            Mixed batch of experiences
        """
        n_recent = int(batch_size * recent_ratio)
        n_historical = batch_size - n_recent

        recent_batch = self.get_recent_batch(n_recent)
        historical_batch = self.sample(n_historical)

        return recent_batch + historical_batch


if __name__ == "__main__":
    # Test experience replay buffer
    print("Testing Experience Replay Buffer...")

    # Create buffer
    buffer = ExperienceReplayBuffer(max_size=100, sampling_strategy="importance")

    print(f"Buffer: {buffer}")

    # Create sample experiences
    print("\nAdding experiences...")
    for i in range(150):
        input_ids = torch.randint(0, 1000, (20,))
        labels = torch.randint(0, 1000, (20,))
        importance = random.uniform(0.5, 1.5)
        domain = random.choice(["math", "code", "general"])

        exp = Experience(
            input_ids=input_ids,
            labels=labels,
            importance=importance,
            domain=domain,
            metadata={"step": i},
        )

        buffer.add(exp)

    # Print stats
    print(f"\nBuffer stats:")
    stats = buffer.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test sampling
    print(f"\nSampling batch...")
    batch = buffer.sample(batch_size=10, importance_weighted=True)
    print(f"Sampled {len(batch)} experiences")
    print(f"Importance scores: {[f'{exp.importance:.2f}' for exp in batch[:5]]}...")

    # Test domain sampling
    print(f"\nSampling from 'math' domain...")
    math_batch = buffer.sample_by_domain("math", batch_size=5)
    print(f"Sampled {len(math_batch)} math experiences")

    # Test save/load
    print(f"\nTesting save/load...")
    buffer.save("test_buffer.pt")

    new_buffer = ExperienceReplayBuffer(max_size=100)
    new_buffer.load("test_buffer.pt")
    print(f"Loaded buffer: {new_buffer}")
    print(f"Matches original: {len(new_buffer) == len(buffer)}")

    import os

    os.remove("test_buffer.pt")

    # Test streaming buffer
    print(f"\n\nTesting Streaming Buffer...")
    streaming_buffer = StreamingReplayBuffer(max_size=100, recent_window=20)

    # Add batch
    batch_experiences = []
    for i in range(30):
        input_ids = torch.randint(0, 1000, (15,))
        labels = torch.randint(0, 1000, (15,))
        exp = Experience(
            input_ids=input_ids, labels=labels, metadata={"loss": random.uniform(0.5, 3.0)}
        )
        batch_experiences.append(exp)

    added = streaming_buffer.add_batch(batch_experiences, auto_importance=True)
    print(f"Added {added} experiences to streaming buffer")

    # Mixed sampling
    mixed_batch = streaming_buffer.mixed_sample(batch_size=10, recent_ratio=0.5)
    print(f"Mixed sample: {len(mixed_batch)} experiences")

    print("\n✓ Experience Replay Buffer implementation complete!")
