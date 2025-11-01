"""
Working Memory - Prefrontal Cortex-Inspired Temporary Storage.

Implements a brain-inspired working memory system based on:
- Prefrontal cortex persistent activity
- Limited capacity (7±2 items - Miller's Law)
- Active maintenance through recurrent connections
- Selective gating of information

Key principles:
- Short-term maintenance of task-relevant information
- Capacity constraints force prioritization
- Interference from new information
- Rapid update and manipulation

References:
- Baddeley & Hitch, 1974: Working memory model
- Miller, 1956: The magical number seven
- Goldman-Rakic, 1995: PFC and working memory
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class WorkingMemoryConfig:
    """
    Configuration for working memory system.

    Args:
        capacity: Maximum items in working memory (default: 7, Miller's Law)
        d_model: Dimensionality of stored items
        decay_rate: Rate of memory decay per step
        gate_threshold: Threshold for gating new information
        update_noise: Noise added during updates (robustness)
    """
    capacity: int = 7  # Miller's Law: 7±2
    d_model: int = 768
    decay_rate: float = 0.01  # Slow decay
    gate_threshold: float = 0.5
    update_noise: float = 0.01


class WorkingMemory(nn.Module):
    """
    Working memory system inspired by prefrontal cortex.

    Maintains task-relevant information through:
    - Persistent neural activity (buffer storage)
    - Selective gating (what gets in)
    - Capacity constraints (what gets removed)
    - Active maintenance (recurrent updates)

    Example:
        >>> config = WorkingMemoryConfig(capacity=7, d_model=768)
        >>> wm = WorkingMemory(config)
        >>>
        >>> # Store new information
        >>> item = torch.randn(1, 768)
        >>> importance = 0.8
        >>> wm.store(item, importance)
        >>>
        >>> # Retrieve relevant items
        >>> query = torch.randn(1, 768)
        >>> retrieved = wm.retrieve(query, k=3)
    """

    def __init__(self, config: WorkingMemoryConfig):
        super().__init__()
        self.config = config

        # Working memory buffer (like PFC persistent activity)
        self.register_buffer(
            'buffer',
            torch.zeros(config.capacity, config.d_model)
        )

        # Importance scores (for capacity management)
        self.register_buffer(
            'importance',
            torch.zeros(config.capacity)
        )

        # Age of items (for decay)
        self.register_buffer(
            'age',
            torch.zeros(config.capacity)
        )

        # Occupancy mask
        self.register_buffer(
            'occupied',
            torch.zeros(config.capacity, dtype=torch.bool)
        )

        # Gating mechanism (learnable)
        self.gate = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid()
        )

        # Maintenance mechanism (recurrent update)
        self.maintenance = nn.GRUCell(config.d_model, config.d_model)

    def store(
        self,
        item: torch.Tensor,
        importance: float = 0.5
    ) -> bool:
        """
        Store new item in working memory.

        Args:
            item: Tensor to store (batch_size, d_model)
            importance: Importance score [0, 1]

        Returns:
            True if stored successfully, False if gated out
        """
        with torch.no_grad():
            # Check if item should be gated in
            gate_value = self.gate(item).item()

            if gate_value < self.config.gate_threshold:
                return False  # Gated out

            # Find slot to store
            if not self.occupied.all():
                # Use first empty slot
                idx = (~self.occupied).nonzero(as_tuple=True)[0][0]
            else:
                # Replace least important item
                idx = self.importance.argmin()

                # Only replace if new item is more important
                if importance <= self.importance[idx].item():
                    return False

            # Store item
            self.buffer[idx] = item.squeeze(0).detach()
            self.importance[idx] = importance
            self.age[idx] = 0.0
            self.occupied[idx] = True

            return True

    def retrieve(
        self,
        query: torch.Tensor,
        k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k most relevant items from working memory.

        Args:
            query: Query tensor (batch_size, d_model)
            k: Number of items to retrieve

        Returns:
            Tuple of (retrieved_items, similarity_scores)
        """
        if not self.occupied.any():
            # Empty working memory
            return (
                torch.zeros(k, self.config.d_model, device=query.device),
                torch.zeros(k, device=query.device)
            )

        # Compute similarity with occupied slots
        query_norm = torch.nn.functional.normalize(query, dim=-1)
        buffer_norm = torch.nn.functional.normalize(self.buffer, dim=-1)

        similarities = (query_norm @ buffer_norm.T).squeeze(0)

        # Mask unoccupied slots
        similarities = torch.where(
            self.occupied,
            similarities,
            torch.tensor(float('-inf'), device=similarities.device)
        )

        # Get top-k
        k_actual = min(k, self.occupied.sum().item())
        top_k_sim, top_k_idx = similarities.topk(k_actual)

        # Pad if necessary
        if k_actual < k:
            padding = k - k_actual
            retrieved = torch.cat([
                self.buffer[top_k_idx],
                torch.zeros(padding, self.config.d_model, device=query.device)
            ])
            scores = torch.cat([
                top_k_sim,
                torch.zeros(padding, device=query.device)
            ])
        else:
            retrieved = self.buffer[top_k_idx]
            scores = top_k_sim

        return retrieved, scores

    def maintain(self) -> None:
        """
        Active maintenance of working memory contents.

        Implements:
        - Recurrent updates (like persistent PFC activity)
        - Gradual decay
        - Age tracking
        """
        with torch.no_grad():
            if not self.occupied.any():
                return

            # Recurrent maintenance (GRU-based)
            for idx in self.occupied.nonzero(as_tuple=True)[0]:
                # Add small noise for robustness
                noise = torch.randn_like(self.buffer[idx]) * self.config.update_noise
                noisy_input = self.buffer[idx] + noise

                # Recurrent update
                self.buffer[idx] = self.maintenance(
                    noisy_input.unsqueeze(0),
                    self.buffer[idx].unsqueeze(0)
                ).squeeze(0)

            # Decay and aging
            self.age += 1.0
            decay_factor = 1.0 - (self.config.decay_rate * self.age)
            decay_factor = torch.clamp(decay_factor, 0.0, 1.0)

            # Apply decay
            for idx in self.occupied.nonzero(as_tuple=True)[0]:
                self.buffer[idx] *= decay_factor[idx]

            # Remove very old/decayed items
            threshold = 0.1
            to_remove = (decay_factor < threshold) & self.occupied
            if to_remove.any():
                self.buffer[to_remove] = 0.0
                self.importance[to_remove] = 0.0
                self.age[to_remove] = 0.0
                self.occupied[to_remove] = False

    def clear(self) -> None:
        """Clear all working memory contents."""
        with torch.no_grad():
            self.buffer.zero_()
            self.importance.zero_()
            self.age.zero_()
            self.occupied.fill_(False)

    def get_occupancy(self) -> float:
        """Get current working memory occupancy [0, 1]."""
        return self.occupied.sum().item() / self.config.capacity

    def get_contents(self) -> List[Tuple[torch.Tensor, float, float]]:
        """
        Get all current working memory contents.

        Returns:
            List of (item, importance, age) tuples
        """
        contents = []
        for idx in self.occupied.nonzero(as_tuple=True)[0]:
            contents.append((
                self.buffer[idx].clone(),
                self.importance[idx].item(),
                self.age[idx].item()
            ))
        return contents


if __name__ == "__main__":
    print("Testing Working Memory System...")

    # Create working memory
    config = WorkingMemoryConfig(capacity=7, d_model=128)
    wm = WorkingMemory(config)

    print(f"\nCapacity: {config.capacity} items (Miller's Law: 7±2)")
    print(f"Dimensionality: {config.d_model}")

    # Store some items
    print("\nStoring items...")
    for i in range(10):
        item = torch.randn(1, config.d_model)
        importance = 0.3 + 0.7 * (i / 10)  # Increasing importance
        stored = wm.store(item, importance)
        if stored:
            print(f"  Item {i}: stored (importance={importance:.2f})")
        else:
            print(f"  Item {i}: rejected (importance={importance:.2f})")

    print(f"\nOccupancy: {wm.get_occupancy():.1%}")

    # Retrieve items
    print("\nRetrieving top-3 items...")
    query = torch.randn(1, config.d_model)
    retrieved, scores = wm.retrieve(query, k=3)

    for i, (item, score) in enumerate(zip(retrieved, scores)):
        print(f"  Rank {i+1}: similarity={score:.3f}")

    # Maintenance over time
    print("\nMaintenance over 50 steps...")
    for step in range(50):
        wm.maintain()
        if step % 10 == 0:
            print(f"  Step {step}: occupancy={wm.get_occupancy():.1%}")

    # Show final contents
    print("\nFinal working memory contents:")
    contents = wm.get_contents()
    for i, (item, imp, age) in enumerate(contents):
        print(f"  Slot {i}: importance={imp:.2f}, age={age:.0f}")

    print("\n✓ Working Memory system operational!")
    print("Implements: capacity limits, selective gating, active maintenance, decay")
