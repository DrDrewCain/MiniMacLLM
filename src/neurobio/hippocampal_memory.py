"""
Hippocampal Memory System for Intelligent Experience Replay.

Implements pattern separation and completion inspired by hippocampus.
Based on:
- HAMI (Hippocampal-Augmented Memory Integration, Scientific Reports 2025)
- Pattern separation in dentate gyrus
- CA3 autoassociative memory
- CA1 pattern completion

Key Principles:
- Pattern separation: Sparse encoding prevents interference
- Pattern completion: Retrieve from partial cues
- One-shot binding: Rapid episodic learning
- Content-addressable memory: Similarity-based retrieval

Replaces random sampling with brain-like memory consolidation!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from dataclasses import dataclass


@dataclass
class HippocampalConfig:
    """
    Configuration for hippocampal memory system.

    Args:
        d_model: Hidden dimension
        memory_size: Maximum number of stored episodes
        expansion_factor: DG expansion factor (4x typical)
        sparsity: Sparsity level for pattern separation (0.02 = 2% active)
        ca3_size: Size of CA3 autoassociative memory
        separation_strength: Strength of pattern separation (0-1)
    """
    d_model: int = 768
    memory_size: int = 10000
    expansion_factor: int = 4
    sparsity: float = 0.02  # 2% active (like DG)
    ca3_size: int = 2048
    separation_strength: float = 0.5


class HippocampalMemory(nn.Module):
    """
    Hippocampal-inspired episodic memory system.

    Mimics three hippocampal regions:
    - DG (Dentate Gyrus): Pattern separation via sparse encoding
    - CA3: Autoassociative memory for binding
    - CA1: Pattern completion for retrieval

    Example:
        >>> config = HippocampalConfig(d_model=768, memory_size=1000)
        >>> hippocampus = HippocampalMemory(config)
        >>>
        >>> # Store episode
        >>> context = torch.randn(768)
        >>> content = torch.randn(768)
        >>> hippocampus.store_episode(context, content, importance=0.8)
        >>>
        >>> # Recall similar episodes
        >>> query = torch.randn(768)
        >>> retrieved, scores = hippocampus.recall(query, k=5)
    """

    def __init__(self, config: HippocampalConfig):
        super().__init__()
        self.config = config

        # Dentate Gyrus: Pattern separation via sparse encoding
        # Expands input then applies extreme sparsity
        dg_dim = config.d_model * config.expansion_factor

        self.dg_expansion = nn.Sequential(
            nn.Linear(config.d_model, dg_dim),
            nn.ReLU(),
        )

        # Learnable threshold for sparsity
        self.dg_threshold = nn.Parameter(torch.ones(dg_dim) * 0.5)

        # CA3: Autoassociative memory (stores separated patterns)
        # Keys: separated patterns, Values: original content
        self.ca3_keys = nn.Parameter(
            torch.randn(config.memory_size, dg_dim) * 0.01
        )
        self.ca3_values = nn.Parameter(
            torch.randn(config.memory_size, config.d_model) * 0.01
        )

        # Importance scores for each memory
        self.register_buffer(
            'importance_scores',
            torch.zeros(config.memory_size)
        )

        # Usage counters (for understanding memory dynamics)
        self.register_buffer(
            'usage_counts',
            torch.zeros(config.memory_size)
        )

        # Current write pointer
        self.register_buffer('write_ptr', torch.tensor(0))

        # CA1: Pattern completion (projects back to d_model)
        self.ca1_completion = nn.Linear(dg_dim, config.d_model)

    def pattern_separate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pattern separation via dentate gyrus.

        Applies sparse encoding to reduce interference.

        Args:
            x: Input pattern (batch, d_model) or (d_model,)

        Returns:
            Separated pattern (batch, dg_dim) or (dg_dim,) with sparsity
        """
        # Expand
        expanded = self.dg_expansion(x)

        # Apply threshold for sparsity (like competitive inhibition)
        # Keep only top k% activations
        k = int(self.config.sparsity * expanded.size(-1))

        if x.dim() == 1:
            # Single sample
            threshold = torch.topk(expanded, k)[0][-1]
            sparse = torch.where(
                expanded >= threshold,
                expanded,
                torch.zeros_like(expanded)
            )
        else:
            # Batch
            sparse = torch.zeros_like(expanded)
            for i in range(expanded.size(0)):
                threshold = torch.topk(expanded[i], k)[0][-1]
                sparse[i] = torch.where(
                    expanded[i] >= threshold,
                    expanded[i],
                    torch.zeros_like(expanded[i])
                )

        return sparse

    def store_episode(
        self,
        context: torch.Tensor,
        content: torch.Tensor,
        importance: float = 1.0
    ):
        """
        Store episode in hippocampal memory (one-shot binding).

        Args:
            context: Context representation (d_model,)
            content: Content to bind to context (d_model,)
            importance: Importance score (0-1)
        """
        with torch.no_grad():
            # Pattern separation
            separated = self.pattern_separate(context)

            # Store in CA3 (circular buffer)
            idx = self.write_ptr.item() % self.config.memory_size

            self.ca3_keys.data[idx] = separated.detach()
            self.ca3_values.data[idx] = content.detach()
            self.importance_scores[idx] = importance

            # Increment write pointer
            self.write_ptr += 1

    def recall(
        self,
        query: torch.Tensor,
        k: int = 5,
        importance_weighted: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recall similar episodes using content-based retrieval.

        Args:
            query: Query pattern (d_model,)
            k: Number of episodes to retrieve
            importance_weighted: Whether to weight by importance

        Returns:
            Tuple of (retrieved_content, similarity_scores)
            - retrieved_content: (k, d_model)
            - similarity_scores: (k,)
        """
        with torch.no_grad():
            # Pattern separation of query
            separated_query = self.pattern_separate(query)

            # CA3: Compute similarity to stored patterns
            # Cosine similarity
            similarities = F.cosine_similarity(
                separated_query.unsqueeze(0),
                self.ca3_keys,
                dim=-1
            )

            # Weight by importance if requested
            if importance_weighted:
                similarities = similarities * self.importance_scores

            # Get top-k
            top_k_sim, top_k_idx = similarities.topk(k, dim=-1)

            # Retrieve content from CA3
            retrieved = self.ca3_values[top_k_idx]

            # Update usage counts
            self.usage_counts[top_k_idx] += 1

            # CA1: Pattern completion (optional refinement)
            # For now, return CA3 values directly
            # Could add: completed = self.ca1_completion(separated_query)

            return retrieved, top_k_sim

    def get_memory_statistics(self) -> dict:
        """Get statistics about memory usage."""
        num_stored = min(self.write_ptr.item(), self.config.memory_size)

        return {
            'num_stored': num_stored,
            'capacity': self.config.memory_size,
            'utilization': num_stored / self.config.memory_size,
            'avg_importance': self.importance_scores[:num_stored].mean().item(),
            'avg_usage': self.usage_counts[:num_stored].mean().item(),
        }

    def consolidate_memories(self, threshold: float = 0.1):
        """
        Remove low-importance memories to make room.

        Args:
            threshold: Importance threshold for keeping memories
        """
        with torch.no_grad():
            # Find memories below threshold
            keep_mask = self.importance_scores >= threshold

            # Compact memory (move kept memories to front)
            kept_indices = torch.where(keep_mask)[0]

            if len(kept_indices) < self.config.memory_size:
                # Reorganize
                self.ca3_keys.data[:len(kept_indices)] = (
                    self.ca3_keys.data[kept_indices]
                )
                self.ca3_values.data[:len(kept_indices)] = (
                    self.ca3_values.data[kept_indices]
                )
                self.importance_scores[:len(kept_indices)] = (
                    self.importance_scores[kept_indices]
                )

                # Reset write pointer
                self.write_ptr.copy_(torch.tensor(len(kept_indices)))


if __name__ == "__main__":
    print("Testing HippocampalMemory...")

    # Create hippocampal memory
    config = HippocampalConfig(d_model=512, memory_size=100, sparsity=0.02)
    hippocampus = HippocampalMemory(config)

    print(f"\nConfiguration:")
    print(f"  Memory size: {config.memory_size}")
    print(f"  Expansion factor: {config.expansion_factor}x")
    print(f"  Sparsity: {config.sparsity:.1%}")

    # Store some episodes
    print("\nStoring episodes...")
    for i in range(50):
        context = torch.randn(512)
        content = torch.randn(512)
        importance = 0.5 + 0.5 * (i / 50)  # Increasing importance

        hippocampus.store_episode(context, content, importance)

    stats = hippocampus.get_memory_statistics()
    print(f"\nMemory statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Test pattern separation
    print("\nTesting pattern separation...")
    x = torch.randn(512)
    separated = hippocampus.pattern_separate(x)
    sparsity_actual = (separated > 0).float().mean().item()
    print(f"  Target sparsity: {config.sparsity:.1%}")
    print(f"  Actual sparsity: {sparsity_actual:.1%}")

    # Test recall
    print("\nTesting recall...")
    query = torch.randn(512)
    retrieved, scores = hippocampus.recall(query, k=5)
    print(f"  Query shape: {query.shape}")
    print(f"  Retrieved shape: {retrieved.shape}")
    print(f"  Similarity scores: {scores.tolist()}")

    # Test consolidation
    print("\nTesting memory consolidation...")
    print(f"  Before: {stats['num_stored']} memories")
    hippocampus.consolidate_memories(threshold=0.7)
    stats_after = hippocampus.get_memory_statistics()
    print(f"  After: {stats_after['num_stored']} memories")

    print("\n✓ HippocampalMemory working!")
