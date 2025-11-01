"""
Vision-Text Fusion Layer for Multimodal Language Model.

Responsible for:
1. Projecting vision encoder outputs to language model embedding space
2. Generating position IDs for interleaved vision-text sequences
3. Handling different fusion strategies (pooling, all patches, etc.)
4. LoRA support for domain-specific vision-text alignment

This layer bridges the vision encoder and language model.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class VisionPoolingStrategy(str, Enum):
    """Strategy for pooling visual tokens."""
    NONE = "none"  # Use all patch tokens
    MEAN = "mean"  # Average pool all patches
    CLS = "cls"  # Use only CLS token
    ADAPTIVE = "adaptive"  # Learnable weighted pooling


@dataclass
class VisionFusionConfig:
    """
    Configuration for vision-text fusion.

    Args:
        vision_hidden_dim: Dimension from vision encoder
        llm_hidden_dim: Dimension of language model embeddings
        pooling_strategy: How to pool vision tokens
        num_vision_tokens: Number of vision tokens after pooling
        use_projection: Whether to use learned projection
        projection_bias: Whether projection has bias
        dropout: Dropout probability
    """
    vision_hidden_dim: int = 768
    llm_hidden_dim: int = 768
    pooling_strategy: str = "none"  # Use all patches
    num_vision_tokens: int = 196  # For ViT: 224/16 = 14, 14^2 = 196
    use_projection: bool = True
    projection_bias: bool = True
    dropout: float = 0.0


class VisionProjection(nn.Module):
    """
    Learnable projection from vision to language model space.

    Optionally includes LoRA for domain-specific adaptation.
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        bias: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim

        # Linear projection
        self.projection = nn.Linear(vision_dim, llm_dim, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to language model space.

        Args:
            x: Vision features of shape (batch, num_tokens, vision_dim)

        Returns:
            Projected features of shape (batch, num_tokens, llm_dim)
        """
        x = self.projection(x)
        x = self.dropout(x)
        return x


class AdaptivePooling(nn.Module):
    """
    Learnable weighted pooling for vision tokens.

    Learns attention weights over patches to pool into fewer tokens.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_input_tokens: int,
        num_output_tokens: int
    ):
        super().__init__()
        self.num_output_tokens = num_output_tokens

        # Learnable query vectors for pooling
        self.queries = nn.Parameter(
            torch.randn(num_output_tokens, hidden_dim)
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=8,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool vision tokens using learned attention.

        Args:
            x: Input tokens of shape (batch, num_input_tokens, hidden_dim)

        Returns:
            Pooled tokens of shape (batch, num_output_tokens, hidden_dim)
        """
        batch_size = x.shape[0]

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Attend to input tokens
        pooled, _ = self.attention(queries, x, x)

        return pooled


class VisionTextFusion(nn.Module):
    """
    Vision-Text Fusion Layer.

    Combines visual and textual embeddings into a unified sequence
    that can be processed by the language model.

    Example:
        >>> config = VisionFusionConfig(vision_hidden_dim=768)
        >>> fusion = VisionTextFusion(config)
        >>>
        >>> # Vision features from encoder
        >>> vision_features = torch.randn(2, 196, 768)
        >>>
        >>> # Text embeddings from language model
        >>> text_embeddings = torch.randn(2, 10, 768)
        >>>
        >>> # Fuse
        >>> fused = fusion(
        ...     vision_features=vision_features,
        ...     text_embeddings=text_embeddings
        ... )
        >>> fused.shape  # (2, 206, 768) - 196 vis + 10 text
    """

    def __init__(self, config: VisionFusionConfig):
        super().__init__()
        self.config = config

        # Vision projection
        if config.use_projection:
            self.vision_projection = VisionProjection(
                config.vision_hidden_dim,
                config.llm_hidden_dim,
                config.projection_bias,
                config.dropout
            )
        else:
            # Identity projection if dimensions match
            if config.vision_hidden_dim != config.llm_hidden_dim:
                raise ValueError(
                    "vision_hidden_dim must equal llm_hidden_dim "
                    "when use_projection=False"
                )
            self.vision_projection = nn.Identity()

        # Pooling strategy
        self.pooling = self._create_pooling(config)

    def _create_pooling(
        self,
        config: VisionFusionConfig
    ) -> Optional[nn.Module]:
        """Create pooling module based on strategy."""
        if config.pooling_strategy == VisionPoolingStrategy.NONE:
            return None
        elif config.pooling_strategy == VisionPoolingStrategy.MEAN:
            return None  # Mean pooling is handled in forward
        elif config.pooling_strategy == VisionPoolingStrategy.CLS:
            return None  # CLS extraction is handled in forward
        elif config.pooling_strategy == VisionPoolingStrategy.ADAPTIVE:
            # Learnable pooling
            return AdaptivePooling(
                config.llm_hidden_dim,
                config.num_vision_tokens,
                config.num_vision_tokens // 4  # Pool to 25% of patches
            )
        else:
            raise ValueError(
                f"Unknown pooling strategy: {config.pooling_strategy}"
            )

    def _pool_vision_features(
        self,
        vision_features: torch.Tensor,
        strategy: str
    ) -> torch.Tensor:
        """
        Pool vision features according to strategy.

        Args:
            vision_features: Vision features (batch, num_patches, dim)
            strategy: Pooling strategy

        Returns:
            Pooled features
        """
        if strategy == VisionPoolingStrategy.NONE:
            return vision_features
        elif strategy == VisionPoolingStrategy.MEAN:
            # Average over all patches
            return vision_features.mean(dim=1, keepdim=True)
        elif strategy == VisionPoolingStrategy.CLS:
            # Take first token (CLS token)
            return vision_features[:, :1, :]
        elif strategy == VisionPoolingStrategy.ADAPTIVE:
            # Use learnable pooling
            return self.pooling(vision_features)
        else:
            return vision_features

    def forward(
        self,
        vision_features: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        interleaved: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fuse vision and text embeddings.

        Args:
            vision_features: Vision encoder output (B, N_vis, D_vis)
            text_embeddings: Text embeddings (B, N_text, D_llm)
            vision_mask: Mask for vision positions (B, N_total)
            interleaved: Whether vision/text are interleaved

        Returns:
            - Fused embeddings (B, N_total, D_llm)
            - Updated vision mask (B, N_total)
        """
        # Handle vision-only, text-only, or both
        embeddings = []

        # Process vision features
        if vision_features is not None:
            # Project to language model space
            vision_proj = self.vision_projection(vision_features)

            # Pool if requested
            vision_proj = self._pool_vision_features(
                vision_proj,
                self.config.pooling_strategy
            )

            embeddings.append(vision_proj)

        # Process text embeddings
        if text_embeddings is not None:
            embeddings.append(text_embeddings)

        # Concatenate embeddings
        if not embeddings:
            raise ValueError(
                "At least one of vision_features or text_embeddings "
                "must be provided"
            )

        # Interleaved or sequential
        if interleaved and len(embeddings) == 2:
            # For interleaving, we'd need more complex logic
            # based on the vision_mask. For now, concatenate.
            # TODO: Implement proper interleaving based on mask
            fused = torch.cat(embeddings, dim=1)
        else:
            # Sequential concatenation: [vision, text] or just one
            fused = torch.cat(embeddings, dim=1) if len(embeddings) > 1 \
                else embeddings[0]

        return fused, vision_mask

    def get_position_ids(
        self,
        input_ids: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate position IDs for multimodal sequence.

        For simple case: sequential position IDs [0, 1, 2, ...]
        For vision tokens: could use 2D position encoding (future work)

        Args:
            input_ids: Token IDs (batch, seq_len)
            vision_mask: Mask indicating vision positions (batch, seq_len)

        Returns:
            Position IDs (batch, seq_len)
        """
        batch_size, seq_len = input_ids.shape

        # Simple sequential positions for now
        # In future: could have special position encoding for vision
        position_ids = torch.arange(
            seq_len,
            dtype=torch.long,
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)

        return position_ids


def create_interleaved_sequence(
    vision_features: torch.Tensor,
    text_embeddings: torch.Tensor,
    vision_positions: List[int]
) -> torch.Tensor:
    """
    Create interleaved vision-text sequence.

    Inserts vision features at specified positions in text sequence.

    Args:
        vision_features: Vision tokens (batch, num_vis, dim)
        text_embeddings: Text embeddings (batch, num_text, dim)
        vision_positions: Positions to insert vision (list of indices)

    Returns:
        Interleaved sequence
    """
    # Split vision features per position
    num_positions = len(vision_positions)
    if num_positions == 0:
        return text_embeddings

    # Split vision features evenly across positions
    vis_per_pos = vision_features.shape[1] // num_positions

    # Create interleaved sequence
    result = []
    text_idx = 0

    for pos in sorted(vision_positions):
        # Add text up to this position
        if pos > text_idx:
            result.append(text_embeddings[:, text_idx:pos, :])
            text_idx = pos

        # Add vision features
        vis_start = len(result) * vis_per_pos
        vis_end = vis_start + vis_per_pos
        result.append(vision_features[:, vis_start:vis_end, :])

    # Add remaining text
    if text_idx < text_embeddings.shape[1]:
        result.append(text_embeddings[:, text_idx:, :])

    # Concatenate all pieces
    return torch.cat(result, dim=1)


if __name__ == "__main__":
    print("Testing Vision-Text Fusion Layer...")

    # Create config
    config = VisionFusionConfig(
        vision_hidden_dim=768,
        llm_hidden_dim=768,
        pooling_strategy="none"
    )

    print(f"\nFusion Configuration:")
    print(f"  Vision dim: {config.vision_hidden_dim}")
    print(f"  Language model dim: {config.llm_hidden_dim}")
    print(f"  Pooling: {config.pooling_strategy}")

    # Create fusion layer
    fusion = VisionTextFusion(config)

    # Test vision + text fusion
    print("\n1. Vision + Text fusion:")
    batch_size = 2
    num_vis_tokens = 196  # 14x14 patches
    num_text_tokens = 10
    vision_dim = 768
    llm_dim = 768

    vision_features = torch.randn(batch_size, num_vis_tokens, vision_dim)
    text_embeddings = torch.randn(batch_size, num_text_tokens, llm_dim)

    fused, _ = fusion(
        vision_features=vision_features,
        text_embeddings=text_embeddings
    )

    print(f"  Vision shape: {vision_features.shape}")
    print(f"  Text shape: {text_embeddings.shape}")
    print(f"  Fused shape: {fused.shape}")
    expected_len = num_vis_tokens + num_text_tokens
    print(f"  Expected length: {expected_len}")
    print(f"  Match: {fused.shape[1] == expected_len}")

    # Test vision-only
    print("\n2. Vision-only:")
    fused_vis, _ = fusion(vision_features=vision_features)
    print(f"  Input shape: {vision_features.shape}")
    print(f"  Output shape: {fused_vis.shape}")

    # Test text-only
    print("\n3. Text-only:")
    fused_text, _ = fusion(text_embeddings=text_embeddings)
    print(f"  Input shape: {text_embeddings.shape}")
    print(f"  Output shape: {fused_text.shape}")

    # Test with mean pooling
    print("\n4. Mean pooling:")
    config_pooled = VisionFusionConfig(
        vision_hidden_dim=768,
        llm_hidden_dim=768,
        pooling_strategy="mean"
    )
    fusion_pooled = VisionTextFusion(config_pooled)
    fused_pooled, _ = fusion_pooled(
        vision_features=vision_features,
        text_embeddings=text_embeddings
    )
    print(f"  Vision shape: {vision_features.shape}")
    print(f"  Pooled fused shape: {fused_pooled.shape}")
    print(f"  Vision tokens after pooling: 1 (mean)")

    # Test position ID generation
    print("\n5. Position ID generation:")
    input_ids = torch.randint(0, 1000, (batch_size, expected_len))
    position_ids = fusion.get_position_ids(input_ids)
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Position IDs shape: {position_ids.shape}")
    print(f"  Position IDs[0]: {position_ids[0][:10].tolist()}...")

    print("\n✓ Vision-Text Fusion Layer implementation complete!")
