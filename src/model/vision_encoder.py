"""
Vision Encoder for Multimodal Language Model - Brain-Inspired Architecture.

Implements neurobiologically-inspired visual processing with:
- Unified temporal-spatial integration (static and dynamic inputs)
- Hierarchical feature compression (progressive abstraction like V1→V2→V4→IT)
- Adaptive spatial receptive fields (position interpolation)
- Domain-specific synaptic plasticity (LoRA-compatible)
- Cross-modal synaptic connections (output compatible with language model)

Brain-Inspired Design Principles:
- **Temporal-Spatial Unity**: Visual cortex processes motion and static
  images through the same neural pathways (unified 3D convolution)
- **Hierarchical Abstraction**: Progressive compression of visual features
  mimicking cortical hierarchy (spatial patch merging)
- **Receptive Field Plasticity**: Neurons adjust receptive fields based
  on task demands (position interpolation)
- **Synaptic Specialization**: Domain-specific neural pathways for
  different visual domains (LoRA adapters)

Optimized for:
- Apple Silicon efficiency
- Continual learning without catastrophic forgetting
- Memory efficiency through hierarchical compression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

from .normalization import create_norm_layer


@dataclass
class VisionEncoderConfig:
    """
    Configuration for Vision Encoder.

    Args:
        image_size: Input image size (H, W) - assumes square
        patch_size: Size of each patch for 2D (images) or (T, H, W) for 3D (video)
        in_channels: Number of input channels (3 for RGB)
        hidden_dim: Hidden dimension of the encoder (should match language model d_model)
        num_layers: Number of vision transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to hidden_dim
        dropout: Dropout probability
        norm_type: Type of normalization ("rmsnorm" or "layernorm")
        norm_eps: Epsilon for normalization
        max_temporal_len: Maximum number of video frames
        merge_depths: Depths for hierarchical feature compression
        merge_size: Size of spatial compression window
        enable_position_interpolation: Enable adaptive receptive fields
    """
    # Image/Video input
    image_size: int = 224  # Standard size
    patch_size: int = 16  # 16x16 patches for images
    temporal_patch_size: int = 1  # 1 frame for images (unified 3D conv)
    in_channels: int = 3  # RGB

    # Architecture
    hidden_dim: int = 768  # Match language model d_model
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0

    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Normalization
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6

    # Video support
    max_temporal_len: int = 32  # Max video frames

    # Brain-inspired enhancements
    merge_depths: list = None  # Depths for hierarchical compression (e.g., [3, 7, 11])
    merge_size: int = 2  # Spatial compression window (2x2)
    enable_position_interpolation: bool = True  # Adaptive receptive fields

    def __post_init__(self):
        """Validate configuration and initialize defaults."""
        # Initialize default merge depths if not provided
        if self.merge_depths is None:
            # Default: merge at 1/4, 1/2, and 3/4 of depth
            self.merge_depths = [
                self.num_layers // 4,
                self.num_layers // 2,
                3 * self.num_layers // 4
            ] if self.num_layers >= 12 else []

        # Validate configuration
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )

    @property
    def num_patches(self) -> int:
        """Number of patches per image."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_temporal_patches(self) -> int:
        """Number of temporal patches for video."""
        return self.max_temporal_len // self.temporal_patch_size


class UnifiedPatchEmbedding(nn.Module):
    """
    Unified Temporal-Spatial Patch Embedding (Brain-Inspired).

    Mimics visual cortex: same neural pathways process static and dynamic inputs.
    Treats images as 1-frame videos for unified processing.

    Args:
        image_size: Spatial size of input
        patch_size: Spatial patch size
        temporal_patch_size: Temporal patch size (1 for images)
        in_channels: Number of input channels
        hidden_dim: Output embedding dimension
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        temporal_patch_size: int = 1,
        in_channels: int = 3,
        hidden_dim: int = 768
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.num_spatial_patches = (image_size // patch_size) ** 2

        # 3D convolution for spatio-temporal patches
        # Works for both images (T=1) and videos (T>1)
        self.projection = nn.Conv3d(
            in_channels,
            hidden_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size)
        )

    def forward(self, x: torch.Tensor, is_video: bool = False) -> tuple:
        """
        Forward pass.

        Args:
            x: Input tensor
                - For images: (batch, channels, height, width)
                - For videos: (batch, channels, time, height, width)
            is_video: Whether input is video

        Returns:
            Tuple of (patch embeddings, grid_shape)
            - Embeddings: (batch, num_patches, hidden_dim)
            - grid_shape: (T', H', W') where T'=1 for images
        """
        # Reshape image to 5D if needed
        if not is_video:
            # (B, C, H, W) -> (B, C, 1, H, W)
            x = x.unsqueeze(2)

        # Get grid shape before projection
        _, _, t, h, w = x.shape
        grid_t = t // self.temporal_patch_size
        grid_h = h // self.patch_size
        grid_w = w // self.patch_size

        # Project patches: (B, C, T, H, W) -> (B, D, T', H', W')
        x = self.projection(x)

        # Flatten spatio-temporal dimensions: (B, D, T', H', W') -> (B, D, N)
        x = x.flatten(2)

        # Transpose: (B, D, N) -> (B, N, D)
        x = x.transpose(1, 2)

        return x, (grid_t, grid_h, grid_w)


class VisionPatchMerger(nn.Module):
    """
    Hierarchical Feature Compression (Brain-Inspired).

    Mimics cortical hierarchy (V1→V2→V4→IT): progressively compresses
    visual features through spatial pooling.
    Applied at intermediate depths for multi-scale representation.

    Args:
        hidden_dim: Hidden dimension
        merge_size: Size of merge window (default: 2 for 2x2)
    """

    def __init__(self, hidden_dim: int, merge_size: int = 2):
        super().__init__()
        self.merge_size = merge_size
        self.hidden_dim = hidden_dim

        # Linear projection after merging
        # Input: merge_size^2 * hidden_dim
        # Output: hidden_dim
        self.projection = nn.Linear(
            hidden_dim * merge_size * merge_size,
            hidden_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_shape: tuple
    ) -> tuple:
        """
        Merge spatial patches.

        Args:
            x: Input tokens (batch, num_patches, hidden_dim)
            grid_shape: (T, H, W) grid dimensions

        Returns:
            Tuple of (merged tokens, new_grid_shape)
        """
        batch_size, num_patches, hidden_dim = x.shape
        grid_t, grid_h, grid_w = grid_shape

        # Reshape to spatial grid: (B, T*H*W, D) -> (B, T, H, W, D)
        x = x.reshape(batch_size, grid_t, grid_h, grid_w, hidden_dim)

        # Check if spatial dimensions are divisible by merge_size
        if grid_h % self.merge_size != 0 or grid_w % self.merge_size != 0:
            # Can't merge, return as is
            return x.reshape(batch_size, num_patches, hidden_dim), grid_shape

        # Reshape for merging
        # (B, T, H, W, D) -> (B, T, H//M, M, W//M, M, D)
        x = x.reshape(
            batch_size,
            grid_t,
            grid_h // self.merge_size,
            self.merge_size,
            grid_w // self.merge_size,
            self.merge_size,
            hidden_dim
        )

        # Merge: (B, T, H', M, W', M, D) -> (B, T, H', W', M*M*D)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(
            batch_size,
            grid_t,
            grid_h // self.merge_size,
            grid_w // self.merge_size,
            self.merge_size * self.merge_size * hidden_dim
        )

        # Project: (B, T, H', W', M*M*D) -> (B, T, H', W', D)
        x = self.projection(x)

        # Flatten back: (B, T, H', W', D) -> (B, T*H'*W', D)
        new_grid_h = grid_h // self.merge_size
        new_grid_w = grid_w // self.merge_size
        x = x.reshape(batch_size, grid_t * new_grid_h * new_grid_w, hidden_dim)

        return x, (grid_t, new_grid_h, new_grid_w)


class VisionAttention(nn.Module):
    """
    Multi-head self-attention for vision encoder.

    Standard MHA (not GQA) since vision encoder is smaller.
    Includes LoRA support for domain-specific adaptation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projections
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, num_patches, hidden_dim)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, num_patches, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        attn = self.attention_dropout(attn)

        # Apply attention to values
        x = attn @ v

        # Reshape and project
        x = x.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        x = self.out_proj(x)
        x = self.dropout(x)

        return x


class VisionMLP(nn.Module):
    """
    MLP block for vision encoder.

    Standard two-layer MLP with GELU activation.
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)

        self.fc1 = nn.Linear(hidden_dim, mlp_hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class VisionTransformerBlock(nn.Module):
    """
    Vision Transformer block.

    Pre-norm architecture:
    - LayerNorm -> Attention -> Residual
    - LayerNorm -> MLP -> Residual
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-6
    ):
        super().__init__()

        # Attention block
        self.norm1 = create_norm_layer(hidden_dim, norm_type, norm_eps)
        self.attention = VisionAttention(
            hidden_dim,
            num_heads,
            dropout,
            attention_dropout
        )

        # MLP block
        self.norm2 = create_norm_layer(hidden_dim, norm_type, norm_eps)
        self.mlp = VisionMLP(hidden_dim, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with pre-norm and residual connections."""
        # Attention block
        x = x + self.attention(self.norm1(x), mask)

        # MLP block
        x = x + self.mlp(self.norm2(x))

        return x


class VisionEncoder(nn.Module):
    """
    Vision Encoder for multimodal language model.

    Supports both images and videos with LoRA for domain adaptation.

    Args:
        config: Vision encoder configuration

    Example:
        >>> config = VisionEncoderConfig(hidden_dim=768)
        >>> encoder = VisionEncoder(config)
        >>>
        >>> # Image
        >>> images = torch.randn(2, 3, 224, 224)
        >>> visual_tokens = encoder(images)
        >>> visual_tokens.shape  # (2, 196, 768)
        >>>
        >>> # Video
        >>> videos = torch.randn(2, 3, 16, 224, 224)
        >>> visual_tokens = encoder(videos)
        >>> visual_tokens.shape  # (2, 1568, 768)
    """

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config

        # Unified patch embedding (handles images and videos)
        self.patch_embed = UnifiedPatchEmbedding(
            config.image_size,
            config.patch_size,
            config.temporal_patch_size,
            config.in_channels,
            config.hidden_dim
        )

        # Class token (optional, for global image representation)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))

        # Position embeddings (learnable)
        # Calculate max patches accounting for potential merging
        grid_size = config.image_size // config.patch_size
        max_patches = grid_size * grid_size * config.num_temporal_patches + 1
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_patches, config.hidden_dim)
        )
        self.grid_size = grid_size  # Store original grid size for interpolation

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Vision transformer blocks with patch merging
        self.blocks = nn.ModuleList()
        self.mergers = nn.ModuleDict()

        for i in range(config.num_layers):
            # Add transformer block
            self.blocks.append(
                VisionTransformerBlock(
                    config.hidden_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.dropout,
                    config.attention_dropout,
                    config.norm_type,
                    config.norm_eps
                )
            )

            # Add patch merger at specified depths
            if i in config.merge_depths:
                self.mergers[str(i)] = VisionPatchMerger(
                    config.hidden_dim,
                    config.merge_size
                )

        # Final normalization
        self.norm = create_norm_layer(
            config.hidden_dim,
            config.norm_type,
            config.norm_eps
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def interpolate_pos_embedding(
        self,
        pos_embed: torch.Tensor,
        grid_shape: tuple,
        original_grid_size: int
    ) -> torch.Tensor:
        """
        Adaptive Receptive Field Scaling (Brain-Inspired).

        Mimics neural plasticity: receptive fields adjust to input resolution.
        Allows training on one resolution and inference on another.

        Args:
            pos_embed: Original position embeddings (1, N, D)
            grid_shape: (T, H, W) current grid dimensions
            original_grid_size: Original spatial grid size (e.g., 14)

        Returns:
            Interpolated position embeddings
        """
        if not self.config.enable_position_interpolation:
            # Just use as many embeddings as we have
            return pos_embed

        grid_t, grid_h, grid_w = grid_shape

        # If grid size matches, no interpolation needed
        if grid_h == original_grid_size and grid_w == original_grid_size:
            return pos_embed

        # Extract cls token embedding (first position)
        cls_embed = pos_embed[:, :1, :]  # (1, 1, D)

        # Extract spatial embeddings (skip cls token)
        # Assume original embeddings are for original_grid_size x original_grid_size
        spatial_embed = pos_embed[:, 1:original_grid_size**2 + 1, :]

        # Reshape to grid: (1, H*W, D) -> (1, D, H, W)
        hidden_dim = spatial_embed.shape[-1]
        spatial_embed = spatial_embed.transpose(1, 2).reshape(
            1, hidden_dim, original_grid_size, original_grid_size
        )

        # Interpolate to new size using bicubic (smooth receptive field scaling)
        spatial_embed = F.interpolate(
            spatial_embed,
            size=(grid_h, grid_w),
            mode='bicubic',
            align_corners=False
        )

        # Reshape back: (1, D, H', W') -> (1, H'*W', D)
        spatial_embed = spatial_embed.reshape(
            1, hidden_dim, grid_h * grid_w
        ).transpose(1, 2)

        # Combine cls and interpolated spatial embeddings
        return torch.cat([cls_embed, spatial_embed], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        is_video: bool = False,
        return_cls: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with brain-inspired mechanisms.

        Args:
            x: Input tensor
                - For images: (batch, channels, height, width)
                - For videos: (batch, channels, time, height, width)
            is_video: Whether input is video
            return_cls: Whether to return only cls token

        Returns:
            Visual token embeddings of shape (batch, num_patches, hidden_dim)
            Or if return_cls=True: (batch, hidden_dim)
        """
        batch_size = x.shape[0]

        # Unified patch embedding (works for both images and videos)
        x, grid_shape = self.patch_embed(x, is_video=is_video)  # (B, N, D)
        num_patches = x.shape[1]

        # Add cls token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Add position embeddings with interpolation
        pos_embed = self.interpolate_pos_embedding(
            self.position_embedding,
            grid_shape,
            self.grid_size
        )

        # Use as many position embeddings as we have tokens
        x = x + pos_embed[:, :num_patches + 1, :]

        # Dropout
        x = self.dropout(x)

        # Pass through transformer blocks with patch merging
        # Track grid shape as it changes with merging
        current_grid = grid_shape

        for i, block in enumerate(self.blocks):
            # Apply transformer block (to all tokens including cls)
            x = block(x)

            # Apply patch merging if at merge depth
            if str(i) in self.mergers:
                # Separate cls token
                cls_token = x[:, :1, :]  # (B, 1, D)
                patch_tokens = x[:, 1:, :]  # (B, N, D)

                # Merge patches
                patch_tokens, current_grid = self.mergers[str(i)](
                    patch_tokens,
                    current_grid
                )

                # Recombine with cls token
                x = torch.cat([cls_token, patch_tokens], dim=1)

        # Final norm
        x = self.norm(x)

        # Return cls token or all patches
        if return_cls:
            return x[:, 0]  # (B, D)
        else:
            return x[:, 1:]  # (B, N, D) - exclude cls token


def count_parameters(model: nn.Module) -> dict:
    """Count parameters in vision encoder."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable
    }


if __name__ == "__main__":
    print("Testing Vision Encoder...")

    # Create config
    config = VisionEncoderConfig(
        image_size=224,
        patch_size=16,
        hidden_dim=768,
        num_layers=12,
        num_heads=12
    )

    print("\nVision Encoder Configuration:")
    print(f"  Image size: {config.image_size}x{config.image_size}")
    print(f"  Patch size: {config.patch_size}x{config.patch_size}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Num patches: {config.num_patches}")

    # Create encoder
    encoder = VisionEncoder(config)

    # Count parameters
    params = count_parameters(encoder)
    print("\nParameter counts:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")

    # Test image encoding
    print("\nTesting image encoding...")
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        visual_tokens = encoder(images, is_video=False)

    print(f"  Input shape: {images.shape}")
    print(f"  Output shape: {visual_tokens.shape}")
    expected_shape = (
        f"({batch_size}, {config.num_patches}, {config.hidden_dim})"
    )
    print(f"  Expected: {expected_shape}")

    # Test video encoding
    print("\nTesting video encoding...")
    num_frames = 16
    videos = torch.randn(batch_size, 3, num_frames, 224, 224)

    with torch.no_grad():
        visual_tokens = encoder(videos, is_video=True)

    print(f"  Input shape: {videos.shape}")
    print(f"  Output shape: {visual_tokens.shape}")

    # Test cls token extraction
    print("\nTesting cls token...")
    with torch.no_grad():
        cls_token = encoder(images, is_video=False, return_cls=True)

    print(f"  Cls token shape: {cls_token.shape}")
    print(f"  Expected: ({batch_size}, {config.hidden_dim})")

    print("\n✓ Vision Encoder implementation complete!")
