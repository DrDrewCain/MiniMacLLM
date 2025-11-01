"""
Multimodal Attention with 2D/3D Rotary Position Embeddings.

Extends GroupedQueryAttention to handle:
- 1D RoPE for text (sequential positions)
- 2D RoPE for images (height, width positions)
- 3D RoPE for videos (time, height, width positions)

This unified attention mechanism enables the language model to process mixed
text, image, and video inputs while preserving spatial/temporal
structure in the position encodings.

References:
    - "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    - Multimodal position encoding with dimensional specialization
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass

from .attention import GroupedQueryAttention, AttentionCache
from .embeddings import rotate_half


@dataclass
class MultimodalRoPEConfig:
    """
    Configuration for multimodal RoPE.

    Args:
        head_dim: Dimension of attention head
        max_seq_len: Maximum sequence length (text)
        max_image_size: Maximum image size (height, width)
        max_video_len: Maximum video temporal length
        base: Base for geometric progression
    """
    head_dim: int = 64
    max_seq_len: int = 2048
    max_image_size: int = 32  # 32x32 patches max
    max_video_len: int = 32  # 32 frames max
    base: float = 10000.0


class MultimodalRoPE(nn.Module):
    """
    Multimodal Rotary Position Embeddings.

    Supports:
    - 1D RoPE for text (sequential)
    - 2D RoPE for images (spatial grid)
    - 3D RoPE for videos (temporal + spatial grid)

    Implementation strategy:
    - Split head_dim into chunks for each dimension
    - For 1D: use full head_dim
    - For 2D: split into [height_dim, width_dim]
    - For 3D: split into [time_dim, height_dim, width_dim]
    """

    def __init__(self, config: MultimodalRoPEConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.base = config.base

        # Precompute inverse frequencies for each dimension
        # For 2D/3D, we split the head_dim across dimensions
        self._precompute_freqs_1d(config.max_seq_len)
        self._precompute_freqs_2d(config.max_image_size)
        self._precompute_freqs_3d(
            config.max_video_len,
            config.max_image_size
        )

    def _precompute_freqs_1d(self, max_len: int):
        """Precompute 1D RoPE frequencies for text."""
        # Standard RoPE for sequential positions
        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(0, self.head_dim, 2).float() / self.head_dim
            )
        )
        self.register_buffer("inv_freq_1d", inv_freq, persistent=False)

        # Precompute cos/sin
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_1d", emb.cos(), persistent=False)
        self.register_buffer("sin_1d", emb.sin(), persistent=False)

    def _precompute_freqs_2d(self, max_size: int):
        """Precompute 2D RoPE frequencies for images."""
        # Split head_dim into height and width components
        dim_h = self.head_dim // 2
        dim_w = self.head_dim - dim_h

        # Height frequencies
        inv_freq_h = 1.0 / (
            self.base ** (torch.arange(0, dim_h, 2).float() / dim_h)
        )

        # Width frequencies
        inv_freq_w = 1.0 / (
            self.base ** (torch.arange(0, dim_w, 2).float() / dim_w)
        )

        self.register_buffer("inv_freq_2d_h", inv_freq_h, persistent=False)
        self.register_buffer("inv_freq_2d_w", inv_freq_w, persistent=False)

        # Precompute for grid
        grid_h = torch.arange(max_size, dtype=torch.float32)
        grid_w = torch.arange(max_size, dtype=torch.float32)

        freqs_h = torch.outer(grid_h, inv_freq_h)
        freqs_w = torch.outer(grid_w, inv_freq_w)

        # Expand and concatenate
        emb_h = torch.cat([freqs_h, freqs_h], dim=-1)
        emb_w = torch.cat([freqs_w, freqs_w], dim=-1)

        self.register_buffer("cos_2d_h", emb_h.cos(), persistent=False)
        self.register_buffer("sin_2d_h", emb_h.sin(), persistent=False)
        self.register_buffer("cos_2d_w", emb_w.cos(), persistent=False)
        self.register_buffer("sin_2d_w", emb_w.sin(), persistent=False)

        self.dim_h = dim_h
        self.dim_w = dim_w

    def _precompute_freqs_3d(self, max_time: int, max_size: int):
        """Precompute 3D RoPE frequencies for videos."""
        # Split head_dim into time, height, width
        # Each dimension gets half for real/complex (will be doubled)
        # Validate head_dim is even and large enough for 3D split
        if self.head_dim < 6:
            raise ValueError(f"head_dim ({self.head_dim}) must be at least 6 for 3D RoPE")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")

        # Calculate initial dimension split
        dim_t_half = max(1, self.head_dim // 6)
        dim_h_half = max(1, self.head_dim // 6)
        # Remaining dims go to width
        dim_w_half = max(1, (self.head_dim // 2) - dim_t_half - dim_h_half)

        # Validate the split before any buffer registration
        total_half = dim_t_half + dim_h_half + dim_w_half
        if total_half * 2 != self.head_dim:
            # Adjust width to make dimensions sum correctly
            dim_w_half = (self.head_dim // 2) - dim_t_half - dim_h_half
            if dim_w_half <= 0:
                raise ValueError(
                    f"head_dim ({self.head_dim}) too small for 3D RoPE split "
                    f"with dim_t={dim_t_half*2}, dim_h={dim_h_half*2}"
                )

        # Time frequencies
        inv_freq_t = 1.0 / (
            self.base ** (
                torch.arange(0, dim_t_half * 2, 2).float() / (dim_t_half * 2)
            )
        )

        # Height frequencies
        inv_freq_h = 1.0 / (
            self.base ** (
                torch.arange(0, dim_h_half * 2, 2).float() / (dim_h_half * 2)
            )
        )

        # Width frequencies
        inv_freq_w = 1.0 / (
            self.base ** (
                torch.arange(0, dim_w_half * 2, 2).float() / (dim_w_half * 2)
            )
        )

        self.register_buffer("inv_freq_3d_t", inv_freq_t, persistent=False)
        self.register_buffer("inv_freq_3d_h", inv_freq_h, persistent=False)
        self.register_buffer("inv_freq_3d_w", inv_freq_w, persistent=False)

        # Precompute
        grid_t = torch.arange(max_time, dtype=torch.float32)
        grid_h = torch.arange(max_size, dtype=torch.float32)
        grid_w = torch.arange(max_size, dtype=torch.float32)

        freqs_t = torch.outer(grid_t, inv_freq_t)
        freqs_h = torch.outer(grid_h, inv_freq_h)
        freqs_w = torch.outer(grid_w, inv_freq_w)

        # Expand (this doubles the dimension)
        emb_t = torch.cat([freqs_t, freqs_t], dim=-1)
        emb_h = torch.cat([freqs_h, freqs_h], dim=-1)
        emb_w = torch.cat([freqs_w, freqs_w], dim=-1)

        self.register_buffer("cos_3d_t", emb_t.cos(), persistent=False)
        self.register_buffer("sin_3d_t", emb_t.sin(), persistent=False)
        self.register_buffer("cos_3d_h", emb_h.cos(), persistent=False)
        self.register_buffer("sin_3d_h", emb_h.sin(), persistent=False)
        self.register_buffer("cos_3d_w", emb_w.cos(), persistent=False)
        self.register_buffer("sin_3d_w", emb_w.sin(), persistent=False)

        # Store final dimensions (validation already done above)
        self.dim_t = dim_t_half * 2
        self.dim_h_3d = dim_h_half * 2
        self.dim_w_3d = dim_w_half * 2

    def get_1d_embeddings(
        self,
        seq_len: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get 1D RoPE embeddings for text."""
        return (
            self.cos_1d[:seq_len].to(device),
            self.sin_1d[:seq_len].to(device)
        )

    def get_2d_embeddings(
        self,
        height: int,
        width: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get 2D RoPE embeddings for images.

        Returns embeddings for a height x width grid.

        Args:
            height: Image height in patches
            width: Image width in patches
            device: Device

        Returns:
            - cos embeddings (height * width, head_dim)
            - sin embeddings (height * width, head_dim)
        """
        # Get height and width embeddings
        cos_h = self.cos_2d_h[:height].to(device)  # (H, dim_h)
        sin_h = self.sin_2d_h[:height].to(device)
        cos_w = self.cos_2d_w[:width].to(device)   # (W, dim_w)
        sin_w = self.sin_2d_w[:width].to(device)

        # Create meshgrid
        # For each position (i,j), concatenate height[i] and width[j]
        cos_h_grid = cos_h.unsqueeze(1).expand(-1, width, -1)
        sin_h_grid = sin_h.unsqueeze(1).expand(-1, width, -1)
        cos_w_grid = cos_w.unsqueeze(0).expand(height, -1, -1)
        sin_w_grid = sin_w.unsqueeze(0).expand(height, -1, -1)

        # Concatenate height and width embeddings
        cos_2d = torch.cat([cos_h_grid, cos_w_grid], dim=-1)
        sin_2d = torch.cat([sin_h_grid, sin_w_grid], dim=-1)

        # Flatten spatial dimensions
        cos_2d = cos_2d.reshape(-1, self.head_dim)  # (H*W, head_dim)
        sin_2d = sin_2d.reshape(-1, self.head_dim)

        return cos_2d, sin_2d

    def get_3d_embeddings(
        self,
        time: int,
        height: int,
        width: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get 3D RoPE embeddings for videos.

        Args:
            time: Number of frames
            height: Frame height in patches
            width: Frame width in patches
            device: Device

        Returns:
            - cos embeddings (time * height * width, head_dim)
            - sin embeddings (time * height * width, head_dim)
        """
        # Get temporal, height, width embeddings
        cos_t = self.cos_3d_t[:time].to(device)
        sin_t = self.sin_3d_t[:time].to(device)
        cos_h = self.cos_3d_h[:height].to(device)
        sin_h = self.sin_3d_h[:height].to(device)
        cos_w = self.cos_3d_w[:width].to(device)
        sin_w = self.sin_3d_w[:width].to(device)

        # Create 3D grid
        cos_t_grid = cos_t.view(time, 1, 1, -1).expand(-1, height, width, -1)
        sin_t_grid = sin_t.view(time, 1, 1, -1).expand(-1, height, width, -1)
        cos_h_grid = cos_h.view(1, height, 1, -1).expand(time, -1, width, -1)
        sin_h_grid = sin_h.view(1, height, 1, -1).expand(time, -1, width, -1)
        cos_w_grid = cos_w.view(1, 1, width, -1).expand(time, height, -1, -1)
        sin_w_grid = sin_w.view(1, 1, width, -1).expand(time, height, -1, -1)

        # Concatenate time, height, width
        cos_3d = torch.cat([cos_t_grid, cos_h_grid, cos_w_grid], dim=-1)
        sin_3d = torch.cat([sin_t_grid, sin_h_grid, sin_w_grid], dim=-1)

        # Flatten spatio-temporal dimensions
        total_dim = self.dim_t + self.dim_h_3d + self.dim_w_3d
        cos_3d = cos_3d.reshape(-1, total_dim)  # (T*H*W, total_dim)
        sin_3d = sin_3d.reshape(-1, total_dim)

        return cos_3d, sin_3d


def apply_multimodal_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    modality_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to Q and K with modality-aware positioning.

    Args:
        q: Query tensor (batch, num_heads, seq_len, head_dim)
        k: Key tensor (batch, num_heads, seq_len, head_dim)
        cos: Cosine embeddings (seq_len, head_dim)
        sin: Sine embeddings (seq_len, head_dim)
        modality_mask: Optional mask indicating modality type per position

    Returns:
        Rotated (q, k) tensors
    """
    # Reshape cos/sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class MultimodalGroupedQueryAttention(GroupedQueryAttention):
    """
    Grouped Query Attention with multimodal RoPE support.

    Extends the base GQA to handle mixed text/image/video sequences
    with appropriate position encodings for each modality.
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 2048,
        max_image_size: int = 32,
        max_video_len: int = 32
    ):
        super().__init__(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            bias=bias,
            max_seq_len=max_seq_len
        )

        # Create multimodal RoPE
        rope_config = MultimodalRoPEConfig(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            max_image_size=max_image_size,
            max_video_len=max_video_len
        )
        self.multimodal_rope = MultimodalRoPE(rope_config)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[AttentionCache] = None,
        use_cache: bool = False,
        modality_types: Optional[torch.Tensor] = None,
        spatial_shape: Optional[Tuple[int, int]] = None,
        temporal_shape: Optional[Tuple[int, int, int]] = None
    ) -> Tuple[torch.Tensor, Optional[AttentionCache]]:
        """
        Forward pass with multimodal position encoding.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            cos: Cosine embeddings (optional, computed if None)
            sin: Sine embeddings (optional, computed if None)
            mask: Attention mask
            cache: KV cache
            use_cache: Return cache
            modality_types: Type of each position (0=text, 1=image, 2=video)
            spatial_shape: (height, width) for images
            temporal_shape: (time, height, width) for videos

        Returns:
            - Output tensor
            - Updated cache
        """
        batch_size, seq_len, _ = x.shape

        # If cos/sin not provided, compute based on modality
        if cos is None or sin is None:
            # Determine modality from input
            if temporal_shape is not None:
                # 3D RoPE for video
                t, h, w = temporal_shape
                cos, sin = self.multimodal_rope.get_3d_embeddings(
                    t, h, w, x.device
                )
            elif spatial_shape is not None:
                # 2D RoPE for image
                h, w = spatial_shape
                cos, sin = self.multimodal_rope.get_2d_embeddings(
                    h, w, x.device
                )
            else:
                # 1D RoPE for text (default)
                cos, sin = self.multimodal_rope.get_1d_embeddings(
                    seq_len, x.device
                )

        # Call parent forward with computed embeddings
        return super().forward(
            x, cos, sin, mask, cache, use_cache
        )


if __name__ == "__main__":
    print("Testing Multimodal Attention...")

    # Create config
    config = MultimodalRoPEConfig(
        head_dim=64,
        max_seq_len=128,
        max_image_size=16,
        max_video_len=8
    )

    print(f"\nMultimodal RoPE Configuration:")
    print(f"  Head dim: {config.head_dim}")
    print(f"  Max seq len: {config.max_seq_len}")
    print(f"  Max image size: {config.max_image_size}")

    # Create RoPE
    rope = MultimodalRoPE(config)

    # Test 1D (text)
    print("\n1. Testing 1D RoPE (text):")
    seq_len = 10
    cos_1d, sin_1d = rope.get_1d_embeddings(seq_len, torch.device("cpu"))
    print(f"  Sequence length: {seq_len}")
    print(f"  Cos shape: {cos_1d.shape}")
    print(f"  Sin shape: {sin_1d.shape}")

    # Test 2D (image)
    print("\n2. Testing 2D RoPE (image):")
    height, width = 14, 14  # 14x14 patches
    cos_2d, sin_2d = rope.get_2d_embeddings(
        height,
        width,
        torch.device("cpu")
    )
    print(f"  Image size: {height}x{width}")
    print(f"  Cos shape: {cos_2d.shape}")
    print(f"  Expected: ({height * width}, {config.head_dim})")

    # Test 3D (video)
    print("\n3. Testing 3D RoPE (video):")
    time, height, width = 8, 14, 14
    cos_3d, sin_3d = rope.get_3d_embeddings(
        time,
        height,
        width,
        torch.device("cpu")
    )
    print(f"  Video shape: {time}x{height}x{width}")
    print(f"  Cos shape: {cos_3d.shape}")
    print(f"  Expected: ({time * height * width}, {config.head_dim})")

    # Test multimodal attention
    print("\n4. Testing Multimodal Attention:")
    batch_size = 2
    seq_len = 20
    d_model = 512
    num_query_heads = 8
    num_kv_heads = 2

    attention = MultimodalGroupedQueryAttention(
        d_model=d_model,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=128,
        max_image_size=16
    )

    x = torch.randn(batch_size, seq_len, d_model)

    # Text mode (1D)
    print("  Text mode (1D RoPE):")
    output, _ = attention(x)
    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {output.shape}")

    # Image mode (2D)
    print("  Image mode (2D RoPE):")
    image_seq_len = 196  # 14x14 patches
    x_image = torch.randn(batch_size, image_seq_len, d_model)
    output_image, _ = attention(
        x_image,
        spatial_shape=(14, 14)
    )
    print(f"    Input shape: {x_image.shape}")
    print(f"    Output shape: {output_image.shape}")

    print("\n✓ Multimodal Attention implementation complete!")
