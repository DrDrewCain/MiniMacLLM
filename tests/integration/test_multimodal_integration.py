"""
Integration tests for multimodal vision-language architecture.

Tests the complete pipeline:
- Vision encoder → Fusion → LLM
- Multimodal tokenization
- Continual learning with vision
"""

import torch
import pytest
from pathlib import Path

from src.model.llm import ContinualLLM, ModelConfig
from src.model.vision_encoder import VisionEncoder, VisionEncoderConfig
from src.model.vision_fusion import VisionTextFusion, VisionFusionConfig
from src.model.multimodal_attention import (
    MultimodalGroupedQueryAttention,
    MultimodalRoPE,
    MultimodalRoPEConfig
)
from src.tokenization.multimodal_tokenizer import (
    MultimodalTokenizer,
    MultimodalInput
)
from src.continual.multimodal_trainer import (
    MultimodalContinualLearner,
    MultimodalContinualConfig
)


class TestVisionEncoderIntegration:
    """Test vision encoder standalone functionality."""

    @pytest.fixture
    def vision_encoder(self):
        """Create vision encoder for testing."""
        config = VisionEncoderConfig(
            image_size=224,
            patch_size=16,
            hidden_dim=512,
            num_layers=4,
            num_heads=8
        )
        return VisionEncoder(config)

    def test_image_encoding(self, vision_encoder):
        """Test encoding single image."""
        batch_size = 2
        image = torch.randn(batch_size, 3, 224, 224)

        features = vision_encoder(image, is_video=False)

        assert features.shape == (batch_size, 196, 512)
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()

    def test_video_encoding(self, vision_encoder):
        """Test encoding video."""
        batch_size = 1
        num_frames = 8
        video = torch.randn(batch_size, 3, num_frames, 224, 224)

        features = vision_encoder(video, is_video=True)

        # 8 frames * (14*14) patches per frame = 1568 tokens
        assert features.shape[0] == batch_size
        assert features.shape[2] == 512
        assert not torch.isnan(features).any()


class TestMultimodalTokenizerIntegration:
    """Test multimodal tokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create and train tokenizer."""
        tokenizer = MultimodalTokenizer(vocab_size=500, min_frequency=1)
        corpus = ["Hello world", "This is a test", "Multimodal learning"]
        tokenizer.train(corpus, verbose=False)
        return tokenizer

    def test_text_only_encoding(self, tokenizer):
        """Test backward compatibility with text-only."""
        text = "Hello world"
        tokens = tokenizer.encode(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_image_text_encoding(self, tokenizer):
        """Test encoding image + text."""
        image = torch.randn(3, 224, 224)
        encoding = tokenizer.encode_multimodal(
            text="This is a cat",
            images=[image]
        )

        # Should have vision tokens + text tokens
        assert len(encoding.input_ids) > 256  # vision tokens
        assert sum(encoding.vision_mask) == 256  # exactly 256 vision tokens
        assert encoding.vision_features is not None


class TestVisionFusionIntegration:
    """Test vision-text fusion layer."""

    @pytest.fixture
    def fusion_layer(self):
        """Create fusion layer."""
        config = VisionFusionConfig(
            vision_hidden_dim=512,
            llm_hidden_dim=512,
            pooling_strategy="none"
        )
        return VisionTextFusion(config)

    def test_vision_text_fusion(self, fusion_layer):
        """Test fusing vision and text embeddings."""
        batch_size = 2
        vision_features = torch.randn(batch_size, 196, 512)
        text_embeddings = torch.randn(batch_size, 20, 512)

        fused, _ = fusion_layer(
            vision_features=vision_features,
            text_embeddings=text_embeddings
        )

        # Should concatenate: 196 + 20 = 216
        assert fused.shape == (batch_size, 216, 512)
        assert not torch.isnan(fused).any()


class TestMultimodalRoPEIntegration:
    """Test multimodal position embeddings."""

    @pytest.fixture
    def rope(self):
        """Create multimodal RoPE."""
        config = MultimodalRoPEConfig(
            head_dim=64,
            max_seq_len=128,
            max_image_size=16,
            max_video_len=8
        )
        return MultimodalRoPE(config)

    def test_1d_rope_for_text(self, rope):
        """Test 1D RoPE for text."""
        seq_len = 20
        cos, sin = rope.get_1d_embeddings(seq_len, torch.device("cpu"))

        assert cos.shape == (seq_len, 64)
        assert sin.shape == (seq_len, 64)

    def test_2d_rope_for_images(self, rope):
        """Test 2D RoPE for images."""
        height, width = 14, 14
        cos, sin = rope.get_2d_embeddings(height, width, torch.device("cpu"))

        # Should have height * width positions
        assert cos.shape == (196, 64)
        assert sin.shape == (196, 64)

    def test_3d_rope_for_videos(self, rope):
        """Test 3D RoPE for videos."""
        time, height, width = 4, 7, 7
        cos, sin = rope.get_3d_embeddings(
            time,
            height,
            width,
            torch.device("cpu")
        )

        # Should have time * height * width positions
        # Note: 3D embeddings split head_dim across 3 dimensions
        assert cos.shape[0] == time * height * width
        assert sin.shape[0] == time * height * width
        # Total dim is sum of t, h, w dimensions (each gets head_dim // 3)
        assert cos.shape[1] == rope.dim_t + rope.dim_h_3d + rope.dim_w_3d


class TestMultimodalContinualLearnerIntegration:
    """Test complete multimodal continual learning system."""

    @pytest.fixture
    def setup_models(self):
        """Create all models needed for testing."""
        # Small models for fast testing
        llm_config = ModelConfig(
            vocab_size=500,
            d_model=256,
            num_layers=2,
            num_query_heads=4,
            num_kv_heads=2,
            max_seq_len=64
        )
        llm = ContinualLLM(llm_config)

        vision_config = VisionEncoderConfig(
            image_size=224,
            patch_size=16,
            hidden_dim=256,
            num_layers=2,
            num_heads=4
        )
        vision_encoder = VisionEncoder(vision_config)

        config = MultimodalContinualConfig(
            lora_r=4,
            vision_lora_r=4,
            enable_vision=True,
            batch_size=1,
            device="cpu",
            replay_buffer_size=10
        )

        return llm, vision_encoder, config

    def test_multimodal_learner_creation(self, setup_models):
        """Test creating multimodal learner."""
        llm, vision_encoder, config = setup_models

        learner = MultimodalContinualLearner(
            base_model=llm,
            vision_encoder=vision_encoder,
            config=config
        )

        assert learner.vision_encoder is not None
        assert learner.vision_lora_model is not None
        assert learner.fusion_layer is not None

    def test_vision_encoding_in_learner(self, setup_models):
        """Test vision encoding through learner."""
        llm, vision_encoder, config = setup_models

        learner = MultimodalContinualLearner(
            base_model=llm,
            vision_encoder=vision_encoder,
            config=config
        )

        images = [torch.randn(3, 224, 224)]
        features = learner.encode_vision(images=images)

        assert features is not None
        assert features.shape[1] == 196  # 14x14 patches
        assert features.shape[2] == 256  # hidden_dim

    def test_domain_management(self, setup_models):
        """Test adding and switching domains."""
        llm, vision_encoder, config = setup_models

        learner = MultimodalContinualLearner(
            base_model=llm,
            vision_encoder=vision_encoder,
            config=config
        )

        # Add domains
        learner.add_domain_adapter("medical")
        learner.add_domain_adapter("satellite")

        assert "medical" in learner.vision_domains
        assert "satellite" in learner.vision_domains

        # Switch domain
        learner.set_active_domain("medical")
        assert learner.active_vision_domain == "medical"


class TestBrainInspiredFeatures:
    """Test brain-inspired vision processing features."""

    @pytest.fixture
    def vision_encoder_with_merging(self):
        """Create vision encoder with hierarchical compression enabled."""
        config = VisionEncoderConfig(
            image_size=224,
            patch_size=16,
            hidden_dim=256,
            num_layers=12,
            num_heads=8,
            merge_depths=[3, 7, 11],  # Hierarchical compression at these depths
            merge_size=2,
            enable_position_interpolation=True
        )
        return VisionEncoder(config)

    def test_temporal_spatial_integration(self):
        """Test unified temporal-spatial processing (brain-inspired)."""
        config = VisionEncoderConfig(
            image_size=224,
            patch_size=16,
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            temporal_patch_size=1  # 1 frame for images
        )
        encoder = VisionEncoder(config)

        # Test image (treated as 1-frame video)
        image = torch.randn(1, 3, 224, 224)
        features = encoder(image, is_video=False)
        assert features.shape == (1, 196, 256)

        # Test video (multiple frames)
        video = torch.randn(1, 3, 8, 224, 224)
        features = encoder(video, is_video=True)
        # 8 frames * 196 patches = 1568 patches
        assert features.shape[0] == 1
        assert features.shape[2] == 256

    def test_hierarchical_compression(self, vision_encoder_with_merging):
        """Test hierarchical feature compression (V1→V2→V4→IT-like)."""
        encoder = vision_encoder_with_merging

        # Image encoding should apply merging
        image = torch.randn(2, 3, 224, 224)
        features = encoder(image, is_video=False)

        # With merging at depths [3, 7, 11]:
        # Start: 14x14 = 196 patches
        # After depth 3: 7x7 = 49 patches (2x2 merge)
        # After depth 7: 3x3 = 9 patches (2x2 merge, rounded down)
        # After depth 11: 1x1 = 1 patch (2x2 merge)
        # Final output should have significantly fewer tokens
        assert features.shape[0] == 2
        assert features.shape[1] < 196  # Reduced from original
        assert features.shape[2] == 256

    def test_adaptive_receptive_fields(self):
        """Test adaptive receptive fields for variable resolutions."""
        # Train on 224x224
        config = VisionEncoderConfig(
            image_size=224,
            patch_size=16,
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            merge_depths=[],  # No merging for this test
            enable_position_interpolation=True
        )
        encoder = VisionEncoder(config)

        # Test on standard resolution
        image_224 = torch.randn(1, 3, 224, 224)
        features_224 = encoder(image_224, is_video=False)
        assert features_224.shape == (1, 196, 256)  # 14x14 patches

        # Test on higher resolution (should interpolate positions)
        # Note: Input must still be divisible by patch_size
        image_448 = torch.randn(1, 3, 448, 448)
        features_448 = encoder(image_448, is_video=False)
        # 448/16 = 28, so 28x28 = 784 patches
        assert features_448.shape == (1, 784, 256)

        # Both should produce valid features
        assert not torch.isnan(features_224).any()
        assert not torch.isnan(features_448).any()

    def test_unified_embedding_backward_compat(self):
        """Test that unified embedding is backward compatible."""
        config = VisionEncoderConfig(
            image_size=224,
            patch_size=16,
            hidden_dim=512,
            num_layers=4,
            num_heads=8
        )
        encoder = VisionEncoder(config)

        # Should work exactly like before for images
        image = torch.randn(2, 3, 224, 224)
        features = encoder(image, is_video=False)
        assert features.shape == (2, 196, 512)

        # Should work exactly like before for videos
        video = torch.randn(1, 3, 16, 224, 224)
        features = encoder(video, is_video=True)
        assert features.shape[0] == 1
        assert features.shape[2] == 512


@pytest.mark.slow
class TestEndToEndMultimodal:
    """End-to-end multimodal pipeline tests."""

    def test_vision_to_text_pipeline(self):
        """Test complete vision → LLM pipeline."""
        # Create minimal models
        llm_config = ModelConfig(
            vocab_size=100,
            d_model=128,
            num_layers=1,
            num_query_heads=2,
            num_kv_heads=1,
            max_seq_len=32
        )
        llm = ContinualLLM(llm_config)

        vision_config = VisionEncoderConfig(
            image_size=224,
            patch_size=16,
            hidden_dim=128,
            num_layers=1,
            num_heads=2
        )
        vision_encoder = VisionEncoder(vision_config)

        fusion_config = VisionFusionConfig(
            vision_hidden_dim=128,
            llm_hidden_dim=128,
            pooling_strategy="mean"  # Pool to 1 token for simplicity
        )
        fusion = VisionTextFusion(fusion_config)

        # Encode image
        image = torch.randn(1, 3, 224, 224)
        vision_features = vision_encoder(image, is_video=False)

        # Get text embeddings (dummy)
        text_ids = torch.randint(0, 100, (1, 5))
        text_embeddings = llm.token_embedding(text_ids)

        # Fuse
        fused, _ = fusion(
            vision_features=vision_features,
            text_embeddings=text_embeddings
        )

        # Pass through LLM (just first layer for test)
        cos, sin = llm.rope(fused.shape[1], fused.device)
        output, _ = llm.layers[0](fused, cos=cos, sin=sin)

        assert output.shape == fused.shape
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
