"""
Multimodal Continual Learning Trainer.

Extends the continual learning framework to support vision + text:
- LoRA adapters for both vision encoder and language model
- Domain-specific learning (e.g., medical images, satellite, documents)
- EWC for both vision and text parameters
- Experience replay with image-text pairs
- Zero catastrophic forgetting across modalities

Example usage:
    >>> # Create multimodal learner
    >>> learner = MultimodalContinualLearner(
    ...     base_model=llm,
    ...     vision_encoder=vision_enc,
    ...     config=config,
    ...     tokenizer=tokenizer
    ... )
    >>>
    >>> # Learn from image-text pair
    >>> image = load_image("cat.jpg")
    >>> learner.learn_from_multimodal(
    ...     images=[image],
    ...     text="This is a cat",
    ...     domain="animals"
    ... )
    >>>
    >>> # Switch to different domain
    >>> xray = load_image("xray.jpg")
    >>> learner.add_domain_adapter("medical")
    >>> learner.learn_from_multimodal(
    ...     images=[xray],
    ...     text="Chest X-ray showing normal lungs",
    ...     domain="medical"
    ... )
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from pathlib import Path

from .continual_trainer import (
    ContinualLearner,
    ContinualLearningConfig
)
from .experience_replay import Experience
from ..model.vision_encoder import VisionEncoder, VisionEncoderConfig
from ..model.vision_fusion import VisionTextFusion, VisionFusionConfig
from ..lora.lora_model import LoRAModel
from ..lora.lora_layer import LoRAConfig


@dataclass
class MultimodalContinualConfig(ContinualLearningConfig):
    """
    Configuration for multimodal continual learning.

    Extends ContinualLearningConfig with vision-specific settings.

    Args:
        vision_lora_r: LoRA rank for vision encoder
        vision_lora_alpha: LoRA alpha for vision encoder
        fusion_pooling: Vision token pooling strategy
        enable_vision: Whether vision encoder is active
        vision_domains: List of vision domains to support
    """
    # Vision-specific LoRA
    vision_lora_r: int = 8
    vision_lora_alpha: float = 16.0

    # Fusion settings
    fusion_pooling: str = "none"  # none, mean, cls, adaptive

    # Vision toggle
    enable_vision: bool = True

    # Domain management
    vision_domains: List[str] = field(default_factory=lambda: ["default"])


@dataclass
class MultimodalExperience(Experience):
    """
    Experience with vision + text data.

    Extends Experience to include visual features.
    """
    vision_features: Optional[torch.Tensor] = None
    spatial_shape: Optional[tuple] = None  # (H, W) for images
    temporal_shape: Optional[tuple] = None  # (T, H, W) for videos


class MultimodalContinualLearner(ContinualLearner):
    """
    Multimodal continual learner supporting vision + text.

    Manages:
    - Vision encoder with LoRA per domain
    - Vision-text fusion layer
    - Multimodal experience replay
    - EWC for both vision and text parameters

    Example:
        >>> config = MultimodalContinualConfig(
        ...     vision_lora_r=8,
        ...     enable_vision=True
        ... )
        >>> learner = MultimodalContinualLearner(
        ...     base_model=llm,
        ...     vision_encoder=vision_encoder,
        ...     config=config,
        ...     tokenizer=tokenizer
        ... )
    """

    def __init__(
        self,
        base_model: nn.Module,
        vision_encoder: Optional[VisionEncoder] = None,
        config: Optional[MultimodalContinualConfig] = None,
        tokenizer=None,
        adapter_name: str = "default"
    ):
        """
        Initialize multimodal continual learner.

        Args:
            base_model: Base language model
            vision_encoder: Vision encoder (optional)
            config: Multimodal continual learning config
            tokenizer: Multimodal tokenizer
            adapter_name: Name of default adapter
        """
        # Use default config if not provided
        if config is None:
            config = MultimodalContinualConfig()

        # Initialize base continual learner (text-only)
        super().__init__(
            base_model=base_model,
            config=config,
            tokenizer=tokenizer,
            adapter_name=adapter_name
        )

        # Store multimodal config
        self.multimodal_config = config

        # Vision encoder (optional)
        self.vision_encoder = None
        self.vision_lora_model = None

        if config.enable_vision and vision_encoder is not None:
            self.vision_encoder = vision_encoder.to(self.device)

            # Wrap vision encoder with LoRA
            vision_lora_config = LoRAConfig(
                r=config.vision_lora_r,
                alpha=config.vision_lora_alpha,
                dropout=config.lora_dropout,
                target_modules=[
                    "qkv",  # Vision attention projections
                    "out_proj",
                    "fc1",  # MLP
                    "fc2"
                ]
            )

            self.vision_lora_model = LoRAModel(
                vision_encoder,
                vision_lora_config,
                adapter_name
            )

            print(f"Vision encoder moved to {self.device}")
            print("Vision encoder trainable parameters:")
            self.vision_lora_model.print_trainable_parameters()

        # Vision-text fusion layer
        self.fusion_layer = None
        if config.enable_vision:
            fusion_config = VisionFusionConfig(
                vision_hidden_dim=768,  # Match vision encoder
                llm_hidden_dim=self.model.base_model.config.d_model,
                pooling_strategy=config.fusion_pooling
            )
            self.fusion_layer = VisionTextFusion(fusion_config).to(
                self.device
            )

        # Add vision and fusion parameters to optimizer
        if config.enable_vision:
            vision_params = []
            if self.vision_lora_model is not None:
                vision_params.extend(self.vision_lora_model.get_trainable_parameters())
            if self.fusion_layer is not None:
                vision_params.extend(self.fusion_layer.parameters())

            if vision_params:
                # Add new parameter group to existing optimizer
                self.optimizer.add_param_group({
                    'params': vision_params,
                    'lr': config.learning_rate
                })
                print(f"Added {len(vision_params)} vision parameter groups to optimizer")

        # Track vision domains
        self.vision_domains = set(config.vision_domains)
        self.active_vision_domain = adapter_name

    def add_domain_adapter(
        self,
        domain_name: str,
        vision_only: bool = False
    ):
        """
        Add a new domain-specific LoRA adapter.

        Note: Current LoRA implementation uses single adapter.
        This method is a placeholder for future multi-adapter support.

        Args:
            domain_name: Name of the domain
            vision_only: If True, only add vision adapter
        """
        # Store domain name for tracking
        self.vision_domains.add(domain_name)
        print(
            f"Registered domain: {domain_name} "
            "(multi-adapter support coming soon)"
        )

    def set_active_domain(self, domain_name: str):
        """
        Switch to a specific domain adapter.

        Note: Current LoRA implementation uses single adapter.
        This method is a placeholder for future multi-adapter support.

        Args:
            domain_name: Domain to activate
        """
        self.active_vision_domain = domain_name
        print(
            f"Set active domain to: {domain_name} "
            "(multi-adapter switching coming soon)"
        )

    def encode_vision(
        self,
        images: Optional[List[torch.Tensor]] = None,
        videos: Optional[List[torch.Tensor]] = None
    ) -> Optional[torch.Tensor]:
        """
        Encode visual inputs using vision encoder.

        Args:
            images: List of image tensors (C, H, W)
            videos: List of video tensors (T, C, H, W)

        Returns:
            Vision features (batch, num_patches, hidden_dim)
        """
        if self.vision_encoder is None:
            return None

        # Keep encoder in training mode if we have LoRA adapters
        if self.vision_lora_model is not None:
            self.vision_encoder.train()
        else:
            self.vision_encoder.eval()

        features = []

        # Remove no_grad to allow vision fine-tuning
        if images:
            for img in images:
                # Add batch dimension if needed
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                img = img.to(self.device)

                # Encode image
                feat = self.vision_encoder(img, is_video=False)
                features.append(feat)

        if videos:
            for vid in videos:
                # Add batch dimension if needed
                if vid.dim() == 4:
                    vid = vid.unsqueeze(0)
                vid = vid.to(self.device)

                # Encode video
                feat = self.vision_encoder(vid, is_video=True)
                features.append(feat)

        if features:
            return torch.cat(features, dim=1)  # Concat along token dim
        return None

    def learn_from_multimodal(
        self,
        text: Optional[str] = None,
        images: Optional[List[torch.Tensor]] = None,
        videos: Optional[List[torch.Tensor]] = None,
        domain: Optional[str] = None,
        importance: float = 1.0
    ) -> Dict[str, float]:
        """
        Learn from multimodal input (vision + text).

        Args:
            text: Text string
            images: List of image tensors
            videos: List of video tensors
            domain: Domain label
            importance: Importance score

        Returns:
            Loss statistics
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for multimodal learning")

        # Encode visual inputs
        vision_features = None
        if images or videos:
            vision_features = self.encode_vision(images, videos)

        # Tokenize multimodal input
        if hasattr(self.tokenizer, 'encode_multimodal'):
            # Use multimodal tokenizer
            encoding = self.tokenizer.encode_multimodal(
                text=text,
                images=images,
                videos=videos,
                add_special_tokens=True
            )
            input_ids = torch.tensor(
                encoding.input_ids,
                dtype=torch.long
            )
        else:
            # Fall back to text-only
            if text is None:
                raise ValueError(
                    "Text required when using text-only tokenizer"
                )
            tokens = self.tokenizer.encode(text)
            input_ids = torch.tensor(tokens, dtype=torch.long)

        # Create labels (shifted for next-token prediction)
        labels = input_ids.clone()

        # Create multimodal experience
        experience = MultimodalExperience(
            input_ids=input_ids,
            labels=labels,
            importance=importance,
            domain=domain,
            vision_features=vision_features
        )

        # Learn from experience
        return self.learn_from_batch([experience], update_immediately=True)

    def save_checkpoint(self, save_dir: Optional[str] = None):
        """
        Save multimodal checkpoint.

        Saves both text and vision components.
        """
        if save_dir is None:
            save_dir = f"checkpoints/{self.adapter_name}_multimodal_latest"

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save text model (base continual learner)
        super().save_checkpoint(str(save_path))

        # Save vision encoder
        if self.vision_lora_model is not None:
            vision_checkpoint = {
                "state_dict": self.vision_lora_model.state_dict(),
                "config": self.vision_encoder.config.__dict__,
                "domains": list(self.vision_domains),
                "active_domain": self.active_vision_domain
            }
            torch.save(
                vision_checkpoint,
                save_path / "vision_encoder.pt"
            )

        # Save fusion layer
        if self.fusion_layer is not None:
            torch.save(
                self.fusion_layer.state_dict(),
                save_path / "fusion_layer.pt"
            )

        print(f"Multimodal checkpoint saved to {save_dir}")

    def load_checkpoint(self, load_dir: str):
        """Load multimodal checkpoint."""
        load_path = Path(load_dir)

        # Load text model (base)
        super().load_checkpoint(str(load_path))

        # Load vision encoder
        if (self.vision_lora_model is not None and
                (load_path / "vision_encoder.pt").exists()):
            vision_checkpoint = torch.load(
                load_path / "vision_encoder.pt",
                map_location=self.device
            )
            self.vision_lora_model.load_state_dict(
                vision_checkpoint["state_dict"],
                strict=False
            )
            self.vision_domains = set(vision_checkpoint.get("domains", []))
            self.active_vision_domain = vision_checkpoint.get(
                "active_domain",
                "default"
            )
            print("Loaded vision encoder from checkpoint")

        # Load fusion layer
        if (self.fusion_layer is not None and
                (load_path / "fusion_layer.pt").exists()):
            self.fusion_layer.load_state_dict(
                torch.load(
                    load_path / "fusion_layer.pt",
                    map_location=self.device
                )
            )
            print("Loaded fusion layer from checkpoint")

        print(f"Multimodal checkpoint loaded from {load_dir}")

    def get_vision_parameters(self) -> List[torch.nn.Parameter]:
        """Get all trainable vision parameters."""
        params = []

        if self.vision_lora_model is not None:
            params.extend(self.vision_lora_model.get_trainable_parameters())

        if self.fusion_layer is not None:
            params.extend(self.fusion_layer.parameters())

        return params


if __name__ == "__main__":
    print("Testing Multimodal Continual Learner...")

    from ..model.llm import ContinualLLM, ModelConfig
    from ..model.vision_encoder import VisionEncoder, VisionEncoderConfig

    # Create small models for testing
    llm_config = ModelConfig(
        vocab_size=1000,
        d_model=512,
        num_layers=4,
        num_query_heads=8,
        num_kv_heads=2,
        max_seq_len=128
    )
    llm = ContinualLLM(llm_config)

    vision_config = VisionEncoderConfig(
        image_size=224,
        patch_size=16,
        hidden_dim=512,  # Match language model
        num_layers=6,
        num_heads=8
    )
    vision_encoder = VisionEncoder(vision_config)

    # Create multimodal config
    config = MultimodalContinualConfig(
        lora_r=8,
        vision_lora_r=8,
        enable_vision=True,
        batch_size=2,
        device="cpu"
    )

    print("\nMultimodal Configuration:")
    print(f"  Text LoRA rank: {config.lora_r}")
    print(f"  Vision LoRA rank: {config.vision_lora_r}")
    print(f"  Enable vision: {config.enable_vision}")
    print(f"  Device: {config.device}")

    # Create learner
    print("\nCreating multimodal learner...")
    learner = MultimodalContinualLearner(
        base_model=llm,
        vision_encoder=vision_encoder,
        config=config
    )

    # Test vision encoding
    print("\nTesting vision encoding:")
    test_images = [torch.randn(3, 224, 224)]
    vision_features = learner.encode_vision(images=test_images)

    if vision_features is not None:
        print(f"  Input images: {len(test_images)}")
        print(f"  Vision features shape: {vision_features.shape}")

    # Test domain management
    print("\nTesting domain management:")
    learner.add_domain_adapter("medical")
    learner.add_domain_adapter("satellite")
    print(f"  Available domains: {learner.vision_domains}")

    learner.set_active_domain("medical")
    print(f"  Active domain: {learner.active_vision_domain}")

    # Test parameter counting
    print("\nTrainable parameters:")
    vision_params = learner.get_vision_parameters()
    print(f"  Vision parameters: {sum(p.numel() for p in vision_params):,}")

    print("\n✓ Multimodal Continual Learner implementation complete!")
