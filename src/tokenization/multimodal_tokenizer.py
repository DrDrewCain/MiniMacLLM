"""
Multimodal Tokenizer for Vision-Language Models.

Extends the byte-level BPE tokenizer to handle mixed text + vision inputs.
Supports:
- Text-only sequences (backward compatible)
- Image + text sequences
- Video + text sequences
- Multiple images/videos in one sequence

Special tokens for vision:
- <|vision_start|>: Marks beginning of visual content
- <|vision_end|>: Marks end of visual content
- <|frame|>: Separator between video frames (optional)

Design:
- Visual tokens are represented as special token IDs in a reserved range
- Position tracking maintains alignment between text and vision
- Batch encoding supports mixed-modality sequences
"""

import torch
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

from .bpe_tokenizer import BPETokenizer, BatchEncoding


@dataclass
class MultimodalInput:
    """
    Container for multimodal input data.

    Attributes:
        text: Text string (optional)
        images: List of image tensors (optional)
        videos: List of video tensors (optional)
        interleaved: Whether to interleave vision and text
    """
    text: Optional[str] = None
    images: Optional[List[torch.Tensor]] = None
    videos: Optional[List[torch.Tensor]] = None
    interleaved: bool = False  # If True, alternate text/vision


@dataclass
class MultimodalEncoding:
    """
    Output from multimodal tokenizer.

    Attributes:
        input_ids: Token IDs (includes vision placeholders)
        attention_mask: Attention mask
        vision_mask: Mask indicating which positions are vision tokens
        vision_features: List of visual feature tensors
        position_ids: Position IDs for the sequence
        modality_types: Type of each position (0=text, 1=image, 2=video)
    """
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    vision_mask: Optional[List[int]] = None
    vision_features: Optional[List[torch.Tensor]] = None
    position_ids: Optional[List[int]] = None
    modality_types: Optional[List[int]] = None


class MultimodalTokenizer(BPETokenizer):
    """
    Multimodal tokenizer extending BPE for vision+text.

    Adds special tokens for vision boundaries and manages mixed sequences.

    Example:
        >>> tokenizer = MultimodalTokenizer.from_pretrained("tokenizer/")
        >>>
        >>> # Text only (backward compatible)
        >>> tokens = tokenizer.encode("Hello world")
        >>>
        >>> # Image + text
        >>> image = torch.randn(3, 224, 224)
        >>> encoding = tokenizer.encode_multimodal(
        ...     text="This is a cat",
        ...     images=[image]
        ... )
        >>> encoding.input_ids  # [vision_start, vis_0, vis_1, ...,
        >>>                     #  vision_end, text_tokens...]
    """

    # Special token definitions
    VISION_START_TOKEN = "<|vision_start|>"
    VISION_END_TOKEN = "<|vision_end|>"
    FRAME_SEP_TOKEN = "<|frame|>"

    # Vision token ID ranges (reserve 1000 IDs for vision placeholders)
    VISION_TOKEN_START = 100000
    VISION_TOKEN_END = 101000

    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        normalization: str = "NFC",
        num_vision_tokens: int = 256,  # Number of visual tokens per image
    ):
        """
        Initialize multimodal tokenizer.

        Args:
            vocab_size: Base vocabulary size for text
            min_frequency: Minimum frequency for BPE merges
            special_tokens: Additional special tokens
            normalization: Unicode normalization
            num_vision_tokens: Number of tokens to represent each image/frame
        """
        # Add multimodal special tokens
        if special_tokens is None:
            special_tokens = ["<|endoftext|>"]

        # Add vision tokens if not present
        vision_tokens = [
            self.VISION_START_TOKEN,
            self.VISION_END_TOKEN,
            self.FRAME_SEP_TOKEN
        ]
        for token in vision_tokens:
            if token not in special_tokens:
                special_tokens.append(token)

        # Initialize base tokenizer
        super().__init__(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            normalization=normalization
        )

        self.num_vision_tokens = num_vision_tokens

    @property
    def vision_start_token_id(self) -> int:
        """Get vision start token ID."""
        return self.vocab.get(self.VISION_START_TOKEN)

    @property
    def vision_end_token_id(self) -> int:
        """Get vision end token ID."""
        return self.vocab.get(self.VISION_END_TOKEN)

    @property
    def frame_sep_token_id(self) -> int:
        """Get frame separator token ID."""
        return self.vocab.get(self.FRAME_SEP_TOKEN)

    def _create_vision_token_ids(
        self,
        num_tokens: int
    ) -> List[int]:
        """
        Create placeholder token IDs for vision features.

        Args:
            num_tokens: Number of vision tokens needed

        Returns:
            List of vision token IDs
        """
        # Use reserved range for vision tokens
        return list(range(
            self.VISION_TOKEN_START,
            self.VISION_TOKEN_START + num_tokens
        ))

    def encode_multimodal(
        self,
        text: Optional[str] = None,
        images: Optional[List[torch.Tensor]] = None,
        videos: Optional[List[torch.Tensor]] = None,
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
        return_vision_mask: bool = True,
        return_position_ids: bool = True,
    ) -> MultimodalEncoding:
        """
        Encode multimodal input (text + images/videos).

        Args:
            text: Text string
            images: List of image tensors
            videos: List of video tensors
            add_special_tokens: Whether to add EOS token
            return_attention_mask: Return attention mask
            return_vision_mask: Return vision token mask
            return_position_ids: Return position IDs

        Returns:
            MultimodalEncoding with token IDs and metadata
        """
        input_ids = []
        vision_mask = []
        modality_types = []  # 0=text, 1=image, 2=video
        vision_features = []

        # Process images first (if any)
        if images:
            for image in images:
                # Add vision start token
                input_ids.append(self.vision_start_token_id)
                vision_mask.append(0)
                modality_types.append(0)

                # Add vision placeholder tokens
                num_vis_tokens = self.num_vision_tokens
                vis_token_ids = self._create_vision_token_ids(num_vis_tokens)
                input_ids.extend(vis_token_ids)
                vision_mask.extend([1] * num_vis_tokens)
                modality_types.extend([1] * num_vis_tokens)

                # Store visual features
                vision_features.append(image)

                # Add vision end token
                input_ids.append(self.vision_end_token_id)
                vision_mask.append(0)
                modality_types.append(0)

        # Process videos (if any)
        if videos:
            for video in videos:
                # Add vision start token
                input_ids.append(self.vision_start_token_id)
                vision_mask.append(0)
                modality_types.append(0)

                # Assume video shape: (T, C, H, W)
                num_frames = video.shape[0] if len(video.shape) == 4 else 1

                for frame_idx in range(num_frames):
                    # Add frame separator (except first frame)
                    if frame_idx > 0:
                        input_ids.append(self.frame_sep_token_id)
                        vision_mask.append(0)
                        modality_types.append(0)

                    # Add vision tokens for this frame
                    num_vis_tokens = self.num_vision_tokens
                    vis_token_ids = self._create_vision_token_ids(
                        num_vis_tokens
                    )
                    input_ids.extend(vis_token_ids)
                    vision_mask.extend([1] * num_vis_tokens)
                    modality_types.extend([2] * num_vis_tokens)

                # Store video features
                vision_features.append(video)

                # Add vision end token
                input_ids.append(self.vision_end_token_id)
                vision_mask.append(0)
                modality_types.append(0)

        # Process text
        if text:
            text_tokens = self.encode(text, add_special_tokens=False)
            input_ids.extend(text_tokens)
            vision_mask.extend([0] * len(text_tokens))
            modality_types.extend([0] * len(text_tokens))

        # Add EOS token if requested
        if add_special_tokens and self.special_tokens:
            input_ids.append(self.vocab[self.special_tokens[0]])
            vision_mask.append(0)
            modality_types.append(0)

        # Create attention mask (all 1s for now)
        attention_mask = (
            [1] * len(input_ids) if return_attention_mask else None
        )

        # Create position IDs (sequential)
        position_ids = (
            list(range(len(input_ids))) if return_position_ids else None
        )

        return MultimodalEncoding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_mask=vision_mask if return_vision_mask else None,
            vision_features=vision_features if vision_features else None,
            position_ids=position_ids,
            modality_types=modality_types
        )

    def batch_encode_multimodal(
        self,
        inputs: List[MultimodalInput],
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Union[List, torch.Tensor]]:
        """
        Batch encode multiple multimodal inputs.

        Args:
            inputs: List of MultimodalInput objects
            padding: Padding strategy
            max_length: Maximum sequence length
            return_tensors: Return format ("pt" for PyTorch, None for lists)

        Returns:
            Dictionary with batched tensors/lists
        """
        # Encode each input
        encodings = [
            self.encode_multimodal(
                text=inp.text,
                images=inp.images,
                videos=inp.videos
            )
            for inp in inputs
        ]

        # Get max length for padding
        if padding:
            if max_length:
                target_length = max_length
            else:
                target_length = max(len(enc.input_ids) for enc in encodings)
        else:
            target_length = 0

        # Prepare batched outputs
        batch_input_ids = []
        batch_attention_mask = []
        batch_vision_mask = []
        batch_position_ids = []
        batch_modality_types = []

        for enc in encodings:
            # Truncate if needed
            if max_length and len(enc.input_ids) > max_length:
                enc.input_ids = enc.input_ids[:max_length]
                if enc.attention_mask:
                    enc.attention_mask = enc.attention_mask[:max_length]
                if enc.vision_mask:
                    enc.vision_mask = enc.vision_mask[:max_length]
                if enc.position_ids:
                    enc.position_ids = enc.position_ids[:max_length]
                if enc.modality_types:
                    enc.modality_types = enc.modality_types[:max_length]

            # Pad if needed
            if padding and target_length > len(enc.input_ids):
                pad_length = target_length - len(enc.input_ids)
                pad_id = self.pad_token_id if self.pad_token_id else 0

                enc.input_ids.extend([pad_id] * pad_length)
                if enc.attention_mask:
                    enc.attention_mask.extend([0] * pad_length)
                if enc.vision_mask:
                    enc.vision_mask.extend([0] * pad_length)
                if enc.position_ids:
                    # Continue position IDs
                    last_pos = enc.position_ids[-1] if enc.position_ids else 0
                    enc.position_ids.extend(
                        range(last_pos + 1, last_pos + 1 + pad_length)
                    )
                if enc.modality_types:
                    enc.modality_types.extend([0] * pad_length)

            batch_input_ids.append(enc.input_ids)
            if enc.attention_mask:
                batch_attention_mask.append(enc.attention_mask)
            if enc.vision_mask:
                batch_vision_mask.append(enc.vision_mask)
            if enc.position_ids:
                batch_position_ids.append(enc.position_ids)
            if enc.modality_types:
                batch_modality_types.append(enc.modality_types)

        # Convert to tensors if requested
        result = {"input_ids": batch_input_ids}

        if batch_attention_mask:
            result["attention_mask"] = batch_attention_mask
        if batch_vision_mask:
            result["vision_mask"] = batch_vision_mask
        if batch_position_ids:
            result["position_ids"] = batch_position_ids
        if batch_modality_types:
            result["modality_types"] = batch_modality_types

        if return_tensors == "pt":
            result = {
                k: torch.tensor(v) if isinstance(v, list) else v
                for k, v in result.items()
            }

        return result

    def decode_multimodal(
        self,
        input_ids: List[int],
        skip_special_tokens: bool = True,
        skip_vision_tokens: bool = True
    ) -> str:
        """
        Decode token IDs back to text (skipping vision tokens).

        Args:
            input_ids: Token IDs
            skip_special_tokens: Skip special tokens
            skip_vision_tokens: Skip vision placeholder tokens

        Returns:
            Decoded text string
        """
        # Filter out vision tokens
        filtered_ids = []

        for token_id in input_ids:
            # Skip vision placeholders
            if skip_vision_tokens:
                if (self.VISION_TOKEN_START <= token_id <
                        self.VISION_TOKEN_END):
                    continue

            # Skip vision boundary tokens
            if skip_special_tokens:
                token_str = self.inverse_vocab.get(token_id, "")
                if token_str in [
                    self.VISION_START_TOKEN,
                    self.VISION_END_TOKEN,
                    self.FRAME_SEP_TOKEN
                ]:
                    continue

            filtered_ids.append(token_id)

        # Decode using base tokenizer
        return self.decode(filtered_ids, skip_special_tokens)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        num_vision_tokens: int = 256
    ) -> "MultimodalTokenizer":
        """
        Load multimodal tokenizer from pretrained BPE tokenizer.

        Args:
            path: Path to tokenizer directory
            num_vision_tokens: Number of vision tokens per image

        Returns:
            Loaded tokenizer
        """
        # Load base tokenizer
        base_tokenizer = BPETokenizer.load(path)

        # Create multimodal tokenizer with same config
        tokenizer = cls(
            vocab_size=base_tokenizer.vocab_size,
            min_frequency=base_tokenizer.min_frequency,
            special_tokens=base_tokenizer.special_tokens,
            normalization=base_tokenizer.normalization,
            num_vision_tokens=num_vision_tokens
        )

        # Copy learned parameters
        tokenizer.vocab = base_tokenizer.vocab
        tokenizer.inverse_vocab = base_tokenizer.inverse_vocab
        tokenizer.merges = base_tokenizer.merges
        tokenizer.merge_ranks = base_tokenizer.merge_ranks

        return tokenizer


if __name__ == "__main__":
    print("Testing Multimodal Tokenizer...")

    # Create tokenizer
    tokenizer = MultimodalTokenizer(vocab_size=1000, min_frequency=1)

    # Train on small corpus
    corpus = [
        "Hello, world!",
        "This is a test.",
        "Multimodal learning is cool."
    ]
    tokenizer.train(corpus, verbose=False)

    print("\nSpecial tokens:")
    print(f"  Vision start: {tokenizer.vision_start_token_id}")
    print(f"  Vision end: {tokenizer.vision_end_token_id}")
    print(f"  Frame sep: {tokenizer.frame_sep_token_id}")

    # Test text-only (backward compatible)
    print("\n1. Text-only encoding:")
    text_tokens = tokenizer.encode("Hello, world!")
    print(f"  Tokens: {text_tokens[:10]}...")
    decoded = tokenizer.decode(text_tokens)
    print(f"  Decoded: '{decoded}'")

    # Test image + text
    print("\n2. Image + text encoding:")
    image = torch.randn(3, 224, 224)
    encoding = tokenizer.encode_multimodal(
        text="This is a cat",
        images=[image]
    )
    print(f"  Total tokens: {len(encoding.input_ids)}")
    print(f"  Vision tokens: {sum(encoding.vision_mask)}")
    print(f"  First 10 token IDs: {encoding.input_ids[:10]}")

    # Test video + text
    print("\n3. Video + text encoding:")
    video = torch.randn(8, 3, 224, 224)  # 8 frames
    encoding = tokenizer.encode_multimodal(
        text="A person walking",
        videos=[video]
    )
    print(f"  Total tokens: {len(encoding.input_ids)}")
    print(f"  Vision tokens: {sum(encoding.vision_mask)}")

    # Test batch encoding
    print("\n4. Batch encoding:")
    inputs = [
        MultimodalInput(text="Hello"),
        MultimodalInput(
            text="World",
            images=[torch.randn(3, 224, 224)]
        )
    ]
    batch = tokenizer.batch_encode_multimodal(
        inputs,
        padding=True,
        return_tensors="pt"
    )
    print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"  Batch attention_mask shape: {batch['attention_mask'].shape}")

    print("\n✓ Multimodal Tokenizer implementation complete!")
