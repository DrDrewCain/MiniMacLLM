"""
Complete Language Model implementation.

Modern transformer-based LLM with:
- Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE)
- SwiGLU feed-forward
- RMSNorm
- Pre-normalization architecture

This is the base model that will be used for continual learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .embeddings import RotaryPositionEmbedding
from .transformer_block import TransformerBlock
from .normalization import create_norm_layer
from .attention import create_causal_mask, AttentionCache


@dataclass
class ModelConfig:
    """
    Configuration for the language model.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Model dimension
        num_layers: Number of transformer blocks
        num_query_heads: Number of query attention heads
        num_kv_heads: Number of key-value attention heads
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        norm_type: Type of normalization ("rmsnorm" or "layernorm")
        ff_type: Type of feed-forward ("swiglu", "geglu", "gelu")
        norm_eps: Epsilon for normalization
        bias: Whether to use bias in linear layers
        rope_base: Base for RoPE geometric progression
        tie_embeddings: Whether to tie input and output embeddings
    """
    # Architecture
    vocab_size: int = 32000
    d_model: int = 512
    num_layers: int = 12
    num_query_heads: int = 8
    num_kv_heads: int = 2
    d_ff: int = 2048
    max_seq_len: int = 2048

    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Component types
    norm_type: str = "rmsnorm"
    ff_type: str = "swiglu"

    # Hyperparameters
    norm_eps: float = 1e-6
    bias: bool = False
    rope_base: float = 10000.0
    tie_embeddings: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads ({self.num_query_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

        if self.d_model % self.num_query_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_query_heads ({self.num_query_heads})"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.num_query_heads

    @property
    def num_queries_per_kv(self) -> int:
        """Number of query heads per KV head."""
        return self.num_query_heads // self.num_kv_heads


class ContinualLLM(nn.Module):
    """
    Complete Language Model with continual learning support.

    This is a modern decoder-only transformer architecture similar to:
    - LLaMA 3 (Meta)
    - Mistral (Mistral AI)
    - GPT-NeoX (EleutherAI)

    Key features:
    - GQA for efficient inference
    - RoPE for better position encoding
    - SwiGLU for better performance
    - Pre-norm for stable training
    - No bias terms (modern standard)

    Args:
        config: Model configuration

    Example:
        >>> config = ModelConfig(vocab_size=32000, d_model=512, num_layers=12)
        >>> model = ContinualLLM(config)
        >>> input_ids = torch.randint(0, 32000, (2, 10))
        >>> logits = model(input_ids)
        >>> logits.shape
        torch.Size([2, 10, 32000])
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Rotary position embeddings (no learned positional embeddings needed)
        self.rope = RotaryPositionEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base
        )

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                num_query_heads=config.num_query_heads,
                num_kv_heads=config.num_kv_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                norm_type=config.norm_type,
                ff_type=config.ff_type,
                norm_eps=config.norm_eps,
                bias=config.bias,
                max_seq_len=config.max_seq_len
            )
            for _ in range(config.num_layers)
        ])

        # Final normalization
        self.norm = create_norm_layer(
            dim=config.d_model,
            norm_type=config.norm_type,
            eps=config.norm_eps
        )

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie input and output embeddings (common practice)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights following LLaMA/GPT-NeoX conventions.

        - Embeddings: N(0, 0.02)
        - Linear: N(0, 0.02)
        - Residual projections: scaled by 1/√(2*num_layers)
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[AttentionCache]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[AttentionCache]], Optional[List[torch.Tensor]]]:
        """
        Forward pass of the model.

        Args:
            input_ids: Token indices of shape (batch, seq_len)
            attention_mask: Mask of shape (batch, seq_len) with 1 for real tokens, 0 for padding
            past_key_values: List of cached key-value pairs from previous forward passes
            use_cache: Whether to return key-value caches
            output_hidden_states: Whether to return hidden states from all layers

        Returns:
            - logits: Token logits of shape (batch, seq_len, vocab_size)
            - past_key_values: List of AttentionCache objects (if use_cache=True)
            - hidden_states: List of hidden state tensors (if output_hidden_states=True)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Validate sequence length
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.config.max_seq_len}"
            )

        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.dropout(hidden_states)

        # Get RoPE embeddings
        cos, sin = self.rope(seq_len, device=device)

        # Create or use attention mask
        if attention_mask is None:
            # Causal mask for autoregressive generation
            causal_mask = create_causal_mask(seq_len, device)
        else:
            # Convert attention mask to additive mask
            # attention_mask: (batch, seq_len) with 1 for real tokens, 0 for padding
            # We need: (batch, 1, seq_len, seq_len) with 0 for attend, -inf for masked
            causal_mask = create_causal_mask(seq_len, device)

            # Expand attention mask to match causal mask shape
            expanded_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            expanded_mask = (1.0 - expanded_mask) * torch.finfo(hidden_states.dtype).min

            # Combine causal and padding masks
            causal_mask = causal_mask + expanded_mask

        # Pass through transformer layers
        new_key_values = [] if use_cache else None
        all_hidden_states = [] if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            # Get cached KV if available
            past_kv = past_key_values[i] if past_key_values is not None else None

            # Forward through layer
            hidden_states, new_kv = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                mask=causal_mask,
                cache=past_kv,
                use_cache=use_cache
            )

            if use_cache:
                new_key_values.append(new_kv)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        return logits, new_key_values, all_hidden_states

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Get the number of parameters in the model.

        Args:
            non_embedding: If True, subtract embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            n_params -= self.token_embedding.weight.numel()

        return n_params

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting tokens of shape (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, nucleus sampling (sample from top p probability mass)
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            do_sample: If False, use greedy decoding

        Returns:
            Generated token ids of shape (batch, seq_len + max_new_tokens)
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop if exceeds max length
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else \
                       input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _, _ = self(idx_cond)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(logits.shape[0]):
                    for token_id in set(input_ids[i].tolist()):
                        logits[i, token_id] /= repetition_penalty

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample or take most likely
            probs = F.softmax(logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, idx_next], dim=1)

        return input_ids


def count_parameters(model: nn.Module) -> dict:
    """
    Count parameters in model by category.

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding = model.token_embedding.weight.numel()

    if hasattr(model, 'lm_head') and model.config.tie_embeddings:
        # Don't double count tied embeddings
        non_embedding = total - embedding
    else:
        non_embedding = total - embedding - model.lm_head.weight.numel()

    return {
        'total': total,
        'trainable': trainable,
        'embedding': embedding,
        'non_embedding': non_embedding
    }


if __name__ == "__main__":
    # Test the model
    print("Testing ContinualLLM...")

    # Create small model configuration
    config = ModelConfig(
        vocab_size=32000,
        d_model=512,
        num_layers=6,
        num_query_heads=8,
        num_kv_heads=2,
        d_ff=2048,
        max_seq_len=2048,
        dropout=0.1
    )

    print(f"Model configuration:")
    print(f"  Vocabulary size: {config.vocab_size:,}")
    print(f"  Model dimension: {config.d_model}")
    print(f"  Number of layers: {config.num_layers}")
    print(f"  Query heads: {config.num_query_heads}")
    print(f"  KV heads: {config.num_kv_heads}")
    print(f"  FF dimension: {config.d_ff}")

    # Create model
    model = ContinualLLM(config)

    # Count parameters
    params = count_parameters(model)
    print(f"\nParameter counts:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Embedding: {params['embedding']:,}")
    print(f"  Non-embedding: {params['non_embedding']:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\nForward pass:")
    print(f"  Input shape: {input_ids.shape}")

    logits, caches, _ = model(input_ids, use_cache=True)
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Number of cached layers: {len(caches)}")
    print(f"  Cache[0] key shape: {caches[0].key.shape}")

    # Test generation
    print(f"\nTesting generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.8,
        top_k=40,
        do_sample=True
    )
    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    print(f"  Generated tokens: {generated[0].tolist()}")

    print("\n✓ ContinualLLM implementation complete!")
