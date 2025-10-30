"""
Custom LLM Implementation - Transformer-based Language Model
Built from scratch for unsupervised learning on text data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SimpleTokenizer:
    """
    Character-level tokenizer for text preprocessing.
    Can be extended to BPE or WordPiece tokenization.
    """

    def __init__(self, vocab: Optional[set] = None):
        if vocab is None:
            # Start with basic ASCII characters
            self.vocab = set(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:\'-\n')
        else:
            self.vocab = vocab

        # Create mappings
        self.char_to_idx = {ch: idx for idx, ch in enumerate(sorted(self.vocab))}
        self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

        # Add special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.char_to_idx[self.pad_token] = self.vocab_size
        self.char_to_idx[self.unk_token] = self.vocab_size + 1
        self.idx_to_char[self.vocab_size] = self.pad_token
        self.idx_to_char[self.vocab_size + 1] = self.unk_token
        self.vocab_size += 2

    def encode(self, text: str) -> list:
        """Convert text to list of indices"""
        return [self.char_to_idx.get(ch, self.char_to_idx[self.unk_token]) for ch in text]

    def decode(self, indices: list) -> str:
        """Convert list of indices back to text"""
        return ''.join([self.idx_to_char.get(idx, self.unk_token) for idx in indices])


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    Core component of transformer architecture.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the last dimension into (num_heads, d_k)"""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()

        # Linear projections and split heads
        Q = self.split_heads(self.W_q(x))  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )

        # Final linear projection
        output = self.W_o(attention_output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Two linear transformations with GELU activation.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block with:
    - Multi-head self-attention (with causal mask)
    - Feed-forward network
    - Layer normalization
    - Residual connections
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class CustomLLM(nn.Module):
    """
    Full transformer-based language model for unsupervised learning.
    Uses next-token prediction as the training objective.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using normal distribution"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask to prevent attending to future tokens"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.view(1, 1, seq_len, seq_len)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = input_ids.size()

        # Create position indices
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)

        # Create causal mask
        mask = self.create_causal_mask(seq_len, input_ids.device)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text using the model.

        Args:
            idx: Starting token indices (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens
        """
        for _ in range(max_new_tokens):
            # Crop context if too long
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]

            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optionally crop to top k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


class LLMTrainer:
    """
    Training loop for the language model with next-token prediction.
    Implements unsupervised learning objective.
    """

    def __init__(
        self,
        model: CustomLLM,
        tokenizer: SimpleTokenizer,
        learning_rate: float = 3e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def prepare_batch(self, texts: list, max_len: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare a batch of text for training.
        Creates input_ids and target_ids for next-token prediction.
        """
        batch_input_ids = []
        batch_target_ids = []

        for text in texts:
            tokens = self.tokenizer.encode(text)[:max_len]

            # Input is all tokens except last
            # Target is all tokens except first (shifted by 1)
            input_ids = tokens[:-1]
            target_ids = tokens[1:]

            batch_input_ids.append(input_ids)
            batch_target_ids.append(target_ids)

        # Pad sequences
        max_batch_len = max(len(seq) for seq in batch_input_ids)

        padded_inputs = []
        padded_targets = []

        for inp, tgt in zip(batch_input_ids, batch_target_ids):
            pad_len = max_batch_len - len(inp)
            padded_inputs.append(inp + [self.tokenizer.char_to_idx[self.tokenizer.pad_token]] * pad_len)
            padded_targets.append(tgt + [-1] * pad_len)  # -1 is ignored in loss

        return (
            torch.tensor(padded_inputs, dtype=torch.long, device=self.device),
            torch.tensor(padded_targets, dtype=torch.long, device=self.device)
        )

    def train_step(self, texts: list) -> float:
        """Single training step"""
        self.model.train()

        # Prepare batch
        input_ids, target_ids = self.prepare_batch(texts)

        # Forward pass
        logits, loss = self.model(input_ids, target_ids)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def generate_text(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8) -> str:
        """Generate text from a prompt"""
        self.model.eval()

        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Generate
        generated_ids = self.model.generate(idx, max_new_tokens, temperature=temperature, top_k=40)

        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())

        return generated_text


if __name__ == "__main__":
    # Example usage
    print("Initializing Custom LLM...")

    # Create tokenizer
    tokenizer = SimpleTokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Create model
    model = CustomLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_seq_len=256,
        dropout=0.1
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = LLMTrainer(model, tokenizer, learning_rate=3e-4)

    # Example training data
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models can learn complex patterns from data."
    ]

    print("\nTraining example...")
    for epoch in range(10):
        loss = trainer.train_step(training_texts)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    # Generate text
    print("\nGenerating text from prompt: 'The quick'")
    generated = trainer.generate_text("The quick", max_new_tokens=50, temperature=0.8)
    print(f"Generated: {generated}")
