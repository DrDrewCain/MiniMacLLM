"""
Script to pre-train the base LLM model.

Usage:
    python scripts/pretrain.py --config configs/small.yaml --data data/train.txt --tokenizer tokenizers/my_tokenizer

This pre-trains the base model before continual learning.
"""

import argparse
import sys
import yaml
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.llm import ContinualLLM, ModelConfig
from src.tokenization.bpe_tokenizer import BPETokenizer


class TextDataset(Dataset):
    """Simple text dataset for pre-training."""

    def __init__(self, file_path: str, tokenizer: BPETokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Read all texts
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(self.texts)} texts")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)

        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Convert to tensor
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels
        }


def collate_fn(batch):
    """Collate function to pad sequences."""
    # Find max length
    max_len = max(item['input_ids'].size(0) for item in batch)

    input_ids = []
    labels = []

    for item in batch:
        # Pad input_ids
        pad_len = max_len - item['input_ids'].size(0)
        padded_input = F.pad(item['input_ids'], (0, pad_len), value=0)
        input_ids.append(padded_input)

        # Pad labels (use -1 for padding, will be ignored in loss)
        padded_labels = F.pad(item['labels'], (0, pad_len), value=-1)
        labels.append(padded_labels)

    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels)
    }


def train_epoch(model, dataloader, optimizer, device, epoch, grad_accumulation_steps=1):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits, _, _ = model(input_ids)

        # Compute loss (cross-entropy on next-token prediction)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-1
        )

        # Scale loss by accumulation steps
        loss = loss / grad_accumulation_steps

        # Backward
        loss.backward()

        # Update weights every grad_accumulation_steps
        if (batch_idx + 1) % grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accumulation_steps
        num_batches += 1

        # Update progress bar
        progress.set_postfix({'loss': total_loss / num_batches})

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Pre-train base LLM")

    # Model config
    parser.add_argument("--config", type=str,
                       help="Path to model config YAML")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_query_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--max_seq_len", type=int, default=512)

    # Training
    parser.add_argument("--data", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--tokenizer", type=str, required=True,
                       help="Path to trained tokenizer")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="mps")

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=1000)

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("="*60)
    print("Pre-training LLM")
    print("="*60)
    print(f"Device: {device}")
    print(f"Data: {args.data}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.grad_accumulation}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print("="*60 + "\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer.load(args.tokenizer)
    print(f"Tokenizer loaded: {len(tokenizer)} tokens\n")

    # Create model config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = ModelConfig(**config_dict)
    else:
        config = ModelConfig(
            vocab_size=len(tokenizer),
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len
        )

    print("Model configuration:")
    print(f"  Vocabulary: {config.vocab_size:,}")
    print(f"  Model dim: {config.d_model}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Query heads: {config.num_query_heads}")
    print(f"  KV heads: {config.num_kv_heads}")
    print(f"  Feed-forward: {config.d_ff}")
    print()

    # Create model
    print("Creating model...")
    model = ContinualLLM(config)
    model.to(device)

    num_params = model.get_num_params()
    print(f"Model created: {num_params:,} parameters\n")

    # Create dataset and dataloader
    dataset = TextDataset(args.data, tokenizer, max_length=args.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Training loop
    print("Starting training...\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(
            model, dataloader, optimizer, device, epoch,
            grad_accumulation_steps=args.grad_accumulation
        )

        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

        # Save checkpoint
        save_path = Path(args.save_dir) / f"epoch_{epoch}"
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.__dict__,
            'loss': avg_loss
        }, save_path / "checkpoint.pt")

        print(f"Checkpoint saved to {save_path}\n")

    elapsed = time.time() - start_time
    print(f"Training complete! Time: {elapsed/60:.2f} minutes")

    # Save final model
    final_path = Path(args.save_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__
    }, final_path / "model.pt")

    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
