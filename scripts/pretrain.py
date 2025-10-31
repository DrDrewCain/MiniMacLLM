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
import math

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.llm import ContinualLLM, ModelConfig
from src.tokenization.bpe_tokenizer import BPETokenizer
from src.evaluation.metrics import compute_perplexity


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


def train_epoch(model, dataloader, optimizer, device, epoch, grad_accumulation_steps=1, scheduler=None):
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
            
            # Step scheduler if provided
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * grad_accumulation_steps
        num_batches += 1

        # Update progress bar
        progress.set_postfix({'loss': total_loss / num_batches})

    return total_loss / num_batches


def evaluate_model(model, dataloader, device, max_batches=None):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits, _, _ = model(input_ids)

            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-1
            )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')  # Avoid overflow
    
    return avg_loss, perplexity


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
    parser.add_argument("--val_data", type=str, default=None,
                       help="Path to validation data (optional)")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Validation split ratio if val_data not provided")
    parser.add_argument("--tokenizer", type=str, required=True,
                       help="Path to trained tokenizer")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Warmup steps for learning rate scheduler")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                       help="Stop if validation loss doesn't improve for N epochs (0 to disable)")

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
    print(f"Validation: {args.val_data if args.val_data else f'{args.val_split*100:.0f}% split'}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.grad_accumulation}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Early stopping patience: {args.early_stopping_patience if args.early_stopping_patience > 0 else 'Disabled'}")
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

    # Create datasets
    print("Loading training data...")
    train_dataset = TextDataset(args.data, tokenizer, max_length=args.max_seq_len)
    
    # Handle validation data
    val_dataloader = None
    if args.val_data:
        print(f"Loading validation data from {args.val_data}...")
        val_dataset = TextDataset(args.val_data, tokenizer, max_length=args.max_seq_len)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
    elif args.val_split > 0:
        print(f"Splitting {args.val_split*100:.0f}% of training data for validation...")
        # Split the dataset
        train_size = int(len(train_dataset) * (1 - args.val_split))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
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
    
    # Create learning rate scheduler with warmup and cosine decay
    total_steps = len(train_dataloader) * args.epochs // args.grad_accumulation
    
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, args.warmup_steps))
        # Cosine decay
        progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"Total training steps: {total_steps}")
    print(f"Learning rate schedule: warmup {args.warmup_steps} steps, then cosine decay\n")

    # Training loop
    print("Starting training...\n")
    start_time = time.time()
    
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        avg_loss = train_epoch(
            model, train_dataloader, optimizer, device, epoch,
            grad_accumulation_steps=args.grad_accumulation,
            scheduler=scheduler
        )
        
        # Compute training perplexity
        train_ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        # Validate
        if val_dataloader is not None:
            val_loss, val_ppl = evaluate_model(model, val_dataloader, device)
            
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"  Train Loss: {avg_loss:.4f} | Train PPL: {train_ppl:.2f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                
                # Save best checkpoint
                best_path = Path(args.save_dir) / "best"
                best_path.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config.__dict__,
                    'train_loss': avg_loss,
                    'val_loss': val_loss,
                    'val_perplexity': val_ppl
                }, best_path / "checkpoint.pt")
                
                print(f"  ✓ New best checkpoint saved (val_loss: {val_loss:.4f}, val_ppl: {val_ppl:.2f})")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement} epoch(s)")
        else:
            print(f"Epoch {epoch}/{args.epochs} - Train Loss: {avg_loss:.4f} | Train PPL: {train_ppl:.2f}")

        # Save regular checkpoint
        save_path = Path(args.save_dir) / f"epoch_{epoch}"
        save_path.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.__dict__,
            'train_loss': avg_loss
        }
        
        if val_dataloader is not None:
            checkpoint_data['val_loss'] = val_loss
            checkpoint_data['val_perplexity'] = val_ppl

        torch.save(checkpoint_data, save_path / "checkpoint.pt")
        print(f"  Checkpoint saved to {save_path}\n")
        
        # Early stopping check
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
            break

    elapsed = time.time() - start_time
    print(f"\nTraining complete! Time: {elapsed/60:.2f} minutes")
    
    if val_dataloader is not None:
        print(f"Best model: Epoch {best_epoch} with validation loss {best_val_loss:.4f}")
        print(f"Best checkpoint available at: {args.save_dir}/best/checkpoint.pt")

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
