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
from src.neurobio.autonomous_learning import (
    AutonomousLearningRateController,
    AutonomousLearningConfig,
    AdaptiveOptimizer
)
from src.neurobio.eprop_optimizer import (
    EPropOptimizer,
    ContinualEPropOptimizer
)


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


def train_epoch(model, dataloader, optimizer, device, epoch, grad_accumulation_steps=1,
                use_autonomous_lr=False, learning_controller=None):
    """Train for one epoch with optional autonomous learning rate."""
    model.train()
    total_loss = 0
    num_batches = 0
    accumulated_loss = None

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

        # Track unscaled loss for adaptive optimizer
        if accumulated_loss is None:
            accumulated_loss = loss.clone()
        else:
            accumulated_loss += loss.clone()

        # Backward
        loss.backward()

        # Update weights every grad_accumulation_steps
        if (batch_idx + 1) % grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if use_autonomous_lr:
                if isinstance(optimizer, EPropOptimizer):
                    # E-prop needs loss for autonomous LR and neuromodulation
                    optimizer.step(loss=accumulated_loss * grad_accumulation_steps)
                elif isinstance(optimizer, AdaptiveOptimizer):
                    # Adaptive optimizer needs loss for autonomous LR
                    optimizer.step(loss=accumulated_loss * grad_accumulation_steps)
                else:
                    optimizer.step()
            else:
                optimizer.step()

            optimizer.zero_grad()
            accumulated_loss = None

        total_loss += loss.item() * grad_accumulation_steps
        num_batches += 1

        # Update progress bar with more info
        if use_autonomous_lr and learning_controller:
            dynamics = learning_controller.get_learning_dynamics()
            current_lr = learning_controller.lr_history[-1]['lr'] if learning_controller.lr_history else learning_controller.config.base_sensitivity
            progress.set_postfix({
                'loss': total_loss / num_batches,
                'lr': f'{current_lr:.2e}',
                'success': f'{dynamics["success_rate"]:.0%}'
            })
        else:
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
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Fixed learning rate (ignored if --use_autonomous_lr)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="mps")

    # Autonomous learning
    parser.add_argument("--use_autonomous_lr", action="store_true",
                       help="Use brain-inspired autonomous learning rate")
    parser.add_argument("--optimizer_type", type=str, default="sgd",
                       choices=["sgd", "adam", "eprop"],
                       help="Optimizer type: sgd (simple), adam (standard), eprop (continual learning)")
    parser.add_argument("--base_sensitivity", type=float, default=1.0,
                       help="Base sensitivity for autonomous LR (not a learning rate!)")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                       help="Minimum allowed emergent learning rate")
    parser.add_argument("--max_lr", type=float, default=1e-2,
                       help="Maximum allowed emergent learning rate")

    # E-prop specific
    parser.add_argument("--trace_decay", type=float, default=0.95,
                       help="Eligibility trace decay for e-prop optimizer")
    parser.add_argument("--beta1", type=float, default=0.9,
                       help="Learning signal momentum (beta1) for e-prop")
    parser.add_argument("--beta2", type=float, default=0.999,
                       help="Neuromodulator scaling (beta2) for e-prop")

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
    if args.use_autonomous_lr:
        print("Using autonomous learning rate (emergent from mathematical principles)")
        autonomous_config = AutonomousLearningConfig(
            base_sensitivity=args.base_sensitivity,
            min_rate=args.min_lr,
            max_rate=args.max_lr,
            adaptation_speed=0.01,
            uncertainty_weight=0.5,
            metabolic_cost_weight=0.1
        )
        learning_controller = AutonomousLearningRateController(autonomous_config)

        # Choose optimizer type
        if args.optimizer_type == "eprop":
            print(f"  Optimizer: EProp (Eligibility Propagation)")
            print(f"  Trace decay: {args.trace_decay}")
            print(f"  Beta1 (learning signal): {args.beta1}")
            print(f"  Beta2 (neuromodulator): {args.beta2}")
            optimizer = EPropOptimizer(
                model.parameters(),
                learning_controller=learning_controller,
                trace_decay=args.trace_decay,
                beta1=args.beta1,
                beta2=args.beta2,
                weight_decay=0.1
            )
        elif args.optimizer_type == "adam":
            print(f"  Optimizer: Adaptive SGD (basic)")
            optimizer = AdaptiveOptimizer(
                model.parameters(),
                learning_controller=learning_controller,
                weight_decay=0.1
            )
        else:  # sgd
            print(f"  Optimizer: Adaptive SGD (basic)")
            optimizer = AdaptiveOptimizer(
                model.parameters(),
                learning_controller=learning_controller,
                weight_decay=0.1
            )

        print(f"  Base sensitivity: {args.base_sensitivity}")
        print(f"  LR will emerge between {args.min_lr} and {args.max_lr}\n")
    else:
        print(f"Using fixed learning rate: {args.learning_rate}")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        learning_controller = None

    # Training loop
    print("Starting training...\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(
            model, dataloader, optimizer, device, epoch,
            grad_accumulation_steps=args.grad_accumulation,
            use_autonomous_lr=args.use_autonomous_lr,
            learning_controller=learning_controller
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

    # Print autonomous learning summary if used
    if args.use_autonomous_lr and learning_controller.lr_history:
        print("\nAutonomous Learning Rate Summary:")
        print(f"  Initial LR: {learning_controller.lr_history[0]['lr']:.6f}")
        print(f"  Final LR: {learning_controller.lr_history[-1]['lr']:.6f}")
        print(f"  Min LR seen: {min(h['lr'] for h in learning_controller.lr_history):.6f}")
        print(f"  Max LR seen: {max(h['lr'] for h in learning_controller.lr_history):.6f}")

        dynamics = learning_controller.get_learning_dynamics()
        print(f"  Final success rate: {dynamics['success_rate']:.2%}")
        print(f"  Final metabolic cost: {dynamics['metabolic_cost']:.4f}")
        print("  LR emerged from mathematical principles - no human presets!")

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
