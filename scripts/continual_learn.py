"""
Script for continual learning from user data.

Usage:
    python scripts/continual_learn.py \
        --model checkpoints/final/model.pt \
        --tokenizer tokenizers/my_tokenizer \
        --data user_data.txt \
        --domain math

Enables real-time learning from streaming user data.
"""

import argparse
import sys
from pathlib import Path
import torch
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.llm import ContinualLLM
from src.tokenization.bpe_tokenizer import BPETokenizer
from src.continual.continual_trainer import ContinualLearner, ContinualLearningConfig


def main():
    parser = argparse.ArgumentParser(description="Continual learning from user data")

    # Model
    parser.add_argument("--model", type=str, required=True,
                       help="Path to base model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True,
                       help="Path to tokenizer")

    # Data
    parser.add_argument("--data", type=str, required=True,
                       help="Path to data file (one example per line)")
    parser.add_argument("--domain", type=str, default="general",
                       help="Domain label for this data")
    parser.add_argument("--importance", type=float, default=1.0,
                       help="Importance score for examples (0.0-1.0)")

    # Continual learning config
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=32.0,
                       help="LoRA alpha")
    parser.add_argument("--replay_buffer_size", type=int, default=10000,
                       help="Size of replay buffer")
    parser.add_argument("--replay_ratio", type=float, default=0.5,
                       help="Ratio of replay data in batches")
    parser.add_argument("--use_ewc", action="store_true",
                       help="Use EWC for preventing forgetting")
    parser.add_argument("--ewc_lambda", type=float, default=1000.0,
                       help="EWC regularization strength")

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--update_steps", type=int, default=100,
                       help="Number of update steps")
    parser.add_argument("--consolidation_freq", type=int, default=1000,
                       help="Steps between knowledge consolidations")

    # Device
    parser.add_argument("--device", type=str, default="mps",
                       help="Device to use (cpu, cuda, mps)")

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="continual_checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--adapter_name", type=str, default="default",
                       help="Name for this adapter")

    # Testing
    parser.add_argument("--test_prompt", type=str,
                       help="Test prompt for generation after learning")

    args = parser.parse_args()

    print("="*60)
    print("Continual Learning from User Data")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Data: {args.data}")
    print(f"Domain: {args.domain}")
    print(f"Device: {args.device}")
    print("="*60 + "\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer.load(args.tokenizer)
    print(f"✓ Tokenizer loaded: {len(tokenizer)} tokens\n")

    # Load base model
    print("Loading base model...")
    checkpoint = torch.load(args.model, map_location='cpu')

    # Get config
    from src.model.llm import ModelConfig
    config = ModelConfig(**checkpoint['config'])

    # Create base model
    base_model = ContinualLLM(config)

    # Handle both checkpoint formats:
    # - Pretrain format: 'model_state_dict' (plain model)
    # - Continual learning format: 'state_dict' (LoRA-wrapped model)
    if 'model_state_dict' in checkpoint:
        # Pretrain checkpoint - load directly
        base_model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        # Continual learning checkpoint - extract base model weights
        state_dict = checkpoint['state_dict']

        # Filter out LoRA-specific keys and remove 'base_model.' prefix
        base_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('base_model.') and 'lora' not in key:
                # Remove 'base_model.' prefix and '.base_layer' if present
                clean_key = key[len('base_model.'):]  # Remove 'base_model.'
                if '.base_layer.' in clean_key:
                    clean_key = clean_key.replace('.base_layer.', '')
                base_state_dict[clean_key] = value

        base_model.load_state_dict(base_state_dict, strict=False)
        print("✓ Loaded base model from continual learning checkpoint")
    else:
        raise KeyError("Checkpoint must contain either 'model_state_dict' or 'state_dict'")

    print(f"✓ Model loaded: {base_model.get_num_params():,} parameters\n")

    # Create continual learner
    print("Creating continual learner...")

    cl_config = ContinualLearningConfig(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        replay_buffer_size=args.replay_buffer_size,
        replay_ratio=args.replay_ratio,
        use_ewc=args.use_ewc,
        ewc_lambda=args.ewc_lambda,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        consolidation_frequency=args.consolidation_freq,
        device=args.device
    )

    learner = ContinualLearner(
        base_model=base_model,
        config=cl_config,
        adapter_name=args.adapter_name,
        tokenizer=tokenizer
    )

    print("✓ Continual learner created\n")

    # Load user data
    print(f"Loading user data from {args.data}...")
    with open(args.data, 'r', encoding='utf-8') as f:
        user_texts = [line.strip() for line in f if line.strip()]

    print(f"✓ Loaded {len(user_texts)} examples\n")

    # Learn from user data
    print("Starting continual learning...")
    print(f"  Domain: {args.domain}")
    print(f"  Importance: {args.importance}")
    print(f"  Update steps: {args.update_steps}")
    print()

    total_loss = 0
    num_updates = 0

    for text in tqdm(user_texts, desc="Learning from data"):
        # Learn from this example
        stats = learner.learn_from_text(
            text=text,
            domain=args.domain,
            importance=args.importance
        )

        if stats:
            total_loss += stats.get('total_loss', 0)
            num_updates += 1

    avg_loss = total_loss / num_updates if num_updates > 0 else 0

    print(f"\n✓ Learning complete!")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Updates: {num_updates}")
    print()

    # Print statistics
    learner.print_stats()

    # Save checkpoint
    print(f"Saving checkpoint to {args.save_dir}...")
    learner.save_checkpoint(args.save_dir)
    print("✓ Checkpoint saved\n")

    # Test generation if requested
    if args.test_prompt:
        print(f"Testing generation with prompt: '{args.test_prompt}'")
        response = learner.generate(
            prompt=args.test_prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=40
        )
        print(f"\nGenerated: {response}\n")

    print("="*60)
    print("Continual Learning Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
