"""
Train model using continual learning on text data.

Usage:
    # Train on single domain
    python scripts/continual_learn.py \
        --model checkpoints/base/model.pt \
        --tokenizer tokenizers/wikitext_8k_byte_level \
        --data data/training/math/gsm8k.txt \
        --domain math

    # Multi-domain training
    python scripts/continual_learn.py \
        --model checkpoints/base/model.pt \
        --tokenizer tokenizers/wikitext_8k_byte_level \
        --data data/training/science/arxiv.txt \
        --domain science \
        --adapter_name science_v1

    # Test with generation
    python scripts/continual_learn.py \
        --model checkpoints/base/model.pt \
        --tokenizer tokenizers/wikitext_8k_byte_level \
        --data data/training/code/code-instructions.txt \
        --domain code \
        --test_prompt "def calculate_sum" \
        --update_steps 200
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
    parser = argparse.ArgumentParser(description="Train model with continual learning")

    # Model
    parser.add_argument("--model", type=str, required=True,
                       help="Path to base model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True,
                       help="Path to tokenizer directory")

    # Data
    parser.add_argument("--data", type=str, required=True,
                       help="Path to training data file (one example per line)")
    parser.add_argument("--domain", type=str, default="general",
                       help="Domain label (general, math, science, code, etc)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Limit number of training samples")
    parser.add_argument("--importance", type=float, default=1.0,
                       help="Sample importance weight (0.0-1.0)")

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank (dimension of adaptation)")
    parser.add_argument("--lora_alpha", type=float, default=32.0,
                       help="LoRA scaling factor")

    # Anti-forgetting
    parser.add_argument("--replay_buffer_size", type=int, default=10000,
                       help="Max experiences stored for replay")
    parser.add_argument("--replay_ratio", type=float, default=0.5,
                       help="Fraction of batch from replay buffer")
    parser.add_argument("--use_ewc", action="store_true",
                       help="Enable Elastic Weight Consolidation")
    parser.add_argument("--ewc_lambda", type=float, default=1000.0,
                       help="EWC regularization strength")

    # Training
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--grad_accum", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm for clipping")

    # Sleep consolidation
    parser.add_argument("--sleep_freq", type=int, default=500,
                       help="Run sleep consolidation every N steps (0=disable)")
    parser.add_argument("--sleep_cycles", type=int, default=50,
                       help="Number of replay cycles during sleep")

    # Device
    parser.add_argument("--device", type=str, default="mps",
                       help="Device: cpu, cuda, or mps")

    # Output
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/continual",
                       help="Directory for saving checkpoints")
    parser.add_argument("--adapter_name", type=str, default="default",
                       help="Name for this LoRA adapter")
    parser.add_argument("--save_freq", type=int, default=100,
                       help="Save checkpoint every N samples")

    # Testing
    parser.add_argument("--test_prompt", type=str,
                       help="Generate text after training with this prompt")
    parser.add_argument("--test_max_tokens", type=int, default=50,
                       help="Max tokens to generate for test")

    args = parser.parse_args()

    print("="*60)
    print("Continual Learning Training")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Data: {args.data}")
    print(f"Domain: {args.domain}")
    print(f"Device: {args.device}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"EWC: {args.use_ewc}, Replay: {args.replay_ratio}")
    print("="*60 + "\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer.load(args.tokenizer)
    print(f"Loaded tokenizer: {len(tokenizer)} tokens\n")

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

    print(f"Loaded model: {base_model.get_num_params():,} parameters\n")

    # Create continual learner
    print("Setting up continual learner...")

    config = ContinualLearningConfig(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        replay_buffer_size=args.replay_buffer_size,
        replay_ratio=args.replay_ratio,
        use_ewc=args.use_ewc,
        ewc_lambda=args.ewc_lambda,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        consolidation_frequency=args.sleep_freq if args.sleep_freq > 0 else 10000,
        device=args.device
    )

    learner = ContinualLearner(
        base_model=base_model,
        config=config,
        adapter_name=args.adapter_name,
        tokenizer=tokenizer
    )

    print("Learner ready\n")
    learner.model.print_trainable_parameters()

    # Load training data
    print(f"\nLoading training data from {args.data}...")
    with open(args.data, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    if args.max_samples:
        texts = texts[:args.max_samples]

    print(f"Loaded {len(texts)} training samples\n")

    # Initialize sleep consolidation if enabled
    sleep_consolidator = None
    if args.sleep_freq > 0:
        from src.neurobio.sleep_consolidation import SleepConsolidation, SleepConfig
        sleep_config = SleepConfig(
            num_replay_cycles=args.sleep_cycles,
            replay_noise_std=0.1,
            hebbian_lr=0.001,
            pruning_threshold=0.01
        )
        sleep_consolidator = SleepConsolidation(
            model=learner.model,
            replay_buffer=learner.replay_buffer,
            config=sleep_config
        )
        print(f"Sleep consolidation enabled (every {args.sleep_freq} steps)\n")

    # Train
    print("Starting training...")
    print(f"  Domain: {args.domain}")
    print(f"  Sample weight: {args.importance}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print()

    losses = []
    step = 0

    from tqdm import tqdm
    for i, text in enumerate(tqdm(texts, desc="Training")):
        # Learn from sample
        stats = learner.learn_from_text(
            text=text,
            domain=args.domain,
            importance=args.importance
        )

        if stats:
            losses.append(stats.get('total_loss', 0))
            step += 1

            # Run sleep consolidation
            if sleep_consolidator and step % args.sleep_freq == 0:
                print(f"\n[Step {step}] Running sleep consolidation...")
                sleep_stats = sleep_consolidator.consolidate(verbose=False)
                print(f"  Strengthened: {sleep_stats['connections_strengthened']:,}, "
                      f"Pruned: {sleep_stats['connections_pruned']:,}")

            # Save checkpoint
            if args.save_freq > 0 and (i + 1) % args.save_freq == 0:
                checkpoint_path = f"{args.checkpoint_dir}/{args.adapter_name}_step{step}"
                learner.save_checkpoint(checkpoint_path)
                print(f"[Step {step}] Saved checkpoint to {checkpoint_path}")

    avg_loss = sum(losses) / len(losses) if losses else 0

    print(f"\nTraining complete!")
    print(f"  Samples processed: {len(texts)}")
    print(f"  Updates: {step}")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}" if losses else "")
    print()

    # Print training stats
    learner.print_stats()

    # Final checkpoint
    final_path = f"{args.checkpoint_dir}/{args.adapter_name}_final"
    print(f"\nSaving final checkpoint to {final_path}...")
    learner.save_checkpoint(final_path)
    print(f"Saved to {final_path}")

    # Test generation
    if args.test_prompt:
        print(f"\n{'='*60}")
        print("Testing Generation")
        print("="*60)
        print(f"Prompt: {args.test_prompt}\n")

        try:
            output = learner.generate(
                prompt=args.test_prompt,
                max_new_tokens=args.test_max_tokens,
                temperature=0.8,
                top_k=40,
                top_p=0.9
            )
            print(f"Generated:\n{output}\n")
        except Exception as e:
            print(f"Generation failed: {e}\n")

    print("="*60)
    print("Training Complete")
    print("="*60)


if __name__ == "__main__":
    main()
