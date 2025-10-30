"""
Script to train a BPE tokenizer.

Usage:
    python scripts/train_tokenizer.py --data data/train.txt --vocab_size 32000 --save tokenizers/my_tokenizer
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.tokenization.vocab_trainer import (
    train_tokenizer_from_files,
    train_tokenizer_from_directory,
    test_tokenizer
)


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")

    # Data
    parser.add_argument("--data", type=str, nargs='+', required=True,
                       help="Path to training data file(s)")
    parser.add_argument("--data_dir", type=str,
                       help="Alternative: directory with training data")
    parser.add_argument("--file_pattern", type=str, default="*.txt",
                       help="File pattern for data_dir (default: *.txt)")

    # Tokenizer config
    parser.add_argument("--vocab_size", type=int, default=32000,
                       help="Target vocabulary size (default: 32000)")
    parser.add_argument("--min_frequency", type=int, default=2,
                       help="Minimum frequency for merges (default: 2)")

    # Special tokens
    parser.add_argument("--special_tokens", type=str, nargs='+',
                       default=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
                       help="Special tokens to add")

    # Limits
    parser.add_argument("--max_lines", type=int,
                       help="Maximum lines to read (default: all)")
    parser.add_argument("--max_files", type=int,
                       help="Maximum files to process from directory")

    # Output
    parser.add_argument("--save", type=str, required=True,
                       help="Path to save trained tokenizer")

    # Testing
    parser.add_argument("--test", action="store_true",
                       help="Test tokenizer after training")
    parser.add_argument("--test_texts", type=str, nargs='+',
                       default=["The quick brown fox", "Machine learning is amazing"],
                       help="Texts to use for testing")

    args = parser.parse_args()

    print("="*60)
    print("BPE Tokenizer Training")
    print("="*60)
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Min frequency: {args.min_frequency}")
    print(f"Special tokens: {args.special_tokens}")
    print(f"Save path: {args.save}")
    print("="*60 + "\n")

    # Train from files or directory
    if args.data_dir:
        print(f"Training from directory: {args.data_dir}")
        tokenizer = train_tokenizer_from_directory(
            directory=args.data_dir,
            vocab_size=args.vocab_size,
            file_pattern=args.file_pattern,
            min_frequency=args.min_frequency,
            special_tokens=args.special_tokens,
            max_files=args.max_files,
            save_path=args.save
        )
    else:
        print(f"Training from {len(args.data)} file(s)")
        tokenizer = train_tokenizer_from_files(
            file_paths=args.data,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=args.special_tokens,
            max_lines=args.max_lines,
            save_path=args.save
        )

    print(f"\n✓ Tokenizer trained and saved to {args.save}")
    print(f"  Final vocab size: {len(tokenizer)}")

    # Test if requested
    if args.test:
        test_tokenizer(tokenizer, args.test_texts)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
