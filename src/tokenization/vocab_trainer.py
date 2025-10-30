"""
Vocabulary training utilities for BPE tokenizer.

Provides functions to:
- Train tokenizer on text files
- Train on datasets
- Create vocabularies from corpora
"""

import os
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from .bpe_tokenizer import BPETokenizer


def train_tokenizer_from_files(
    file_paths: List[str],
    vocab_size: int = 32000,
    min_frequency: int = 2,
    special_tokens: Optional[List[str]] = None,
    max_lines: Optional[int] = None,
    save_path: Optional[str] = None,
) -> BPETokenizer:
    """
    Train BPE tokenizer from text files.

    Args:
        file_paths: List of paths to text files
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for merges
        special_tokens: Special tokens to add
        max_lines: Maximum lines to read (None = all)
        save_path: Path to save trained tokenizer (None = don't save)

    Returns:
        Trained tokenizer

    Example:
        >>> tokenizer = train_tokenizer_from_files(
        ...     ["data/train.txt"],
        ...     vocab_size=10000,
        ...     save_path="tokenizers/my_tokenizer"
        ... )
    """
    print(f"Training tokenizer from {len(file_paths)} file(s)...")

    # Read texts
    texts = []
    total_lines = 0

    for file_path in file_paths:
        print(f"Reading {file_path}...")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            file_texts = []

            for line_num, line in enumerate(f):
                if max_lines and total_lines >= max_lines:
                    break

                line = line.strip()
                if line:  # Skip empty lines
                    file_texts.append(line)
                    total_lines += 1

            texts.extend(file_texts)

        print(f"  Read {len(file_texts)} lines")

    print(f"\nTotal lines: {total_lines}")

    # Create and train tokenizer
    tokenizer = BPETokenizer(
        vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens
    )

    tokenizer.train(texts, verbose=True)

    # Save if requested
    if save_path:
        tokenizer.save(save_path)

    return tokenizer


def train_tokenizer_from_directory(
    directory: str,
    vocab_size: int = 32000,
    file_pattern: str = "*.txt",
    min_frequency: int = 2,
    special_tokens: Optional[List[str]] = None,
    max_files: Optional[int] = None,
    max_lines_per_file: Optional[int] = None,
    save_path: Optional[str] = None,
) -> BPETokenizer:
    """
    Train tokenizer from all files in a directory.

    Args:
        directory: Path to directory
        vocab_size: Target vocabulary size
        file_pattern: Glob pattern for files (e.g., "*.txt", "*.py")
        min_frequency: Minimum frequency for merges
        special_tokens: Special tokens to add
        max_files: Maximum files to process
        max_lines_per_file: Maximum lines per file
        save_path: Path to save tokenizer

    Returns:
        Trained tokenizer

    Example:
        >>> tokenizer = train_tokenizer_from_directory(
        ...     "data/corpus",
        ...     vocab_size=10000,
        ...     file_pattern="*.txt"
        ... )
    """
    directory = Path(directory)

    # Find all matching files
    file_paths = list(directory.rglob(file_pattern))

    if max_files:
        file_paths = file_paths[:max_files]

    print(f"Found {len(file_paths)} files matching '{file_pattern}'")

    # Read texts from all files
    texts = []

    for file_path in tqdm(file_paths, desc="Reading files"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_texts = []

                for line_num, line in enumerate(f):
                    if max_lines_per_file and line_num >= max_lines_per_file:
                        break

                    line = line.strip()
                    if line:
                        file_texts.append(line)

                texts.extend(file_texts)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    print(f"Total lines collected: {len(texts)}")

    # Train tokenizer
    tokenizer = BPETokenizer(
        vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens
    )

    tokenizer.train(texts, verbose=True)

    if save_path:
        tokenizer.save(save_path)

    return tokenizer


def train_tokenizer_from_texts(
    texts: List[str],
    vocab_size: int = 32000,
    min_frequency: int = 2,
    special_tokens: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> BPETokenizer:
    """
    Train tokenizer directly from list of texts.

    Args:
        texts: List of text strings
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for merges
        special_tokens: Special tokens to add
        save_path: Path to save tokenizer

    Returns:
        Trained tokenizer

    Example:
        >>> texts = ["Hello world", "Machine learning"]
        >>> tokenizer = train_tokenizer_from_texts(texts, vocab_size=1000)
    """
    tokenizer = BPETokenizer(
        vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens
    )

    tokenizer.train(texts, verbose=True)

    if save_path:
        tokenizer.save(save_path)

    return tokenizer


def create_domain_specific_tokenizer(
    domain: str, data_paths: List[str], vocab_size: int = 32000, save_path: Optional[str] = None
) -> BPETokenizer:
    """
    Create tokenizer optimized for specific domain.

    Adds domain-specific special tokens and trains on domain data.

    Args:
        domain: Domain name ("code", "math", "medical", etc.)
        data_paths: Paths to domain-specific data
        vocab_size: Target vocabulary size
        save_path: Path to save tokenizer

    Returns:
        Domain-specific tokenizer

    Example:
        >>> tokenizer = create_domain_specific_tokenizer(
        ...     domain="code",
        ...     data_paths=["data/python_code.txt"],
        ...     vocab_size=15000
        ... )
    """
    # Define domain-specific special tokens
    domain_special_tokens = {
        "code": ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<INDENT>", "<DEDENT>", "<NEWLINE>"],
        "math": ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<EQUATION>", "<PROOF>", "<THEOREM>"],
        "medical": ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<DIAGNOSIS>", "<SYMPTOM>", "<TREATMENT>"],
        "general": ["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
    }

    special_tokens = domain_special_tokens.get(domain, domain_special_tokens["general"])

    print(f"Creating {domain} domain tokenizer...")
    print(f"Special tokens: {special_tokens}")

    return train_tokenizer_from_files(
        file_paths=data_paths,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        save_path=save_path,
    )


def test_tokenizer(tokenizer: BPETokenizer, test_texts: List[str]):
    """
    Test tokenizer on sample texts.

    Args:
        tokenizer: Trained tokenizer
        test_texts: List of test texts

    Prints encoding/decoding results.
    """
    print("\n" + "=" * 60)
    print("Testing Tokenizer")
    print("=" * 60)

    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: '{text}'")

        # Encode
        tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"  Tokens ({len(tokens)}): {tokens}")

        # Show token strings
        token_strings = [tokenizer.inverse_vocab.get(t, "<UNK>") for t in tokens]
        print(f"  Subwords: {token_strings}")

        # Decode
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"  Decoded: '{decoded}'")

        # Check match
        match = text.lower().strip() == decoded.lower().strip()
        print(f"  Match: {'✓' if match else '✗'}")


def get_tokenizer_stats(tokenizer: BPETokenizer, texts: List[str]) -> dict:
    """
    Get statistics about tokenizer performance.

    Args:
        tokenizer: Trained tokenizer
        texts: List of texts to analyze

    Returns:
        Dictionary with statistics
    """
    total_chars = sum(len(text) for text in texts)
    total_tokens = sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)
    total_words = sum(len(text.split()) for text in texts)

    return {
        "vocab_size": len(tokenizer),
        "total_texts": len(texts),
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "avg_chars_per_token": total_chars / total_tokens if total_tokens > 0 else 0,
        "compression_ratio": total_chars / total_tokens if total_tokens > 0 else 0,
        "tokens_per_word": total_tokens / total_words if total_words > 0 else 0,
    }


if __name__ == "__main__":
    # Example usage
    print("Tokenizer Training Examples")
    print("=" * 60)

    # Create sample data
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Python programming is fun and useful.",
        "Natural language processing with transformers.",
        "Deep learning models require large datasets.",
    ]

    # Example 1: Train from texts
    print("\nExample 1: Train from texts")
    tokenizer = train_tokenizer_from_texts(
        texts=sample_texts * 10,  # Repeat for more data
        vocab_size=300,
        save_path="test_tokenizer_output",
    )

    # Test it
    test_texts = ["The quick fox", "Machine learning transformers", "Python programming"]

    test_tokenizer(tokenizer, test_texts)

    # Get stats
    stats = get_tokenizer_stats(tokenizer, test_texts)
    print("\nTokenizer Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Cleanup
    import shutil

    if os.path.exists("test_tokenizer_output"):
        shutil.rmtree("test_tokenizer_output")

    print("\n✓ Vocabulary trainer implementation complete!")
