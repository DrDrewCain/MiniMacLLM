"""
Byte-Pair Encoding (BPE) Tokenizer from scratch.

BPE is the most widely used tokenization algorithm for LLMs:
- GPT-2, GPT-3, GPT-4 (tiktoken)
- LLaMA 2 (SentencePiece)
- Most modern LLMs

Algorithm:
1. Start with character-level tokens
2. Iteratively merge most frequent byte pairs
3. Build vocabulary of subwords

References:
    - "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)
    - Used in GPT, BERT, RoBERTa, etc.
"""

import json
import regex as re
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
from pathlib import Path
import pickle


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer.

    Implements the BPE algorithm for subword tokenization.

    Args:
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for a merge
        special_tokens: List of special tokens to add

    Example:
        >>> tokenizer = BPETokenizer(vocab_size=10000)
        >>> tokenizer.train(["Hello world", "Hello there"])
        >>> tokens = tokenizer.encode("Hello")
        >>> text = tokenizer.decode(tokens)
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        # Special tokens
        if special_tokens is None:
            special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.special_tokens = special_tokens

        # Vocabulary and merges
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

        # Regex pattern for pre-tokenization (splits on whitespace and punctuation)
        # This pattern is similar to GPT-2's tokenizer
        self.pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.IGNORECASE
        )

        # Cache for encoding
        self.cache: Dict[str, List[int]] = {}

    def train(self, texts: List[str], verbose: bool = True):
        """
        Train BPE tokenizer on a corpus.

        Args:
            texts: List of text strings to train on
            verbose: Whether to print progress
        """
        if verbose:
            print(f"Training BPE tokenizer on {len(texts)} texts...")
            print(f"Target vocabulary size: {self.vocab_size}")

        # Step 1: Pre-tokenize into words
        words = []
        for text in texts:
            words.extend(self.pattern.findall(text))

        if verbose:
            print(f"Pre-tokenized into {len(words)} words")

        # Step 2: Split words into characters and count frequencies
        word_freqs = Counter(words)

        # Convert words to character sequences
        # We use spaces to separate characters: "hello" -> "h e l l o</w>"
        # The </w> marker indicates end of word
        splits = {word: [c for c in word] + ['</w>'] for word in word_freqs.keys()}

        # Step 3: Initialize vocabulary with all characters
        vocab = set()
        for word in splits.values():
            vocab.update(word)

        # Add special tokens first
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        start_idx = len(self.special_tokens)

        # Add base characters
        for idx, char in enumerate(sorted(vocab)):
            self.vocab[char] = start_idx + idx

        if verbose:
            print(f"Initial vocabulary size: {len(self.vocab)}")

        # Step 4: Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(self.vocab)

        for merge_idx in range(num_merges):
            # Count all pairs
            pair_freqs = defaultdict(int)

            for word, freq in word_freqs.items():
                split = splits[word]
                if len(split) < 2:
                    continue

                # Count pairs in this word
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            pair_freq = pair_freqs[best_pair]

            if pair_freq < self.min_frequency:
                break

            # Merge this pair in all words
            new_token = best_pair[0] + best_pair[1]

            # Update splits
            for word in word_freqs:
                split = splits[word]
                i = 0
                new_split = []

                while i < len(split):
                    if i < len(split) - 1 and (split[i], split[i + 1]) == best_pair:
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1

                splits[word] = new_split

            # Add to vocabulary
            self.vocab[new_token] = len(self.vocab)

            # Record merge
            self.merges[best_pair] = new_token
            self.merge_ranks[best_pair] = merge_idx

            if verbose and (merge_idx + 1) % 500 == 0:
                print(f"Merge {merge_idx + 1}/{num_merges}: "
                      f"{best_pair[0]} + {best_pair[1]} = {new_token} "
                      f"(freq: {pair_freq})")

        # Create inverse vocabulary
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

        if verbose:
            print(f"\nTraining complete!")
            print(f"Final vocabulary size: {len(self.vocab)}")
            print(f"Number of merges: {len(self.merges)}")

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using learned merges.

        Args:
            word: Word to tokenize

        Returns:
            List of subword tokens
        """
        # Start with character-level split
        tokens = [c for c in word] + ['</w>']

        # Apply merges in order of their rank
        while len(tokens) > 1:
            # Find all possible pairs and their ranks
            pairs = [(tokens[i], tokens[i + 1], self.merge_ranks.get((tokens[i], tokens[i + 1]), float('inf')))
                    for i in range(len(tokens) - 1)]

            # Find pair with lowest rank (earliest merge)
            if not pairs:
                break

            best_pair = min(pairs, key=lambda x: x[2])

            if best_pair[2] == float('inf'):
                # No more valid merges
                break

            # Merge this pair
            pair_to_merge = (best_pair[0], best_pair[1])
            new_token = self.merges[pair_to_merge]

            # Replace all occurrences
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair_to_merge:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        # Check cache
        cache_key = (text, add_special_tokens)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Pre-tokenize into words
        words = self.pattern.findall(text)

        # Tokenize each word
        tokens = []

        if add_special_tokens:
            tokens.append(self.vocab["<BOS>"])

        for word in words:
            word_tokens = self._tokenize_word(word)

            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    # Unknown token
                    tokens.append(self.vocab["<UNK>"])

        if add_special_tokens:
            tokens.append(self.vocab["<EOS>"])

        # Cache result
        if len(self.cache) < 10000:  # Limit cache size
            self.cache[cache_key] = tokens

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text
        """
        tokens = []

        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]

                # Skip special tokens if requested
                if skip_special_tokens and token in self.special_tokens:
                    continue

                tokens.append(token)
            else:
                tokens.append("<UNK>")

        # Join tokens and remove end-of-word markers
        text = ''.join(tokens).replace('</w>', ' ')

        return text.strip()

    def save(self, save_path: str):
        """
        Save tokenizer to disk.

        Args:
            save_path: Path to save tokenizer
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save vocabulary
        with open(save_path / "vocab.json", 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save merges
        merges_list = [{"pair": list(pair), "token": token, "rank": self.merge_ranks[pair]}
                      for pair, token in self.merges.items()]

        with open(save_path / "merges.json", 'w', encoding='utf-8') as f:
            json.dump(merges_list, f, ensure_ascii=False, indent=2)

        # Save config
        config = {
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'special_tokens': self.special_tokens
        }

        with open(save_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"Tokenizer saved to {save_path}")

    @classmethod
    def load(cls, load_path: str) -> 'BPETokenizer':
        """
        Load tokenizer from disk.

        Args:
            load_path: Path to load from

        Returns:
            Loaded tokenizer
        """
        load_path = Path(load_path)

        # Load config
        with open(load_path / "config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Create tokenizer
        tokenizer = cls(
            vocab_size=config['vocab_size'],
            min_frequency=config['min_frequency'],
            special_tokens=config['special_tokens']
        )

        # Load vocabulary
        with open(load_path / "vocab.json", 'r', encoding='utf-8') as f:
            tokenizer.vocab = json.load(f)

        tokenizer.inverse_vocab = {int(idx): token for token, idx in tokenizer.vocab.items()}

        # Load merges
        with open(load_path / "merges.json", 'r', encoding='utf-8') as f:
            merges_list = json.load(f)

        for merge_data in merges_list:
            pair = tuple(merge_data['pair'])
            tokenizer.merges[pair] = merge_data['token']
            tokenizer.merge_ranks[pair] = merge_data['rank']

        print(f"Tokenizer loaded from {load_path}")

        return tokenizer

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary."""
        return self.vocab.copy()

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    def __repr__(self) -> str:
        """String representation."""
        return (f"BPETokenizer(vocab_size={len(self.vocab)}, "
                f"merges={len(self.merges)}, "
                f"special_tokens={len(self.special_tokens)})")


if __name__ == "__main__":
    # Test BPE tokenizer
    print("Testing BPE Tokenizer...")

    # Create sample corpus
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "The dog was not amused by the fox.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning models can learn complex patterns from data.",
        "Natural language processing enables computers to understand human language.",
        "Python is a high-level programming language.",
        "Python programming is used for data science and machine learning.",
        "The transformer architecture revolutionized natural language processing."
    ]

    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size=500, min_frequency=2)
    tokenizer.train(corpus, verbose=True)

    # Test encoding
    print("\n" + "="*60)
    print("Testing Encoding/Decoding")
    print("="*60)

    test_text = "The quick brown fox"
    print(f"\nOriginal text: '{test_text}'")

    tokens = tokenizer.encode(test_text, add_special_tokens=True)
    print(f"Encoded tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")

    # Show token strings
    token_strings = [tokenizer.inverse_vocab.get(t, "<UNK>") for t in tokens]
    print(f"Token strings: {token_strings}")

    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"Decoded text: '{decoded}'")
    print(f"Match: {test_text.lower() == decoded.lower()}")

    # Test save/load
    print("\n" + "="*60)
    print("Testing Save/Load")
    print("="*60)

    save_dir = "test_tokenizer"
    tokenizer.save(save_dir)

    loaded_tokenizer = BPETokenizer.load(save_dir)
    print(f"\nLoaded tokenizer: {loaded_tokenizer}")

    # Test loaded tokenizer
    loaded_tokens = loaded_tokenizer.encode(test_text, add_special_tokens=True)
    print(f"Tokens match: {tokens == loaded_tokens}")

    # Cleanup
    import shutil
    shutil.rmtree(save_dir)

    print("\n✓ BPE Tokenizer implementation complete!")
