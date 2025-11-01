"""
Byte-Level BPE Tokenizer.

Modern byte-level BPE implementing published tokenization techniques:
- Byte-level encoding for robust Unicode handling
- Byte-to-unicode mapping for printable characters
- Efficient regex patterns and caching

Key improvements over naive BPE:
1. Byte-level encoding - handles ANY Unicode without unknown tokens
2. NFC normalization - consistent text representation
3. LRU caching - fast repeated encodings
4. Optimized regex - better pre-tokenization
5. Proper special token handling

Technical References:
    - Sennrich et al., 2016. "Neural Machine Translation of Rare Words with
      Subword Units" ACL 2016 (BPE foundation)
    - Radford et al., 2019. "Language Models are Unsupervised Multitask Learners"
      (Byte-level BPE approach)
    - Wang et al., 2024. "Qwen2.5 Technical Report" arXiv:2412.15115
      (Advanced byte-level encoding techniques)
"""

import json
import regex as re
import unicodedata
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path
from functools import lru_cache
from enum import Enum
from dataclasses import dataclass


class PaddingStrategy(str, Enum):
    """Padding strategies for batch encoding."""
    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TruncationStrategy(str, Enum):
    """Truncation strategies for sequence handling."""
    LONGEST_FIRST = "longest_first"
    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    DO_NOT_TRUNCATE = "do_not_truncate"


@dataclass
class BatchEncoding:
    """
    Container for batch tokenization outputs.

    Attributes:
        input_ids: List of token IDs for each sequence
        attention_mask: List of attention masks (1 for real tokens, 0 for padding)
        offset_mapping: Optional character offsets for each token
        token_type_ids: Optional token type IDs for sequence pairs
    """
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]] = None
    offset_mapping: Optional[List[List[Tuple[int, int]]]] = None
    token_type_ids: Optional[List[List[int]]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        result = {"input_ids": self.input_ids}
        if self.attention_mask is not None:
            result["attention_mask"] = self.attention_mask
        if self.offset_mapping is not None:
            result["offset_mapping"] = self.offset_mapping
        if self.token_type_ids is not None:
            result["token_type_ids"] = self.token_type_ids
        return result


@lru_cache()
def bytes_to_unicode():
    """
    Create bijective mapping from bytes to Unicode strings (byte-level encoding approach).

    Returns a dictionary mapping all 256 byte values to printable Unicode characters.
    Avoids mapping to whitespace/control characters by using higher Unicode codepoints.

    This ensures:
    - All bytes 0-255 are representable as single Unicode characters
    - No conflicts with actual text characters
    - Reversible encoding/decoding

    Returns:
        Dict[int, str]: Mapping from byte (0-255) to Unicode character
    """
    # Start with printable ASCII (excludes whitespace/control chars)
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0

    # Map remaining bytes to unused Unicode range
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1

    # Convert to characters
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class BPETokenizer:
    """
    Modern Byte-Level BPE Tokenizer.

    Implements byte-level BPE for robust tokenization of ANY Unicode text.

    Key Features:
    - Byte-level encoding: No unknown tokens, handles all Unicode
    - NFC normalization: Consistent representation
    - Efficient caching: LRU cache + token cache
    - Advanced regex: Better pre-tokenization patterns
    - Special token support: Proper handling of control tokens

    Args:
        vocab_size: Target vocabulary size (default: 32000)
        min_frequency: Minimum pair frequency for merging (default: 2)
        special_tokens: List of special tokens (default: ["<|endoftext|>"])
        normalization: Unicode normalization form ('NFC', 'NFKC', 'NFD', 'NFKD', or None)

    Example:
        >>> tokenizer = BPETokenizer(vocab_size=10000)
        >>> tokenizer.train(["Hello world! 你好🌍", "More text"])
        >>> tokens = tokenizer.encode("Hello 你好")
        >>> text = tokenizer.decode(tokens)
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        normalization: str = "NFC",
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.normalization = normalization

        # Special tokens (minimal by default)
        if special_tokens is None:
            special_tokens = ["<|endoftext|>"]  # Standard end-of-text marker
        self.special_tokens = special_tokens

        # Byte encoder/decoder for byte-level BPE
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Vocabulary and merges
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

        # Advanced regex pattern for pre-tokenization
        # Handles: contractions, words, numbers, punctuation, whitespace
        self.pattern = re.compile(
            r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        )

        # BPE cache with LRU eviction for faster repeated tokenization
        self.cache: OrderedDict[str, Tuple[int, ...]] = OrderedDict()
        self.cache_maxsize = 50000

    def _normalize_text(self, text: str) -> str:
        """Apply Unicode normalization if configured."""
        if self.normalization:
            return unicodedata.normalize(self.normalization, text)
        return text

    def train(self, texts: List[str], verbose: bool = True):
        """
        Train byte-level BPE tokenizer on a corpus.

        Uses byte-level encoding to ensure ALL Unicode is handled without unknown tokens.

        Args:
            texts: List of text strings to train on
            verbose: Whether to print progress
        """
        # Reset all learned state to ensure clean training
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = {}
        self.merge_ranks = {}
        self.cache.clear()

        if verbose:
            print(f"Training byte-level BPE tokenizer on {len(texts)} texts...")
            print(f"Target vocabulary size: {self.vocab_size}")

        # Step 1: Pre-tokenize into words and convert to byte-level
        word_freqs = Counter()
        for text in texts:
            # Normalize text first
            normalized = self._normalize_text(text)
            # Pre-tokenize using advanced regex pattern
            words = self.pattern.findall(normalized)

            for word in words:
                # Convert to byte-level representation
                byte_word = "".join(self.byte_encoder[b] for b in word.encode("utf-8"))
                word_freqs[byte_word] += 1

        if verbose:
            print(f"Pre-tokenized into {len(word_freqs)} unique words")
            print(f"Total word occurrences: {sum(word_freqs.values())}")

        # Step 2: Initialize vocabulary with 256 byte tokens
        # Start with all possible bytes (0-255) mapped to Unicode
        base_vocab = ["".join(self.byte_encoder[b] for b in bytes([i])) for i in range(256)]

        # Add special tokens first
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        start_idx = len(self.special_tokens)

        # Add all 256 byte tokens
        for idx, byte_token in enumerate(base_vocab):
            self.vocab[byte_token] = start_idx + idx

        if verbose:
            print(
                f"Initial vocabulary size: {len(self.vocab)} ({len(self.special_tokens)} special + 256 bytes)"
            )

        # Step 3: Split words into byte-level characters
        splits = {word: list(word) for word in word_freqs.keys()}

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
                print(
                    f"Merge {merge_idx + 1}/{num_merges}: "
                    f"{best_pair[0]} + {best_pair[1]} = {new_token} "
                    f"(freq: {pair_freq})"
                )

        # Create inverse vocabulary
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

        if verbose:
            print(f"\nTraining complete!")
            print(f"Final vocabulary size: {len(self.vocab)}")
            print(f"Number of merges: {len(self.merges)}")

    def _tokenize_word(self, word: str) -> Tuple[int, ...]:
        """
        Tokenize a single word using learned merges (byte-level).

        Converts word to byte-level representation, applies BPE merges,
        then returns token IDs. Uses caching for efficiency.

        Args:
            word: Word to tokenize (will be converted to bytes)

        Returns:
            Tuple of token IDs (tuple for hashability in cache)
        """
        # Check cache first (and move to end for LRU)
        if word in self.cache:
            self.cache.move_to_end(word)
            return self.cache[word]

        # Convert to byte-level representation
        byte_word = "".join(self.byte_encoder[b] for b in word.encode("utf-8"))

        # Start with byte-level character split
        tokens = list(byte_word)

        # Apply merges in order of their rank (optimized to avoid O(n²))
        while len(tokens) > 1:
            # Find all possible pairs and their ranks
            pairs = [
                (
                    i,
                    tokens[i],
                    tokens[i + 1],
                    self.merge_ranks.get((tokens[i], tokens[i + 1]), float("inf")),
                )
                for i in range(len(tokens) - 1)
            ]

            if not pairs:
                break

            # Find pair with lowest rank (earliest merge)
            best_pair = min(pairs, key=lambda x: x[3])

            if best_pair[3] == float("inf"):
                # No more valid merges
                break

            # Merge the best pair efficiently
            i, first, second, _ = best_pair
            pair_to_merge = (first, second)
            new_token = self.merges[pair_to_merge]

            # Merge tokens in-place using slice assignment for O(n) per merge
            # This is more efficient than list unpacking which creates a new list
            tokens[i:i + 2] = [new_token]

        # Convert tokens to IDs
        # Note: With byte-level BPE, all tokens MUST exist in vocab after training
        token_ids = []
        for token in tokens:
            if token not in self.vocab:
                raise ValueError(
                    f"Token '{token}' not found in vocabulary. "
                    "This should not happen with byte-level BPE. "
                    "Ensure the tokenizer has been properly trained."
                )
            token_ids.append(self.vocab[token])
        token_ids = tuple(token_ids)

        # Cache result with LRU eviction
        if len(self.cache) >= self.cache_maxsize:
            # Remove oldest entry (first item in OrderedDict)
            self.cache.popitem(last=False)
        self.cache[word] = token_ids

        return token_ids

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs using byte-level BPE.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add endoftext token

        Returns:
            List of token IDs
        """
        # Normalize text
        normalized = self._normalize_text(text)

        # Pre-tokenize using advanced regex pattern
        words = self.pattern.findall(normalized)

        # Tokenize each word
        tokens = []

        for word in words:
            word_token_ids = self._tokenize_word(word)
            tokens.extend(word_token_ids)

        if add_special_tokens and self.special_tokens:
            # Add endoftext token at the end
            tokens.append(self.vocab[self.special_tokens[0]])

        return list(tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text using byte-level decoding.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text
        """
        # Convert IDs to tokens
        tokens = []

        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]

                # Skip special tokens if requested
                if skip_special_tokens and token in self.special_tokens:
                    continue

                tokens.append(token)

        # Join tokens and decode from byte-level representation
        byte_string = "".join(tokens)

        # Convert byte-level characters back to actual bytes
        try:
            byte_array = bytearray([self.byte_decoder[c] for c in byte_string])
            # Decode from UTF-8
            text = byte_array.decode("utf-8", errors="replace")
        except (KeyError, UnicodeDecodeError):
            # Fallback for malformed sequences
            text = byte_string

        return text

    def save(self, save_path: str):
        """
        Save tokenizer to disk.

        Args:
            save_path: Path to save tokenizer
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save vocabulary
        with open(save_path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save merges
        merges_list = [
            {"pair": list(pair), "token": token, "rank": self.merge_ranks[pair]}
            for pair, token in self.merges.items()
        ]

        with open(save_path / "merges.json", "w", encoding="utf-8") as f:
            json.dump(merges_list, f, ensure_ascii=False, indent=2)

        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
            "normalization": self.normalization,
        }

        with open(save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"✓ Byte-level BPE tokenizer saved to {save_path}")

    @classmethod
    def load(cls, load_path: str) -> "BPETokenizer":
        """
        Load tokenizer from disk.

        Args:
            load_path: Path to load from

        Returns:
            Loaded tokenizer
        """
        load_path = Path(load_path)

        # Load config
        with open(load_path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        # Create tokenizer
        tokenizer = cls(
            vocab_size=config["vocab_size"],
            min_frequency=config["min_frequency"],
            special_tokens=config["special_tokens"],
            normalization=config.get("normalization", "NFC"),
        )

        # Load vocabulary
        with open(load_path / "vocab.json", "r", encoding="utf-8") as f:
            tokenizer.vocab = json.load(f)

        tokenizer.inverse_vocab = {int(idx): token for token, idx in tokenizer.vocab.items()}

        # Load merges
        with open(load_path / "merges.json", "r", encoding="utf-8") as f:
            merges_list = json.load(f)

        for merge_data in merges_list:
            pair = tuple(merge_data["pair"])
            tokenizer.merges[pair] = merge_data["token"]
            tokenizer.merge_ranks[pair] = merge_data["rank"]

        print(f"✓ Byte-level BPE tokenizer loaded from {load_path}")

        return tokenizer

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary."""
        return self.vocab.copy()

    @property
    def pad_token_id(self) -> Optional[int]:
        """Get padding token ID (uses first special token)."""
        if self.special_tokens:
            return self.vocab.get(self.special_tokens[0])
        return None

    @property
    def eos_token_id(self) -> Optional[int]:
        """Get end-of-sequence token ID."""
        if self.special_tokens:
            return self.vocab.get(self.special_tokens[0])
        return None

    def encode_plus(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> Dict:
        """
        Encode text with additional features like offset mapping.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens
            return_offsets_mapping: Whether to return character offsets

        Returns:
            Dictionary with input_ids and optionally offset_mapping
        """
        # Normalize text
        normalized = self._normalize_text(text)

        # Pre-tokenize using regex
        words = self.pattern.findall(normalized)

        # Track character positions if needed
        tokens = []
        offsets = [] if return_offsets_mapping else None
        char_pos = 0

        for word in words:
            # Find word position in normalized text
            word_start = normalized.find(word, char_pos)
            word_token_ids = self._tokenize_word(word)
            tokens.extend(word_token_ids)

            # Calculate offsets if requested
            if return_offsets_mapping:
                # Approximate offset (byte-level makes this complex)
                chars_per_token = len(word) / len(word_token_ids) if word_token_ids else 0
                for i, _ in enumerate(word_token_ids):
                    start = int(word_start + i * chars_per_token)
                    end = int(word_start + (i + 1) * chars_per_token)
                    offsets.append((start, min(end, word_start + len(word))))

            char_pos = word_start + len(word)

        if add_special_tokens and self.special_tokens:
            tokens.append(self.vocab[self.special_tokens[0]])
            if return_offsets_mapping:
                offsets.append((len(normalized), len(normalized)))

        result = {"input_ids": tokens}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets

        return result

    def batch_encode_plus(
        self,
        texts: List[str],
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        add_special_tokens: bool = False,
        return_attention_mask: bool = True,
        return_offsets_mapping: bool = False,
    ) -> BatchEncoding:
        """
        Encode multiple texts with padding and truncation.

        Args:
            texts: List of texts to encode
            padding: Padding strategy ("longest", "max_length", or False)
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            add_special_tokens: Whether to add special tokens
            return_attention_mask: Whether to return attention masks
            return_offsets_mapping: Whether to return character offsets

        Returns:
            BatchEncoding with input_ids, attention_mask, and optionally offset_mapping

        Example:
            >>> tokenizer = BPETokenizer(vocab_size=1000)
            >>> tokenizer.train(["Hello world", "Test"])
            >>> result = tokenizer.batch_encode_plus(
            ...     ["Hello", "Hi there!"],
            ...     padding="longest",
            ...     return_attention_mask=True
            ... )
            >>> result.input_ids  # [[1, 2], [3, 4, 0]]
            >>> result.attention_mask  # [[1, 1], [1, 1, 0]]
        """
        # Encode all texts
        all_input_ids = []
        all_offsets = [] if return_offsets_mapping else None

        for text in texts:
            encoded = self.encode_plus(
                text,
                add_special_tokens=add_special_tokens,
                return_offsets_mapping=return_offsets_mapping,
            )
            all_input_ids.append(encoded["input_ids"])
            if return_offsets_mapping:
                all_offsets.append(encoded["offset_mapping"])

        # Apply truncation
        if truncation and max_length:
            all_input_ids = [ids[:max_length] for ids in all_input_ids]
            if return_offsets_mapping:
                all_offsets = [offs[:max_length] for offs in all_offsets]

        # Apply padding
        attention_masks = None
        if padding:
            # Determine target length
            if padding == "max_length" or padding == PaddingStrategy.MAX_LENGTH:
                if max_length is None:
                    raise ValueError("max_length must be specified with 'max_length' padding")
                target_length = max_length
            elif padding == "longest" or padding == PaddingStrategy.LONGEST:
                target_length = max(len(ids) for ids in all_input_ids)
            else:
                target_length = 0  # No padding

            # Pad sequences
            if target_length > 0:
                pad_id = self.pad_token_id if self.pad_token_id is not None else 0
                padded_ids = []
                attention_masks = []

                for ids in all_input_ids:
                    padding_length = target_length - len(ids)
                    padded_ids.append(ids + [pad_id] * padding_length)
                    attention_masks.append([1] * len(ids) + [0] * padding_length)

                all_input_ids = padded_ids

                # Pad offsets too
                if return_offsets_mapping:
                    padded_offsets = []
                    for offs in all_offsets:
                        padding_length = target_length - len(offs)
                        padded_offsets.append(offs + [(0, 0)] * padding_length)
                    all_offsets = padded_offsets

        # Create attention masks if requested but not created by padding
        if return_attention_mask and attention_masks is None:
            attention_masks = [[1] * len(ids) for ids in all_input_ids]

        return BatchEncoding(
            input_ids=all_input_ids,
            attention_mask=attention_masks,
            offset_mapping=all_offsets,
        )

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BPETokenizer(vocab_size={len(self.vocab)}, "
            f"merges={len(self.merges)}, "
            f"special_tokens={len(self.special_tokens)})"
        )


if __name__ == "__main__":
    # Test Byte-Level BPE tokenizer
    print("=" * 70)
    print("Testing Byte-Level BPE Tokenizer")
    print("=" * 70)

    # Create sample corpus with challenging Unicode
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, world! 你好世界 🌍",
        "Machine learning is transforming AI.",
        "Python 🐍 is widely used in data science.",
        "émojis and àccents are handled correctly.",
        "Special characters: @#$%^&*()_+-=[]{}|;':\",./<>?",
        "Emojis: 😀😃😄😁😆😅🤣😂",
        "Math: ∑∏∫√∞≈≠±×÷",
    ]

    # Train tokenizer
    print("\n1. Training byte-level tokenizer...")
    tokenizer = BPETokenizer(vocab_size=500, min_frequency=1, normalization="NFC")
    tokenizer.train(corpus, verbose=True)

    # Test encoding/decoding with various Unicode
    print("\n2. Testing encoding/decoding with Unicode...")
    print("=" * 70)

    test_cases = [
        "Hello, world!",
        "你好世界 🌍",
        "Python 🐍 rocks!",
        "émojis and àccents",
        "Math: ∑∏∫√",
        "Mixed: Hello你好🌍",
    ]

    for text in test_cases:
        print(f"\nOriginal: '{text}'")
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''} (count: {len(tokens)})")
        print(f"Decoded: '{decoded}'")
        print(f"✓ Match: {text == decoded}")

    # Test with special tokens
    print("\n3. Testing special token handling...")
    print("=" * 70)
    test_text = "Hello, world!"
    tokens_with_special = tokenizer.encode(test_text, add_special_tokens=True)
    tokens_without_special = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"Without special: {len(tokens_without_special)} tokens")
    print(f"With special: {len(tokens_with_special)} tokens")
    print(f"Special token added: {tokens_with_special[-1] == tokenizer.vocab['<|endoftext|>']}")

    # Test save/load
    print("\n4. Testing save/load...")
    print("=" * 70)

    save_dir = "test_byte_tokenizer"
    tokenizer.save(save_dir)

    loaded_tokenizer = BPETokenizer.load(save_dir)
    print(f"Loaded: {loaded_tokenizer}")

    # Verify loaded tokenizer works
    test_text = "Hello, world! 你好🌍"
    original_tokens = tokenizer.encode(test_text)
    loaded_tokens = loaded_tokenizer.encode(test_text)
    print(f"✓ Tokens match: {original_tokens == loaded_tokens}")

    # Test edge cases
    print("\n5. Testing edge cases...")
    print("=" * 70)
    edge_cases = [
        "",  # Empty string
        " ",  # Single space
        "\n\n\n",  # Multiple newlines
        "a" * 100,  # Long repetition
    ]

    for text in edge_cases:
        try:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            status = "✓" if text == decoded else "✗"
            print(f"{status} Edge case (len={len(text)}): encoded to {len(tokens)} tokens")
        except Exception as e:
            print(f"✗ Edge case failed: {e}")

    # Cleanup
    import shutil

    shutil.rmtree(save_dir)

    print("\n" + "=" * 70)
    print("✓ Byte-level BPE tokenizer implementation complete!")
    print("=" * 70)
