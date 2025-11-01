"""
Unit tests for BPE Tokenizer with byte-level encoding.

Tests cover:
- Basic tokenization and decoding
- Unicode handling (Chinese, emojis, accents)
- LRU cache behavior
- Error handling for missing tokens
- Performance optimizations
"""

import pytest
import time
from collections import OrderedDict

from src.tokenization.bpe_tokenizer import BPETokenizer


class TestBPETokenizer:
    """Test suite for byte-level BPE tokenizer."""

    def test_initialization(self):
        """Test tokenizer initialization with default parameters."""
        tokenizer = BPETokenizer(vocab_size=1000)

        assert tokenizer.vocab_size == 1000
        assert tokenizer.min_frequency == 2
        assert tokenizer.normalization == "NFC"
        assert tokenizer.special_tokens == ["<|endoftext|>"]
        assert isinstance(tokenizer.cache, OrderedDict)
        assert tokenizer.cache_maxsize == 50000

    def test_byte_encoder_decoder(self):
        """Test byte-to-unicode mapping is bijective."""
        tokenizer = BPETokenizer()

        # Check all 256 bytes are mapped
        assert len(tokenizer.byte_encoder) == 256
        assert len(tokenizer.byte_decoder) == 256

        # Check bijection
        for byte_val, unicode_char in tokenizer.byte_encoder.items():
            assert tokenizer.byte_decoder[unicode_char] == byte_val

    def test_basic_tokenization(self):
        """Test basic encode/decode functionality."""
        tokenizer = BPETokenizer(vocab_size=500)

        training_data = [
            "Hello, world!",
            "Python programming is fun!",
            "Machine learning with transformers."
        ]

        tokenizer.train(training_data, verbose=False)

        # Test encoding and decoding
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert decoded == text
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_unicode_handling(self):
        """Test tokenizer handles various Unicode correctly."""
        tokenizer = BPETokenizer(vocab_size=1000)

        training_data = [
            "ASCII text",
            "Chinese: 你好世界",
            "Emojis: 🐍 🌍 🚀",
            "Accents: café, naïve, émojis",
            "Math: ∑ ∏ ∫ √",
            "Mixed: Hello你好🌍"
        ]

        tokenizer.train(training_data, verbose=False)

        # Test each type
        test_cases = [
            "Hello, world!",
            "你好世界 🌍",
            "Python 🐍 rocks!",
            "émojis and àccents",
            "Math: ∑∏∫√",
            "Hello你好🌍"
        ]

        for text in test_cases:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Failed to round-trip: {text}"

    def test_lru_cache_behavior(self):
        """Test LRU cache eviction and hit/miss behavior."""
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(["test data for cache"], verbose=False)

        # Set small cache for testing
        tokenizer.cache_maxsize = 3
        tokenizer.cache.clear()

        # Add items to cache
        words = ["word1", "word2", "word3"]
        for word in words:
            tokenizer._tokenize_word(word)

        assert len(tokenizer.cache) == 3
        assert list(tokenizer.cache.keys()) == words

        # Access first word (should move to end)
        tokenizer._tokenize_word("word1")
        assert list(tokenizer.cache.keys()) == ["word2", "word3", "word1"]

        # Add new word (should evict word2)
        tokenizer._tokenize_word("word4")
        assert len(tokenizer.cache) == 3
        assert "word2" not in tokenizer.cache
        assert "word1" in tokenizer.cache
        assert "word4" in tokenizer.cache

    def test_cache_performance(self):
        """Test cache improves performance."""
        tokenizer = BPETokenizer(vocab_size=500)
        tokenizer.train(["test " * 100], verbose=False)

        # Clear cache
        tokenizer.cache.clear()

        # Time first tokenization
        start = time.time()
        tokens1 = tokenizer._tokenize_word("testing")
        time_no_cache = time.time() - start

        # Time cached tokenization
        start = time.time()
        tokens2 = tokenizer._tokenize_word("testing")
        time_cached = time.time() - start

        assert tokens1 == tokens2
        assert time_cached < time_no_cache

    def test_error_handling(self):
        """Test error handling for missing tokens."""
        tokenizer = BPETokenizer()

        # Set up tokenizer with minimal vocab
        tokenizer.vocab = {"<|endoftext|>": 0}
        tokenizer.byte_encoder = {i: chr(i) for i in range(256)}
        tokenizer.merges = {}
        tokenizer.merge_ranks = {}

        # Should raise ValueError for untrained tokenizer
        with pytest.raises(ValueError) as exc_info:
            tokenizer._tokenize_word("test")

        assert "not found in vocabulary" in str(exc_info.value)
        assert "byte-level BPE" in str(exc_info.value)

    def test_special_tokens(self):
        """Test special token handling."""
        special_tokens = ["<|endoftext|>", "<|startoftext|>", "<|pad|>"]
        tokenizer = BPETokenizer(vocab_size=500, special_tokens=special_tokens)

        tokenizer.train(["Some training data"], verbose=False)

        # Check special tokens are in vocab
        for token in special_tokens:
            assert token in tokenizer.vocab

        # Test encoding with special tokens
        text = "Hello world"
        tokens_without = tokenizer.encode(text, add_special_tokens=False)
        tokens_with = tokenizer.encode(text, add_special_tokens=True)

        # With special tokens should be longer (adds endoftext)
        assert len(tokens_with) == len(tokens_without) + 1
        # Current implementation adds endoftext at the end only
        assert tokens_with[-1] == tokenizer.vocab["<|endoftext|>"]
        assert tokens_with[:-1] == tokens_without

    def test_normalization(self):
        """Test Unicode normalization options."""
        test_text = "café"  # Can be represented differently in Unicode

        for norm in ["NFC", "NFD", "NFKC", "NFKD", None]:
            tokenizer = BPETokenizer(vocab_size=100, normalization=norm)
            tokenizer.train([test_text], verbose=False)

            tokens = tokenizer.encode(test_text)
            decoded = tokenizer.decode(tokens)

            # Should handle text correctly regardless of normalization
            assert len(tokens) > 0
            # NFD/NFKD decompose characters, so exact match depends on normalization
            # The important thing is that encode/decode round-trips correctly
            if norm in ["NFD", "NFKD"]:
                # Normalize both for comparison
                import unicodedata
                assert unicodedata.normalize("NFC", decoded) == unicodedata.normalize("NFC", test_text)
            else:
                assert decoded == test_text

    def test_list_unpacking_optimization(self):
        """Test that list unpacking optimization works correctly."""
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(["test data for unpacking"], verbose=False)

        # The internal merge operation uses list unpacking
        # We test it indirectly by ensuring tokenization works
        text = "test unpacking"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert decoded == text

    def test_save_and_load(self):
        """Test tokenizer serialization."""
        import tempfile
        import shutil

        tokenizer = BPETokenizer(vocab_size=500)
        training_data = ["Hello world", "Test data", "Save and load"]
        tokenizer.train(training_data, verbose=False)

        # Save tokenizer
        temp_dir = tempfile.mkdtemp()
        save_path = f"{temp_dir}/test_tokenizer"
        tokenizer.save(save_path)

        # Load tokenizer
        loaded_tokenizer = BPETokenizer.load(save_path)

        # Test loaded tokenizer works the same
        test_text = "Hello world test"
        tokens1 = tokenizer.encode(test_text)
        tokens2 = loaded_tokenizer.encode(test_text)

        assert tokens1 == tokens2
        assert tokenizer.vocab == loaded_tokenizer.vocab
        assert tokenizer.merges == loaded_tokenizer.merges

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_empty_input(self):
        """Test handling of empty and edge case inputs."""
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(["test"], verbose=False)

        # Empty string
        tokens = tokenizer.encode("")
        assert len(tokens) == 0
        assert tokenizer.decode(tokens) == ""

        # Whitespace only
        tokens = tokenizer.encode("   ")
        decoded = tokenizer.decode(tokens)
        assert decoded == "   "

    def test_long_text(self):
        """Test handling of long text."""
        tokenizer = BPETokenizer(vocab_size=1000)

        # Train on repeated patterns
        long_text = "The quick brown fox jumps over the lazy dog. " * 100
        tokenizer.train([long_text], verbose=False)

        # Test encoding/decoding long text
        tokens = tokenizer.encode(long_text)
        decoded = tokenizer.decode(tokens)

        assert decoded == long_text
        assert len(tokens) < len(long_text)  # Should compress

    @pytest.mark.parametrize("vocab_size", [100, 300, 500, 1000, 5000])
    def test_different_vocab_sizes(self, vocab_size):
        """Test tokenizer with different vocabulary sizes."""
        tokenizer = BPETokenizer(vocab_size=vocab_size)

        training_data = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning with neural networks.",
            "Python programming language."
        ] * 10

        tokenizer.train(training_data, verbose=False)

        # For byte-level BPE, minimum vocab is 256 (bytes) + special tokens
        # The tokenizer adjusts small vocab_size to this minimum
        expected_size = max(vocab_size, 256 + len(tokenizer.special_tokens))
        assert len(tokenizer.vocab) <= expected_size

        # Test round-trip
        test_text = "The quick brown fox"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        assert decoded == test_text

    def test_encode_plus_with_offsets(self):
        """Test encode_plus with offset mapping."""
        tokenizer = BPETokenizer(vocab_size=500)
        tokenizer.train(["Hello world test"], verbose=False)

        result = tokenizer.encode_plus(
            "Hello world",
            return_offsets_mapping=True
        )

        assert "input_ids" in result
        assert "offset_mapping" in result
        assert len(result["input_ids"]) == len(result["offset_mapping"])
        assert all(isinstance(offset, tuple) for offset in result["offset_mapping"])
        assert all(len(offset) == 2 for offset in result["offset_mapping"])

    def test_batch_encode_plus_no_padding(self):
        """Test batch encoding without padding."""
        tokenizer = BPETokenizer(vocab_size=500)
        tokenizer.train(["Hello world test"], verbose=False)

        texts = ["Hello", "Hello world"]
        result = tokenizer.batch_encode_plus(texts, padding=False)

        assert len(result.input_ids) == 2
        assert len(result.input_ids[0]) < len(result.input_ids[1])

    def test_batch_encode_plus_longest_padding(self):
        """Test batch encoding with longest padding."""
        tokenizer = BPETokenizer(vocab_size=500)
        tokenizer.train(["Hello world test data"], verbose=False)

        texts = ["Hi", "Hello world"]
        result = tokenizer.batch_encode_plus(
            texts,
            padding="longest",
            return_attention_mask=True
        )

        # All sequences should have same length
        assert len(result.input_ids[0]) == len(result.input_ids[1])
        assert len(result.attention_mask[0]) == len(result.attention_mask[1])

        # Shorter sequence should have padding
        assert sum(result.attention_mask[0]) < len(result.attention_mask[0])

    def test_batch_encode_plus_max_length_padding(self):
        """Test batch encoding with max_length padding."""
        tokenizer = BPETokenizer(vocab_size=500)
        tokenizer.train(["Hello world test"], verbose=False)

        texts = ["Hi", "Hello"]
        result = tokenizer.batch_encode_plus(
            texts,
            padding="max_length",
            max_length=10,
            return_attention_mask=True
        )

        # All sequences should be exactly max_length
        assert all(len(ids) == 10 for ids in result.input_ids)
        assert all(len(mask) == 10 for mask in result.attention_mask)

    def test_batch_encode_plus_truncation(self):
        """Test batch encoding with truncation."""
        tokenizer = BPETokenizer(vocab_size=500)
        tokenizer.train(["Hello world test data"], verbose=False)

        texts = ["Hello world test data extra"]
        result = tokenizer.batch_encode_plus(
            texts,
            truncation=True,
            max_length=5
        )

        # Should be truncated to max_length
        assert all(len(ids) <= 5 for ids in result.input_ids)

    def test_batch_encode_plus_with_offsets(self):
        """Test batch encoding with offset mapping."""
        tokenizer = BPETokenizer(vocab_size=500)
        tokenizer.train(["Hello world"], verbose=False)

        texts = ["Hello", "World"]
        result = tokenizer.batch_encode_plus(
            texts,
            padding="longest",
            return_offsets_mapping=True
        )

        assert result.offset_mapping is not None
        assert len(result.offset_mapping) == 2
        # Padded positions should have (0, 0) offsets
        assert all(
            offset == (0, 0) or (offset[0] < offset[1])
            for offsets in result.offset_mapping
            for offset in offsets
        )

    def test_pad_token_id(self):
        """Test pad_token_id property."""
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(["test"], verbose=False)

        assert tokenizer.pad_token_id is not None
        assert tokenizer.pad_token_id == tokenizer.vocab["<|endoftext|>"]

    def test_eos_token_id(self):
        """Test eos_token_id property."""
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(["test"], verbose=False)

        assert tokenizer.eos_token_id is not None
        assert tokenizer.eos_token_id == tokenizer.vocab["<|endoftext|>"]

    def test_retrain_clears_state(self):
        """Test that calling train() multiple times clears previous state."""
        tokenizer = BPETokenizer(vocab_size=300)

        # First training
        tokenizer.train(["Hello world test"], verbose=False)
        first_vocab_size = len(tokenizer.vocab)
        first_merges = len(tokenizer.merges)
        first_vocab_copy = tokenizer.vocab.copy()

        # Second training on different data
        tokenizer.train(["Python programming language"], verbose=False)
        second_vocab_size = len(tokenizer.vocab)
        second_merges = len(tokenizer.merges)

        # Vocab size should be similar (both aiming for 300)
        assert abs(first_vocab_size - second_vocab_size) < 10

        # But the actual vocab should be different
        assert tokenizer.vocab != first_vocab_copy

        # Cache should be cleared
        assert len(tokenizer.cache) == 0

        # Should be able to encode new data without errors
        text = "Python programming"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text
