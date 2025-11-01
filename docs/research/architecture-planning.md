# Modern LLM Architecture Design

## Overview
This document details the architectural design for upgrading our LLM to include state-of-the-art 2024-2025 techniques while maintaining modularity and extensibility for NLP tasks.

---

## Current Architecture Analysis

### What We Have (Model.py)
✅ Basic transformer decoder architecture
✅ Multi-Head Attention (MHA)
✅ Feed-forward networks with GELU
✅ Layer normalization (post-norm style)
✅ Absolute positional embeddings
✅ Character-level tokenizer
✅ Next-token prediction training
✅ Basic text generation

### What Needs Upgrading
❌ MHA → **GQA** (Grouped Query Attention)
❌ Absolute positions → **RoPE** (Rotary Position Embeddings)
❌ GELU → **SwiGLU** activation
❌ LayerNorm → **RMSNorm**
❌ Post-norm → **Pre-norm** architecture
❌ Character tokenizer → **BPE/SentencePiece**
❌ No efficient training features (mixed precision, gradient checkpointing)
❌ No fine-tuning support (LoRA, QLoRA)
❌ No evaluation framework
❌ No data pipeline

---

## Proposed Architecture

### File Structure
```text
Custom_ML_Agent/
├── Model.py                    # Legacy basic model (keep for reference)
├── REQUIREMENTS.md            # Complete requirements doc ✅
├── ARCHITECTURE_DESIGN.md     # This file ✅
├── requirements.txt           # Python dependencies
│
├── src/
│   ├── __init__.py
│   │
│   ├── tokenization/
│   │   ├── __init__.py
│   │   ├── bpe_tokenizer.py       # BPE implementation
│   │   ├── vocab_trainer.py       # Vocabulary training
│   │   └── tokenizer_utils.py     # Helper functions
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── attention.py           # GQA, MHA, MLA implementations
│   │   ├── embeddings.py          # RoPE, learned positional
│   │   ├── feedforward.py         # SwiGLU and GLU variants
│   │   ├── normalization.py       # RMSNorm, LayerNorm
│   │   ├── transformer_block.py   # Modern transformer layer
│   │   ├── llm.py                 # Main LLM model
│   │   └── moe.py                 # Optional: Mixture of Experts
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Main training loop
│   │   ├── optimization.py        # Optimizers, schedulers
│   │   ├── mixed_precision.py     # AMP utilities
│   │   ├── checkpoint.py          # Model checkpointing
│   │   └── distributed.py         # DDP, FSDP support
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # Dataset classes
│   │   ├── preprocessing.py       # Data cleaning, filtering
│   │   ├── dataloader.py          # Efficient data loading
│   │   └── synthetic.py           # Synthetic data generation
│   │
│   ├── finetuning/
│   │   ├── __init__.py
│   │   ├── lora.py                # LoRA implementation
│   │   ├── qlora.py               # QLoRA with quantization
│   │   ├── dora.py                # DoRA variant
│   │   └── finetuner.py           # Fine-tuning orchestration
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── generation.py          # Generation strategies
│   │   ├── sampling.py            # Top-k, top-p, etc.
│   │   ├── kv_cache.py            # KV caching
│   │   └── quantization.py        # INT8, INT4 quantization
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Perplexity, BLEU, ROUGE
│   │   ├── benchmarks.py          # MMLU, HumanEval, etc.
│   │   └── evaluator.py           # Evaluation orchestration
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py             # Logging setup
│       ├── config.py              # Configuration management
│       └── helpers.py             # Misc utilities
│
├── configs/
│   ├── model_small.yaml           # 100M-500M config
│   ├── model_medium.yaml          # 1B-7B config
│   └── model_large.yaml           # 13B-70B config
│
├── scripts/
│   ├── train.py                   # Training script
│   ├── finetune.py               # Fine-tuning script
│   ├── evaluate.py               # Evaluation script
│   ├── generate.py               # Text generation script
│   └── train_tokenizer.py        # Tokenizer training script
│
├── notebooks/
│   ├── 01_tokenizer_demo.ipynb
│   ├── 02_model_architecture.ipynb
│   └── 03_training_demo.ipynb
│
├── tests/
│   ├── test_attention.py
│   ├── test_embeddings.py
│   ├── test_tokenizer.py
│   └── test_training.py
│
└── data/
    ├── raw/                       # Raw text data
    ├── processed/                 # Preprocessed data
    └── tokenizers/                # Trained tokenizers
```text
---

## Core Components Design

### 1. Grouped Query Attention (GQA)

```python
class GroupedQueryAttention(nn.Module):
    """
    GQA reduces KV cache by sharing key/value heads across query heads.

    Example: 32 query heads, 8 KV heads → 4 query heads share each KV head
    """
    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,  # New parameter!
        dropout: float = 0.1
    ):
        super().__init__()
        assert num_query_heads % num_kv_heads == 0

        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.d_k = d_model // num_query_heads

        # Q has all heads, K/V have fewer heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin, mask=None, use_cache=False):
        # 1. Project to Q, K, V
        # 2. Apply RoPE to Q, K
        # 3. Expand K, V to match Q heads (repeat each KV head)
        # 4. Compute attention
        # 5. Return output + cache
        pass
```text
**Key Benefits:**
- 4-8x reduction in KV cache size
- Minimal quality degradation vs MHA
- Modern best practice

---

### 2. Rotary Position Embeddings (RoPE)

```python
class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE applies rotation to Q and K based on position.
    No learnable parameters needed!
    """
    def __init__(self, dim: int, max_seq_len: int = 32768, base: float = 10000.0):
        super().__init__()
        # Precompute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin for all positions
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, x, seq_len):
        # Return cos/sin for current sequence length
        return self.freqs_cos[:seq_len], self.freqs_sin[:seq_len]

def apply_rotary_emb(x, freqs_cos, freqs_sin):
    """Apply rotation to input tensor"""
    # Reshape x to separate even/odd dimensions
    # Apply rotation: x_out = x * cos + rotate_half(x) * sin
    pass
```

**Key Benefits:**
- No learned parameters
- Better length extrapolation
- Encodes relative positions naturally
- Universal in modern LLMs

---

### 3. SwiGLU Activation

```python
class SwiGLU(nn.Module):
    """
    SwiGLU: Swish-Gated Linear Unit
    Modern best practice

    Formula: SwiGLU(x, W, V) = Swish(xW) ⊗ (xV)
    where Swish(x) = x * sigmoid(x)
    """
    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        # SwiGLU needs 3 projections instead of 2
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)  # Gate
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)  # Down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=bias)  # Up projection

    def forward(self, x):
        # SwiGLU formula
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

**Key Benefits:**
- Better than GELU, ReLU empirically
- Used in all SOTA models
- Slightly more parameters but worth it

---

### 4. RMSNorm

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Simpler and faster than LayerNorm
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
```

**Key Benefits:**
- Faster than LayerNorm (no mean subtraction)
- Same performance
- Modern best practice

---

### 5. Modern Transformer Block (Pre-Norm)

```python
class ModernTransformerBlock(nn.Module):
    """
    Modern transformer block with:
    - Pre-normalization
    - GQA attention
    - SwiGLU feedforward
    - RMSNorm
    """
    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = GroupedQueryAttention(d_model, num_query_heads, num_kv_heads, dropout)
        self.feed_forward = SwiGLU(d_model, d_ff)

        # Pre-norm: normalize BEFORE attention/FFN
        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

    def forward(self, x, freqs_cos, freqs_sin, mask=None):
        # Pre-norm attention with residual
        x = x + self.attention(
            self.attention_norm(x),
            freqs_cos,
            freqs_sin,
            mask
        )

        # Pre-norm FFN with residual
        x = x + self.feed_forward(self.ffn_norm(x))

        return x
```

**Key Benefits:**
- Pre-norm: better gradient flow, more stable training
- Residual connections: enables deep networks
- Modern components: GQA + SwiGLU + RMSNorm

---

## Model Configuration Examples

### Small Model (Research/Dev)
```yaml
model_name: "custom-llm-small"
architecture: "decoder-only-transformer"

# Architecture
d_model: 768
num_layers: 12
num_query_heads: 12
num_kv_heads: 4          # GQA: 3 queries per KV head
d_ff: 3072               # 4x d_model for SwiGLU
max_seq_len: 2048

# Embeddings
vocab_size: 32000
rope_theta: 10000.0      # RoPE base frequency

# Regularization
dropout: 0.1
attention_dropout: 0.1

# Norm
norm_eps: 1e-6
use_rms_norm: true

# Estimated parameters: ~350M
```

### Medium Model (Production)
```yaml
model_name: "custom-llm-7b"
architecture: "decoder-only-transformer"

# Architecture
d_model: 4096
num_layers: 32
num_query_heads: 32
num_kv_heads: 8          # GQA: 4 queries per KV head
d_ff: 14336              # ~3.5x d_model for SwiGLU
max_seq_len: 8192

# Embeddings
vocab_size: 50000
rope_theta: 10000.0
extended_rope: true      # For longer context

# Regularization
dropout: 0.0             # No dropout for large models
attention_dropout: 0.0

# Norm
norm_eps: 1e-5
use_rms_norm: true

# Estimated parameters: ~7B
```

---

## Training Configuration

### Efficient Training Setup
```yaml
training:
  # Optimization
  optimizer: "adamw"
  learning_rate: 3.0e-4
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  eps: 1.0e-8

  # Schedule
  warmup_steps: 2000
  lr_scheduler: "cosine"
  min_lr_ratio: 0.1

  # Gradient
  max_grad_norm: 1.0
  gradient_accumulation_steps: 8

  # Mixed Precision
  mixed_precision: "bf16"    # Use bfloat16 if available

  # Efficiency
  gradient_checkpointing: true
  flash_attention: true

  # Batch sizes
  per_device_batch_size: 4
  total_batch_size: 512      # Global batch size

  # Checkpointing
  save_steps: 1000
  eval_steps: 500
  logging_steps: 10
```

---

## Fine-Tuning with LoRA

### LoRA Configuration
```yaml
lora:
  # Which modules to apply LoRA
  target_modules:
    - "q_proj"     # Query projection
    - "k_proj"     # Key projection
    - "v_proj"     # Value projection
    - "o_proj"     # Output projection
    - "gate_proj"  # SwiGLU gate
    - "up_proj"    # SwiGLU up
    - "down_proj"  # SwiGLU down

  # LoRA hyperparameters
  r: 16              # Rank (8-64 typical)
  alpha: 32          # Scaling (usually 2*r)
  dropout: 0.05

  # Training
  learning_rate: 1.0e-4
  batch_size: 16
  epochs: 3

  # QLoRA specific (if using quantization)
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_quant_type: "nf4"
```

---

## Data Pipeline

### Preprocessing Pipeline
```python
class DataPreprocessor:
    """
    Multi-stage preprocessing pipeline:
    1. Text cleaning
    2. Language filtering
    3. Quality filtering
    4. Deduplication
    5. Tokenization
    """

    def __init__(self, config):
        self.tokenizer = load_tokenizer(config.tokenizer_path)
        self.quality_model = load_quality_classifier()

    def process(self, raw_text):
        # 1. Clean
        text = self.clean_text(raw_text)

        # 2. Filter by language
        if not self.is_target_language(text):
            return None

        # 3. Quality score
        if self.quality_score(text) < 0.5:
            return None

        # 4. Tokenize
        tokens = self.tokenizer.encode(text)

        # 5. Check length
        if not (self.min_length <= len(tokens) <= self.max_length):
            return None

        return tokens
```

---

## Evaluation Framework

### Benchmark Suite
```python
class BenchmarkEvaluator:
    """
    Comprehensive evaluation across multiple benchmarks
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # Load benchmarks
        self.benchmarks = {
            'mmlu': MMLUBenchmark(),
            'hellaswag': HellaSwagBenchmark(),
            'humaneval': HumanEvalBenchmark(),
            'truthfulqa': TruthfulQABenchmark(),
        }

    def evaluate_all(self):
        results = {}
        for name, benchmark in self.benchmarks.items():
            print(f"Evaluating {name}...")
            score = benchmark.evaluate(self.model, self.tokenizer)
            results[name] = score
        return results
```

---

## Implementation Roadmap

### Phase 1: Core Architecture Upgrade (Priority 1)
**Goal:** Modernize the transformer architecture

**Tasks:**
1. Implement RMSNorm → `src/model/normalization.py`
2. Implement RoPE → `src/model/embeddings.py`
3. Implement GQA → `src/model/attention.py`
4. Implement SwiGLU → `src/model/feedforward.py`
5. Assemble ModernTransformerBlock → `src/model/transformer_block.py`
6. Update main model → `src/model/llm.py`
7. Write tests for all components

**Estimated Time:** 3-5 days

---

### Phase 2: Tokenization (Priority 1)
**Goal:** Replace character tokenizer with BPE

**Tasks:**
1. Implement BPE algorithm → `src/tokenization/bpe_tokenizer.py`
2. Add vocabulary training → `src/tokenization/vocab_trainer.py`
3. Integrate SentencePiece → wrapper in tokenization module
4. Create tokenizer training script → `scripts/train_tokenizer.py`
5. Add special tokens handling
6. Test on sample data

**Estimated Time:** 2-3 days

---

### Phase 3: Training Infrastructure (Priority 1)
**Goal:** Efficient, production-ready training

**Tasks:**
1. Implement mixed precision training → `src/training/mixed_precision.py`
2. Add gradient checkpointing → `src/training/trainer.py`
3. Create optimized dataloader → `src/data/dataloader.py`
4. Implement learning rate scheduling → `src/training/optimization.py`
5. Add checkpointing logic → `src/training/checkpoint.py`
6. Add logging and monitoring (Weights & Biases)
7. Create training script → `scripts/train.py`

**Estimated Time:** 3-4 days

---

### Phase 4: Data Pipeline (Priority 2)
**Goal:** High-quality data processing

**Tasks:**
1. Implement data cleaning → `src/data/preprocessing.py`
2. Add quality filtering
3. Add deduplication logic
4. Create dataset classes → `src/data/dataset.py`
5. Optional: synthetic data generation → `src/data/synthetic.py`

**Estimated Time:** 2-3 days

---

### Phase 5: Fine-Tuning Support (Priority 2)
**Goal:** Enable efficient fine-tuning

**Tasks:**
1. Implement LoRA → `src/finetuning/lora.py`
2. Add 4-bit quantization → `src/finetuning/qlora.py`
3. Create fine-tuning trainer → `src/finetuning/finetuner.py`
4. Add adapter merging utilities
5. Create fine-tuning script → `scripts/finetune.py`

**Estimated Time:** 3-4 days

---

### Phase 6: Inference Optimization (Priority 2)
**Goal:** Fast, efficient generation

**Tasks:**
1. Implement KV caching → `src/inference/kv_cache.py`
2. Add sampling strategies → `src/inference/sampling.py`
3. Implement generation methods → `src/inference/generation.py`
4. Optional: INT8 quantization → `src/inference/quantization.py`
5. Create generation script → `scripts/generate.py`

**Estimated Time:** 2-3 days

---

### Phase 7: Evaluation (Priority 2)
**Goal:** Comprehensive model evaluation

**Tasks:**
1. Implement core metrics → `src/evaluation/metrics.py`
2. Integrate MMLU benchmark → `src/evaluation/benchmarks.py`
3. Add HumanEval, HellaSwag, others
4. Create evaluation orchestrator → `src/evaluation/evaluator.py`
5. Create evaluation script → `scripts/evaluate.py`

**Estimated Time:** 3-4 days

---

### Phase 8: Advanced Features (Priority 3)
**Goal:** State-of-the-art capabilities

**Tasks:**
1. Optional: Implement MoE architecture → `src/model/moe.py`
2. Optional: FlashAttention integration
3. Optional: Long context support (128K tokens)
4. Optional: DPO alignment training
5. Distributed training support → `src/training/distributed.py`

**Estimated Time:** 5-7 days (if implemented)

---

## Testing Strategy

### Unit Tests
```python
# tests/test_attention.py
def test_gqa_forward():
    """Test GQA produces correct output shape"""

def test_gqa_cache():
    """Test KV caching works correctly"""

def test_rope_rotation():
    """Test RoPE applies rotations correctly"""
```

### Integration Tests
```python
# tests/test_model.py
def test_full_forward_pass():
    """Test complete model forward pass"""

def test_training_step():
    """Test training step without errors"""

def test_generation():
    """Test text generation works"""
```

### Performance Tests
```python
# tests/test_performance.py
def test_training_throughput():
    """Measure tokens/second during training"""

def test_inference_latency():
    """Measure ms/token during inference"""

def test_memory_usage():
    """Verify memory usage is within bounds"""
```

---

## Success Metrics

### Code Quality
- [ ] All components have unit tests (>80% coverage)
- [ ] Code follows PEP 8 style guide
- [ ] Docstrings for all classes and functions
- [ ] Type hints throughout codebase

### Performance
- [ ] Training: >50K tokens/sec on single A100
- [ ] Inference: <50ms/token for 7B model
- [ ] Memory: 7B model inference fits in 16GB

### Model Quality
- [ ] Perplexity < 15 on validation set
- [ ] MMLU score competitive with baseline
- [ ] Coherent text generation
- [ ] Stable training (no divergence)

---

## Next Steps

**Immediate Actions:**
1. ✅ Review REQUIREMENTS.md
2. ✅ Review ARCHITECTURE_DESIGN.md (this document)
3. 🔄 Create project structure
4. 🔄 Set up development environment
5. 🔄 Create requirements.txt
6. 🔄 Begin Phase 1 implementation

**Questions to Address:**
1. Which model size to start with? (Recommend: Small 350M for development)
2. Do we have training data ready?
3. What hardware is available? (CPU/GPU/TPU)
4. Should we implement MoE or stick to dense models initially?
5. Priority: Speed to first working model vs. completeness?

---

**Document Version:** 1.0
**Last Updated:** October 2025
**Status:** Design Complete - Ready for Implementation
