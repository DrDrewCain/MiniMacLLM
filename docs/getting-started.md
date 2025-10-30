# Getting Started

Get up and running with Continual LLM in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/continual-llm.git
cd continual-llm

# Install dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- CPU (works on any CPU)

**Recommended:**
- Python 3.10+
- 16GB+ RAM
- Apple Silicon (M1/M2/M3) or NVIDIA GPU

## Quick Start

### 1. Verify Installation

```bash
# Run smoke tests (~1 second)
pytest tests/test_smoke.py -v

# Expected: 22 passed ✅
```

### 2. Train a Tokenizer

```bash
python scripts/train_tokenizer.py \
  --data data/raw/your_data.txt \
  --vocab_size 8000 \
  --save data/tokenizers/my_tokenizer
```

### 3. Pre-train a Model

```bash
python scripts/pretrain.py \
  --config configs/small.yaml \
  --data data/raw/your_data.txt \
  --tokenizer data/tokenizers/my_tokenizer \
  --epochs 5 \
  --batch_size 4 \
  --save_dir checkpoints/my_model
```

### 4. Test Continual Learning

```bash
python scripts/continual_learn.py \
  --model checkpoints/my_model/final/model.pt \
  --tokenizer data/tokenizers/my_tokenizer \
  --data data/new_domain.txt \
  --domain specialized \
  --epochs 3
```

### 5. Generate Text

```bash
python scripts/generate.py \
  --model checkpoints/my_model/final/model.pt \
  --tokenizer data/tokenizers/my_tokenizer \
  --prompt "Your prompt here" \
  --max_tokens 100
```

## Python API

```python
from src.model.llm import ContinualLLM, ModelConfig
from src.tokenization.bpe_tokenizer import BPETokenizer
from src.continual.continual_trainer import ContinualLearner, ContinualLearningConfig

# Load tokenizer
tokenizer = BPETokenizer.load("data/tokenizers/my_tokenizer")

# Create model
config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    num_layers=6,
    num_query_heads=8,
    num_kv_heads=2
)
model = ContinualLLM(config)

# Create continual learner
learner = ContinualLearner(
    base_model=model,
    config=ContinualLearningConfig(
        lora_r=8,
        replay_buffer_size=1000,
        use_ewc=True,
        device="mps"  # or "cpu" or "cuda"
    ),
    tokenizer=tokenizer
)

# Learn from text
learner.learn_from_text(
    "Your training text here...",
    domain="your_domain",
    importance=1.0
)

# Generate
response = learner.generate("Your prompt", max_length=50)
print(response)

# Save
learner.save_checkpoint("checkpoints/my_checkpoint")
```

## Troubleshooting

### Out of Memory

**Solution:** Reduce batch size or sequence length
```bash
python scripts/pretrain.py \
  --batch_size 2 \
  --max_seq_len 512 \
  ...
```

### Import Errors

**Solution:** Install in development mode
```bash
pip install -e .
```

### MPS (Apple Silicon) Errors

**Solution:** Remove memory limit
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

## Next Steps

- **[User Guide](user-guide.md)** - Detailed training and usage
- **[Examples](../examples/)** - Code examples and tutorials
- **[Architecture](architecture.md)** - System design

---

Need help? Open an [issue](../../issues) or check [discussions](../../discussions).
