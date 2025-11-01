# Getting Started

Get up and running with MiniMacLLM - the brain-inspired continual learning language model!

## Installation

```bash
# Clone the repository
git clone https://github.com/DrDrewCain/MiniMacLLM.git
cd MiniMacLLM

# Install dependencies
pip install -r requirements.txt

# Or install with development dependencies
pip install -r requirements.txt -r requirements-dev.txt
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
# Run tests
pytest tests/ -v

# Expected: 165+ tests passing ✅
```

### 2. Download Training Data

**From HuggingFace:**
```bash
# Install datasets library
pip install datasets

# Download math dataset (tested, ~2MB)
python scripts/download_training_data.py --domains math

# Or download all domains (~2-3GB)
python scripts/download_training_data.py --domains all
```

**From External Sources:**
```bash
# Install requests
pip install requests

# Download classic literature from Project Gutenberg
python scripts/download_external_data.py --sources gutenberg --max-books 100

# Get instructions for Wikipedia, arXiv, etc.
python scripts/download_external_data.py --sources wikipedia arxiv
```

See [DATA_GUIDE.md](../DATA_GUIDE.md) for comprehensive data download instructions.

### 3. Train a Tokenizer

```bash
# Using existing WikiText data
python scripts/train_tokenizer.py \
  --data data/raw/wikitext2_train.txt \
  --vocab_size 8000 \
  --save data/tokenizers/my_tokenizer

# Or use downloaded data
python scripts/train_tokenizer.py \
  --data data/training/math/gsm8k.txt \
  --vocab_size 8000 \
  --save data/tokenizers/math_tokenizer
```

### 4. Pre-train a Model

```bash
python scripts/pretrain.py \
  --config configs/medium.yaml \
  --data data/raw/wikitext2_train.txt \
  --tokenizer data/tokenizers/my_tokenizer \
  --epochs 3 \
  --batch_size 4 \
  --save_dir checkpoints/base_model
```

**Brain-Inspired Features Active:**
- ✅ Autonomous learning rate (system decides speed)
- ✅ Neuromodulation (dopamine/serotonin/ACh)
- ✅ Homeostatic plasticity (prevents dead neurons)

### 5. Test Continual Learning

```bash
# Learn a new domain (e.g., code)
python scripts/continual_learn.py \
  --model checkpoints/base_model/final/model.pt \
  --tokenizer data/tokenizers/my_tokenizer \
  --data data/training/code/python-code.txt \
  --domain code \
  --epochs 2
```

**Anti-Forgetting Mechanisms Active:**
- ✅ LoRA (only 2.2% params updated)
- ✅ Experience Replay (intelligent rehearsal)
- ✅ EWC (elastic weight protection)
- ✅ Sleep Consolidation (offline strengthening)
- ✅ Hippocampal Memory (pattern separation)

### 6. Generate Text

```bash
python scripts/generate.py \
  --model checkpoints/code/model.pt \
  --tokenizer data/tokenizers/my_tokenizer \
  --prompt "def fibonacci(n):" \
  --max_tokens 100
```

## Python API

### Basic Usage with Brain-Inspired Features

```python
from src.model.llm import ContinualLLM, ModelConfig
from src.tokenization.bpe_tokenizer import BPETokenizer
from src.continual.continual_trainer import ContinualLearner, ContinualLearningConfig

# Load tokenizer
tokenizer = BPETokenizer.load("data/tokenizers/my_tokenizer")

# Create model
config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=768,
    num_layers=12,
    num_query_heads=12,
    num_kv_heads=3
)
model = ContinualLLM(config)

# Create continual learner with brain-inspired features
learner = ContinualLearner(
    base_model=model,
    config=ContinualLearningConfig(
        lora_r=8,
        replay_buffer_size=10000,
        use_ewc=True,
        device="mps",  # or "cpu" or "cuda"

        # Brain-inspired mechanisms (enabled by default)
        use_neuromodulation=True,      # Dopamine/serotonin/ACh
        use_homeostasis=True,           # Prevents dead neurons
        use_autonomous_lr=True,         # System decides learning rate
        autonomous_lr_sensitivity=1.0   # Base sensitivity
    ),
    tokenizer=tokenizer
)

# Learn from text (system decides its own learning speed!)
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

### Advanced: Accessing Brain Mechanisms

```python
# Check autonomous learning dynamics
if learner.autonomous_lr is not None:
    dynamics = learner.autonomous_lr.get_learning_dynamics()
    print(f"Current learning state:")
    print(f"  Success rate: {dynamics['success_rate']:.2%}")
    print(f"  Metabolic cost: {dynamics['metabolic_cost']:.4f}")
    print(f"  Gradient variance: {dynamics['gradient_variance']:.4f}")

# Trigger sleep consolidation manually
if hasattr(learner, 'sleep_consolidation'):
    result = learner.sleep_consolidation.consolidate(verbose=True)
    print(f"Sleep: {result['connections_strengthened']} strengthened")
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
