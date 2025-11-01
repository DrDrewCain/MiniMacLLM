# User Guide

Complete guide to training, continual learning, and using your language model.

## Table of Contents
1. [Training a Tokenizer](#training-a-tokenizer)
2. [Pre-training a Model](#pre-training-a-model)
3. [Continual Learning](#continual-learning)
4. [Text Generation](#text-generation)
5. [Checkpointing](#checkpointing)
6. [Python API](#python-api)

---

## Training a Tokenizer

### Basic Usage

```bash
python scripts/train_tokenizer.py \
  --data data/raw/corpus.txt \
  --vocab_size 8000 \
  --save data/tokenizers/my_tokenizer
```

### Options

- `--data`: Path to training data (required)
- `--vocab_size`: Target vocabulary size (default: 8000)
- `--min_frequency`: Minimum token frequency (default: 2)
- `--save`: Where to save tokenizer (required)

### Best Practices

- **Vocab size**: 8K for testing, 32K for production
- **Data size**: At least 10MB of text
- **Domain coverage**: Include diverse examples from target domains

---

## Pre-training a Model

### Basic Usage

```bash
python scripts/pretrain.py \
  --config configs/medium.yaml \
  --data data/raw/training_data.txt \
  --tokenizer data/tokenizers/my_tokenizer \
  --epochs 5 \
  --batch_size 4 \
  --save_dir checkpoints/my_model
```

### Configuration Files

Choose from pre-configured model sizes:

```bash
configs/tiny.yaml      # 10M params (testing)
configs/small.yaml     # 50M params (development)
configs/medium.yaml    # 200M params (production)
configs/large.yaml     # 500M params (high-quality)
```

### Training Options

**Required:**
- `--config`: Model configuration file
- `--data`: Training data path
- `--tokenizer`: Tokenizer directory

**Optional:**
- `--epochs`: Number of epochs (default: 10)
- `--batch_size`: Batch size (default: 8)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--max_seq_len`: Max sequence length (default: from config)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 1)
- `--save_dir`: Checkpoint directory
- `--device`: Device to use (default: auto-detect)

### Memory Optimization

If you run out of memory:

```bash
# Reduce batch size
--batch_size 2

# Reduce sequence length
--max_seq_len 512

# Use gradient accumulation
--batch_size 2 --gradient_accumulation_steps 4

# Remove MPS memory limit (Apple Silicon)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

---

## Continual Learning

### Basic Usage

```bash
python scripts/continual_learn.py \
  --model checkpoints/my_model/final/model.pt \
  --tokenizer data/tokenizers/my_tokenizer \
  --data data/new_domain.txt \
  --domain programming \
  --epochs 3 \
  --adapter_name python_expert
```

### Options

**Required:**
- `--model`: Path to pre-trained model
- `--tokenizer`: Tokenizer directory
- `--data`: New domain data

**Optional:**
- `--domain`: Domain name (default: "general")
- `--adapter_name`: Name for LoRA adapter (default: domain name)
- `--epochs`: Training epochs (default: 3)
- `--lora_r`: LoRA rank (default: 8)
- `--replay_buffer_size`: Buffer size (default: 1000)
- `--use_ewc`: Enable EWC (default: True)
- `--ewc_lambda`: EWC penalty strength (default: 1000)

### Multi-Domain Learning

Learn multiple domains sequentially:

```bash
# Learn Python
python scripts/continual_learn.py \
  --model checkpoints/base/model.pt \
  --data data/python.txt \
  --domain programming \
  --adapter_name python

# Learn Math (without forgetting Python!)
python scripts/continual_learn.py \
  --model checkpoints/base/model.pt \
  --data data/math.txt \
  --domain mathematics \
  --adapter_name math
```

---

## Text Generation

### Basic Usage

```bash
python scripts/generate.py \
  --model checkpoints/my_model/final/model.pt \
  --tokenizer data/tokenizers/my_tokenizer \
  --prompt "The quick brown fox" \
  --max_tokens 100
```

### Options

- `--model`: Model checkpoint path
- `--tokenizer`: Tokenizer directory
- `--prompt`: Input prompt
- `--max_tokens`: Maximum tokens to generate (default: 50)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top_p`: Nucleus sampling threshold (default: 0.9)
- `--top_k`: Top-k sampling (default: 50)

### Interactive Mode

```bash
python scripts/generate.py \
  --model checkpoints/my_model/final/model.pt \
  --tokenizer data/tokenizers/my_tokenizer \
  --interactive
```

---

## Checkpointing

### Automatic Checkpointing

During training, checkpoints are saved:
- After each epoch: `checkpoints/model_name/epoch_N/`
- Final model: `checkpoints/model_name/final/`

### Manual Save/Load

```python
from src.continual.continual_trainer import ContinualLearner

# Save
learner.save_checkpoint("checkpoints/my_checkpoint")

# Load
learner.load_checkpoint("checkpoints/my_checkpoint")
```

### Checkpoint Structure

```
checkpoints/my_model/
├── epoch_1/
│   ├── model.pt
│   ├── adapter.pt
│   ├── config.json
│   ├── replay_buffer.pt
│   └── ewc.pt
├── epoch_2/
│   └── ...
└── final/
    └── ...
```

---

## Python API

### Complete Example

```python
from src.model.llm import ContinualLLM, ModelConfig
from src.tokenization.bpe_tokenizer import BPETokenizer
from src.continual.continual_trainer import ContinualLearner, ContinualLearningConfig

# 1. Load tokenizer
tokenizer = BPETokenizer.load("data/tokenizers/my_tokenizer")

# 2. Create model
config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=768,
    num_layers=12,
    num_query_heads=12,
    num_kv_heads=3
)
model = ContinualLLM(config)

# 3. Create continual learner
learner = ContinualLearner(
    base_model=model,
    config=ContinualLearningConfig(
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        replay_buffer_size=1000,
        use_ewc=True,
        ewc_lambda=1000,
        device="mps"
    ),
    tokenizer=tokenizer
)

# 4. Learn from text
learner.learn_from_text(
    text="Python is a high-level programming language...",
    domain="programming",
    importance=1.0,
    num_epochs=3
)

# 5. Generate
response = learner.generate(
    prompt="Python is used for",
    max_length=100,
    temperature=0.8,
    top_p=0.9
)
print(response)

# 6. Save
learner.save_checkpoint("checkpoints/python_model")
```

### Loading Existing Models

```python
# Load from checkpoint
learner = ContinualLearner.from_checkpoint(
    "checkpoints/my_model",
    tokenizer=tokenizer
)

# Use immediately
response = learner.generate("Your prompt here")
```

---

## Best Practices

### For Training
1. Start with small model and dataset to verify pipeline
2. Use gradient accumulation for larger effective batch sizes
3. Monitor loss - should decrease steadily
4. Save checkpoints frequently

### For Continual Learning
1. Use importance weighting for critical examples
2. Keep replay buffer size proportional to data diversity
3. Adjust EWC lambda based on forgetting tolerance
4. Test on old domains to verify no forgetting

### For Generation
1. Temperature 0.7-0.9 for creative text
2. Temperature 0.1-0.3 for factual/deterministic output
3. Use top_p (nucleus sampling) for quality
4. Set max_tokens appropriately for task

---

## Troubleshooting

See [Getting Started - Troubleshooting](getting-started.md#troubleshooting) for common issues.

---

**Next:** [Evaluation Guide](evaluation.md) - Test and benchmark your models
