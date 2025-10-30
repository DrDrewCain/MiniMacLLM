# API Reference

Complete API documentation for Continual LLM.

## Core Modules

### `src.model.llm`

#### `ContinualLLM`

Main language model class.

```python
from src.model.llm import ContinualLLM, ModelConfig

config = ModelConfig(
    vocab_size=8000,
    d_model=512,
    num_layers=6,
    num_query_heads=8,
    num_kv_heads=2
)
model = ContinualLLM(config)
```

**Methods:**
- `forward(input_ids, cache=None)` - Forward pass
- `generate(prompt_ids, max_length, temperature, top_p, top_k)` - Generate text

---

### `src.continual.continual_trainer`

#### `ContinualLearner`

Main interface for continual learning.

```python
from src.continual.continual_trainer import ContinualLearner, ContinualLearningConfig

learner = ContinualLearner(
    base_model=model,
    config=ContinualLearningConfig(
        lora_r=8,
        lora_alpha=16,
        replay_buffer_size=1000,
        use_ewc=True,
        ewc_lambda=1000
    ),
    tokenizer=tokenizer
)
```

**Methods:**
- `learn_from_text(text, domain, importance, num_epochs)` - Learn from text
- `generate(prompt, max_length, **kwargs)` - Generate text
- `save_checkpoint(path)` - Save model state
- `load_checkpoint(path)` - Load model state
- `from_checkpoint(path, tokenizer)` - Load from checkpoint (class method)

---

### `src.tokenization.bpe_tokenizer`

#### `BPETokenizer`

Byte-Pair Encoding tokenizer.

```python
from src.tokenization.bpe_tokenizer import BPETokenizer

# Train new tokenizer
tokenizer = BPETokenizer(vocab_size=8000)
tokenizer.train(texts)
tokenizer.save("path/to/tokenizer")

# Load existing tokenizer
tokenizer = BPETokenizer.load("path/to/tokenizer")
```

**Methods:**
- `train(texts, min_frequency)` - Train on texts
- `encode(text)` - Encode text to IDs
- `decode(ids)` - Decode IDs to text
- `save(path)` - Save tokenizer
- `load(path)` - Load tokenizer (class method)

---

### `src.lora.lora_model`

#### `LoRAModel`

LoRA wrapper for models.

```python
from src.lora.lora_model import LoRAModel, LoRAConfig

lora_model = LoRAModel(
    base_model,
    LoRAConfig(r=8, alpha=16, dropout=0.05)
)
```

**Methods:**
- `forward(x)` - Forward with LoRA
- `merge_adapter()` - Merge LoRA into base weights
- `save_adapter(path)` - Save LoRA weights only
- `load_adapter(path)` - Load LoRA weights

---

## Configuration Classes

### `ModelConfig`

```python
ModelConfig(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_query_heads: int,
    num_kv_heads: int,
    d_ff: int = None,
    max_seq_len: int = 2048,
    dropout: float = 0.0,
    attention_dropout: float = 0.0,
    norm_type: str = "rmsnorm",
    ff_type: str = "swiglu",
    rope_base: float = 10000.0
)
```

### `ContinualLearningConfig`

```python
ContinualLearningConfig(
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    replay_buffer_size: int = 1000,
    replay_sample_strategy: str = "uniform",
    use_ewc: bool = True,
    ewc_lambda: float = 1000.0,
    device: str = "auto"
)
```

---

## Utilities

### `src.evaluation.metrics`

```python
from src.evaluation.metrics import calculate_perplexity, calculate_accuracy

perplexity = calculate_perplexity(model, dataloader)
accuracy = calculate_accuracy(predictions, targets)
```

---

For more examples, see [User Guide](user-guide.md) and [examples/](../examples/).
