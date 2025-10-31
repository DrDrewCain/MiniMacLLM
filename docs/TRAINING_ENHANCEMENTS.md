# Training Quality Enhancements

## Overview

The pretraining script now includes several enhancements to improve model quality and training efficiency:

1. **Validation Tracking** - Monitor validation loss and perplexity during training
2. **Learning Rate Scheduling** - Warmup + cosine decay for better convergence
3. **Early Stopping** - Automatically stop when validation loss plateaus
4. **Best Checkpoint Saving** - Save the best model (not just the final one)
5. **Perplexity Monitoring** - Track perplexity in real-time

## Quick Start

### Basic Training (with defaults)

```bash
python scripts/pretrain.py \
  --config configs/medium.yaml \
  --data data/raw/wikitext2_train.txt \
  --tokenizer data/tokenizers/my_tokenizer \
  --epochs 30 \
  --batch_size 4 \
  --save_dir checkpoints/base_model
```

This will:
- Automatically use 10% of training data for validation
- Apply learning rate warmup (500 steps) and cosine decay
- Stop early if no improvement for 3 epochs
- Save best checkpoint to `checkpoints/base_model/best/`

### Advanced Training (custom settings)

```bash
python scripts/pretrain.py \
  --config configs/medium.yaml \
  --data data/raw/wikitext2_train.txt \
  --val_data data/raw/wikitext2_val.txt \
  --tokenizer data/tokenizers/my_tokenizer \
  --epochs 50 \
  --batch_size 4 \
  --learning_rate 3e-4 \
  --warmup_steps 1000 \
  --early_stopping_patience 5 \
  --val_split 0.0 \
  --save_dir checkpoints/base_model_v2
```

This configuration:
- Uses separate validation file (`--val_data`)
- Longer warmup period (1000 steps)
- More patient early stopping (5 epochs)
- Disables automatic validation split since we have a separate file

## Training Output

You'll see output like this:

```
==============================================================
Pre-training LLM
==============================================================
Device: mps
Data: data/raw/wikitext2_train.txt
Validation: 10% split
Tokenizer: data/tokenizers/my_tokenizer
Batch size: 4
Gradient accumulation: 4
Learning rate: 0.0003
Epochs: 30
Warmup steps: 500
Early stopping patience: 3
==============================================================

Loading data from data/raw/wikitext2_train.txt...
Loaded 36718 texts
Splitting 10% of training data for validation...
Training samples: 33046, Validation samples: 3672

Model configuration:
  Vocabulary: 8,000
  Model dim: 768
  Layers: 12
  Query heads: 12
  KV heads: 3
  Feed-forward: 3072

Model created: 200,123,456 parameters

Total training steps: 8261
Learning rate schedule: warmup 500 steps, then cosine decay

Starting training...

Epoch 1/30
  Train Loss: 5.2341 | Train PPL: 187.23
  Val Loss:   5.1234 | Val PPL:   168.45
  ✓ New best checkpoint saved (val_loss: 5.1234, val_ppl: 168.45)
  Checkpoint saved to checkpoints/base_model/epoch_1

Epoch 2/30
  Train Loss: 4.8765 | Train PPL: 131.52
  Val Loss:   4.9123 | Val PPL:   136.23
  ✓ New best checkpoint saved (val_loss: 4.9123, val_ppl: 136.23)
  Checkpoint saved to checkpoints/base_model/epoch_2

...

Epoch 15/30
  Train Loss: 3.2341 | Train PPL: 25.36
  Val Loss:   3.4567 | Val PPL:   31.68
  No improvement for 1 epoch(s)
  Checkpoint saved to checkpoints/base_model/epoch_15

Epoch 16/30
  Train Loss: 3.1876 | Train PPL: 24.21
  Val Loss:   3.4512 | Val PPL:   31.51
  ✓ New best checkpoint saved (val_loss: 3.4512, val_ppl: 31.51)
  Checkpoint saved to checkpoints/base_model/epoch_16

...

Training complete! Time: 125.34 minutes
Best model: Epoch 16 with validation loss 3.4512
Best checkpoint available at: checkpoints/base_model/best/checkpoint.pt
```

## Key Parameters

### Validation Options

- `--val_data PATH`: Path to separate validation file (optional)
- `--val_split RATIO`: Fraction of training data to use for validation (default: 0.1)
  - Set to 0.0 if using `--val_data`

### Learning Rate Scheduling

- `--learning_rate FLOAT`: Base learning rate (default: 3e-4)
- `--warmup_steps INT`: Number of warmup steps (default: 500)
  - LR linearly increases from 0 to base LR
  - Recommended: ~1-5% of total training steps

### Early Stopping

- `--early_stopping_patience INT`: Stop after N epochs without improvement (default: 3)
  - Set to 0 to disable early stopping
  - Recommended: 3-5 for most cases

### Training Length

- `--epochs INT`: Maximum number of epochs (default: 3)
  - Recommended: 20-50 for good convergence
  - With early stopping, training may end sooner

## Target Metrics

For a well-trained WikiText-2 base model:

- **Validation Perplexity**: < 20
- **Training Loss**: < 3.0
- **Validation Loss**: < 3.5
- **Generation Quality**: Grammatically coherent sentences

## Checkpoint Files

After training, you'll have:

```
checkpoints/base_model/
├── best/
│   └── checkpoint.pt          # Best validation loss
├── epoch_1/
│   └── checkpoint.pt
├── epoch_2/
│   └── checkpoint.pt
...
└── final/
    └── model.pt               # Final epoch (for compatibility)
```

**Recommendation**: Use `best/checkpoint.pt` for downstream tasks and continual learning.

## Loading a Checkpoint

```python
import torch
from src.model.llm import ContinualLLM, ModelConfig

# Load best checkpoint
checkpoint = torch.load('checkpoints/base_model/best/checkpoint.pt')

# Recreate model
config = ModelConfig(**checkpoint['config'])
model = ContinualLLM(config)
model.load_state_dict(checkpoint['model_state_dict'])

# Check metrics
print(f"Epoch: {checkpoint['epoch']}")
print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
print(f"Validation Perplexity: {checkpoint['val_perplexity']:.2f}")
```

## Troubleshooting

### High Perplexity After Many Epochs

**Problem**: Perplexity stays above 50 after 20+ epochs

**Solutions**:
1. Check if model is too small (try larger config)
2. Increase learning rate to 5e-4
3. Verify data quality and tokenization
4. Try longer warmup (1000 steps)

### Early Stopping Too Soon

**Problem**: Training stops after only a few epochs

**Solutions**:
1. Increase patience: `--early_stopping_patience 5`
2. Check if learning rate is too high (causing instability)
3. Verify validation data isn't too different from training data

### Validation Loss Increasing

**Problem**: Validation loss starts increasing while training loss decreases

**Solutions**:
1. This is overfitting - reduce model size or increase data
2. Enable dropout in model config
3. Stop training earlier (model was best a few epochs ago)
4. Use the best checkpoint, not the final one

### Out of Memory

**Problem**: GPU/MPS runs out of memory during training

**Solutions**:
1. Reduce batch size: `--batch_size 2`
2. Increase gradient accumulation: `--grad_accumulation 8`
3. Reduce max sequence length: `--max_seq_len 512`
4. Use smaller model config

## Best Practices

1. **Always use validation**: Either via `--val_data` or `--val_split`
2. **Monitor perplexity**: Target < 20 for WikiText-2
3. **Use best checkpoint**: Load from `best/` directory for production
4. **Train longer**: 20-50 epochs for convergence (with early stopping as safety)
5. **Save checkpoints**: Keep them for comparison and rollback
6. **Check generation**: Manually test with `scripts/generate.py` to verify quality

## Example Training Pipeline

```bash
#!/bin/bash

# Step 1: Train tokenizer
python scripts/train_tokenizer.py \
  --data data/raw/wikitext2_train.txt \
  --vocab_size 8000 \
  --save data/tokenizers/wikitext_8k

# Step 2: Pretrain with quality settings
python scripts/pretrain.py \
  --config configs/medium.yaml \
  --data data/raw/wikitext2_train.txt \
  --val_data data/raw/wikitext2_val.txt \
  --tokenizer data/tokenizers/wikitext_8k \
  --epochs 50 \
  --batch_size 4 \
  --warmup_steps 1000 \
  --early_stopping_patience 5 \
  --save_dir checkpoints/wikitext_medium_v2

# Step 3: Test generation quality
python scripts/generate.py \
  --model checkpoints/wikitext_medium_v2/best/checkpoint.pt \
  --tokenizer data/tokenizers/wikitext_8k \
  --prompt "Python is a programming language that" \
  --max_tokens 50

# Step 4: If quality is good, use for continual learning
python scripts/continual_learn.py \
  --model checkpoints/wikitext_medium_v2/best/checkpoint.pt \
  --tokenizer data/tokenizers/wikitext_8k \
  --data data/medical_domain.txt \
  --domain medical \
  --epochs 3
```

## References

- Related: Evaluation metrics in `src/evaluation/metrics.py`
- Tests: `tests/unit/test_pretrain_enhancements.py`
