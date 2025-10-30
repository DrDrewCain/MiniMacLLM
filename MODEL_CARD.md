# MiniMacLLM-127M Model Card

**A 127M parameter continual learning language model**

---

## Model Summary

- **Model Type**: Decoder-only Transformer
- **Parameters**: 127,224,576
- **Architecture**: Modern LLM (GQA, RoPE, SwiGLU, RMSNorm)
- **Primary Feature**: Continual learning without catastrophic forgetting
- **Training Data**: WikiText-2 (Wikipedia articles)
- **License**: MIT

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Parameters | 127.2M |
| Vocabulary | 8,000 tokens (BPE) |
| Layers | 12 |
| Hidden Size | 768 |
| Attention Heads | 12 query / 3 key-value (GQA 4:1) |
| Max Context | 2,048 tokens |
| Model Size | 485 MB (fp32) |
| Training Loss | 5.32 |
| Training Time | 2.5 hours (M3 Max) |

---

## Architecture

```yaml
Type: Decoder-only Transformer (LLaMA-style)
Position Encoding: RoPE (Rotary Position Embeddings)
Attention: Grouped Query Attention (4:1 ratio)
Activation: SwiGLU
Normalization: RMSNorm (pre-norm)
```

---

## Continual Learning System

### The Innovation

This model demonstrates **real-time learning without forgetting** using:

1. **LoRA**: Add domains with only ~2.8M params (+2.2%)
2. **Experience Replay**: Rehearse old knowledge while learning new
3. **EWC**: Mathematical protection against forgetting

### Multi-Domain Scaling

```
Base model: 127M (frozen)
+ Python:   +2.8M (+2.2%)
+ Math:     +2.8M (+2.2%)
+ Medical:  +2.8M (+2.2%)
─────────────────────────
Total: ~135M for 3 domains

vs 3 separate models: 381M
Savings: 2.8× fewer parameters
```

---

## Training Details

- **Dataset**: WikiText-2 (~2.1M tokens, 10MB)
- **Epochs**: 5
- **Batch Size**: 16 (4 physical × 4 accumulation)
- **Learning Rate**: 3e-4 (AdamW)
- **Hardware**: Apple M3 Max (36GB)
- **Device**: MPS (Metal Performance Shaders)
- **Duration**: 152 minutes

**Loss progression**: 6.74 → 6.22 → 5.91 → 5.62 → 5.32

---

## Usage

See [README.md](README.md) for complete usage examples.

### Quick Generation

```python
import torch
from src.model.llm import ContinualLLM
from src.tokenization.bpe_tokenizer import BPETokenizer

# Load
tokenizer = BPETokenizer.load("data/tokenizers/wikitext_8k")
checkpoint = torch.load("checkpoints/wikitext_medium/final/model.pt")
model = ContinualLLM.from_checkpoint(checkpoint)

# Generate
prompt = "The game takes place"
response = model.generate(prompt, tokenizer, max_new_tokens=50)
```

### Continual Learning

```bash
# Learn Python domain
python scripts/continual_learn.py \
  --model checkpoints/wikitext_medium/final/model.pt \
  --tokenizer data/tokenizers/wikitext_8k \
  --data data/continual_learning/python_basics.txt \
  --domain programming \
  --update_steps 500 \
  --use_ewc
```

---

## Limitations

### Current State

⚠️ **This is a research/demonstration model**, not production-ready:

- **High loss (5.32)**: Needs more training for coherent generation
- **Small vocabulary (8K)**: vs typical 32K-50K tokens
- **Limited training data**: Only ~10MB of text
- **No safety alignment**: May generate inappropriate content

### Recommendations for Production Use

1. Train 15-20 more epochs (target loss < 4.0)
2. Use larger dataset (WikiText-103 or custom corpus)
3. Train 32K token vocabulary
4. Add safety filtering and alignment
5. Consider scaling to 200M-500M parameters

---

## Intended Use

### ✅ Appropriate Uses

- Research on continual learning
- Educational demonstrations
- Prototyping domain adaptation
- Experimenting with LLM architectures
- Low-resource environments (Apple Silicon)

### ❌ NOT Recommended For

- Production chat applications
- High-stakes decisions
- Tasks requiring factual accuracy
- General-purpose assistance
- User-facing applications (without extensive testing)

---

## Ethical Considerations

### Biases

- Trained on Wikipedia (known demographic and topic biases)
- No debiasing applied
- Should not be used for decisions affecting people

### Safety

- No safety alignment performed
- No red-teaming conducted
- May generate harmful content
- Requires output filtering for any public use

### Environmental Impact

- Training: ~2.5 hours on M3 Max
- Energy: ~0.3 kWh (estimated)
- Carbon: Minimal (Apple Silicon efficiency)

---

## Model Details

### File Structure

```
checkpoints/wikitext_medium/final/
├── model.pt                 # Full checkpoint (485MB)
│   ├── model_state_dict     # Model weights
│   ├── config              # Model configuration
│   ├── loss                # Final training loss
│   └── epoch               # Training epoch
│
data/tokenizers/wikitext_8k/
├── vocab.json              # Vocabulary (148KB)
├── merges.json             # BPE merges (671KB)
└── config.json             # Tokenizer config (124B)
```

### Model Checkpoint Contents

```python
checkpoint = {
    'model_state_dict': OrderedDict(...),  # 127.2M parameters
    'config': {
        'vocab_size': 32000,
        'd_model': 768,
        'num_layers': 12,
        'num_query_heads': 12,
        'num_kv_heads': 3,
        'd_ff': 3072,
        'max_seq_len': 2048,
        # ... more config
    },
    'loss': 5.324,
    'epoch': 5
}
```

---

## Citation

```bibtex
@misc{minimacllm2025,
  title={MiniMacLLM: A Continual Learning Language Model},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/MiniMacLLM},
  note={127M parameter model demonstrating continual learning without catastrophic forgetting}
}
```

---

## References

### Architecture Papers

- **RoPE**: Su et al. (2021) - RoFormer: Enhanced Transformer with Rotary Position Embedding
- **GQA**: Ainslie et al. (2023) - GQA: Training Generalized Multi-Query Transformer Models
- **SwiGLU**: Shazeer (2020) - GLU Variants Improve Transformer

### Continual Learning Papers

- **LoRA**: Hu et al. (2021) - Low-Rank Adaptation of Large Language Models
- **EWC**: Kirkpatrick et al. (2017) - Overcoming catastrophic forgetting in neural networks
- **Experience Replay**: Zenke et al. (2017) - Continual Learning Through Synaptic Intelligence

### Model Inspirations

- **LLaMA 3** (Meta AI, 2024) - Architecture design
- **Mistral** (Mistral AI, 2023) - Efficiency techniques
- **T5** (Google, 2020) - RMSNorm normalization

---

## License

MIT License - See [LICENSE](LICENSE) file

---

## Version

**Version 1.0.0** (2025-01-30)

- Initial release
- Base model trained on WikiText-2
- Continual learning system fully implemented
- Scripts for training, learning, and generation

---

**Questions?** Open an issue on [GitHub](https://github.com/YOUR_USERNAME/MiniMacLLM)
