# Architecture

## Overview

MiniMacLLM is a brain-inspired continual learning language model that combines modern transformer architecture with neurobiologically-realistic learning mechanisms. Unlike traditional AI, the system decides its own learning dynamics from mathematical principles—no human presets.

## System Architecture

```text
┌─────────────────────────────────────────────────────────┐
│              MiniMacLLM Brain System                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │        Neurobiological Learning Control        │    │
│  │  • Autonomous Learning Rate (emergent)         │    │
│  │  • Neuromodulation (dopamine/serotonin/ACh)    │    │
│  │  • Homeostatic Plasticity (stability)          │    │
│  └────────────────┬───────────────────────────────┘    │
│                   │                                     │
│                   ▼                                     │
│  ┌──────────────┐      ┌──────────────────┐           │
│  │ Base Model   │◄─────┤ LoRA Adapters    │           │
│  │ (Frozen)     │      │ (Trainable)      │           │
│  │ 127M params  │      │ ~3M params each  │           │
│  └──────┬───────┘      └──────────────────┘           │
│         │                                               │
│         ▼                                               │
│  ┌─────────────────────────────────────────────────┐  │
│  │         Memory & Consolidation System           │  │
│  │  • Hippocampal Memory (pattern sep/completion)  │  │
│  │  • Experience Replay (intelligent rehearsal)    │  │
│  │  • Sleep Consolidation (offline strengthening)  │  │
│  │  • EWC (elastic weight protection)              │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
## Core Components

### 1. Brain-Inspired Learning Mechanisms

**Autonomous Learning Rate Controller** (`src/neurobio/autonomous_learning.py`):
- **No human presets** - Learning rate emerges from 5 mathematical factors:
  1. Prediction error (urgency)
  2. Gradient statistics (natural step size)
  3. Uncertainty (exploration vs exploitation)
  4. Success history (meta-learning)
  5. Metabolic cost (resource constraints)
- System decides its own learning speed, just like biological brains

**Neuromodulation System** (`src/neurobio/neuromodulation.py`):
- **Dopamine** - Modulates learning rate based on reward prediction error
- **Serotonin** - Balances stability vs plasticity
- **Acetylcholine** - Gates attention and memory formation
- **Noradrenaline** - Controls arousal and alertness
- Learnable parameters that adapt to task dynamics

**Hippocampal Memory** (`src/neurobio/hippocampal_memory.py`):
- **Pattern Separation** - 2% sparse encoding in dentate gyrus
- **Pattern Completion** - Content-based retrieval from CA3
- **Episodic Memory** - Stores important experiences for replay
- Replaces random sampling with intelligent memory selection

**Sleep Consolidation** (`src/neurobio/sleep_consolidation.py`):
- **Offline Replay** - Rehearse stored memories without new data
- **Hebbian Strengthening** - "Neurons that fire together, wire together"
- **Synaptic Pruning** - Remove weak connections
- **Noisy Dreams** - Add noise during replay for robustness

**Homeostatic Plasticity** (`src/neurobio/homeostasis.py`):
- **Synaptic Scaling** - Maintains stable neural activity
- **Adaptive Thresholds** - Prevents dead neurons
- **Target Firing Rate** - Self-regulating activation levels
- Nature 2024 finding: Critical for preventing forgetting

**Dendritic Computation** (`src/neurobio/dendritic.py`):
- **Compartmentalized Processing** - Multiple computational units per neuron
- **Context-Dependent Gating** - NMDA-like receptor modulation
- **Network-in-a-Neuron** - Efficient parameter usage
- Enables complex nonlinear computations

### 2. Modern Transformer Architecture

**Key Features:**
- **Grouped Query Attention (GQA)**: 4x KV cache reduction
- **Rotary Position Embeddings (RoPE)**: Better length extrapolation
- **SwiGLU Activation**: State-of-the-art feed-forward networks
- **RMSNorm**: Efficient normalization
- **Pre-normalization**: Stable training

**Model Configurations:**

| Size | Params | Dim | Layers | Heads (Q/KV) | FFN Dim |
|------|--------|-----|--------|--------------|---------|
| Tiny | 10M | 256 | 4 | 4/2 | 1024 |
| Small | 50M | 512 | 6 | 8/2 | 2048 |
| Medium | 127M | 768 | 12 | 12/3 | 3072 |
| Large | 500M | 1024 | 16 | 16/4 | 4096 |

### 2. Continual Learning System

**LoRA (Low-Rank Adaptation):**
- Adds trainable low-rank matrices to frozen base model
- Only 2-3% parameter overhead per domain
- Fast adaptation (seconds vs hours)
- Multiple adapters can coexist

**Experience Replay Buffer:**
- Stores important past examples
- Prevents forgetting through rehearsal
- Importance-weighted sampling
- Configurable size and strategies

**Elastic Weight Consolidation (EWC):**
- Computes Fisher Information Matrix
- Penalizes changes to important weights
- Mathematical guarantee against forgetting
- Online EWC for continuous updates

### 3. Tokenization

**BPE (Byte-Pair Encoding):**
- Trained from scratch on custom data
- Configurable vocabulary size (8K-32K)
- Handles any Unicode text
- Special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`

## Module Structure

```
src/
├── neurobio/               # Brain-inspired mechanisms ⭐ NEW
│   ├── autonomous_learning.py    # Emergent learning rate
│   ├── neuromodulation.py        # Dopamine/serotonin/ACh
│   ├── homeostasis.py            # Synaptic scaling
│   ├── dendritic.py              # Compartmentalized computation
│   ├── hippocampal_memory.py     # Pattern separation/completion
│   └── sleep_consolidation.py    # Offline strengthening
│
├── model/                  # Core language model
│   ├── llm.py             # Main model
│   ├── attention.py       # GQA implementation
│   ├── embeddings.py      # RoPE
│   ├── feedforward.py     # SwiGLU
│   ├── normalization.py   # RMSNorm
│   └── transformer_block.py
│
├── multimodal/             # Vision-language ⭐ NEW
│   ├── vision_encoder.py          # Image processing
│   ├── multimodal_tokenizer.py    # Unified tokenization
│   ├── fusion_layers.py           # Cross-modal attention
│   └── multimodal_llm.py          # Main multimodal model
│
├── lora/                   # Parameter-efficient fine-tuning
│   ├── lora_layer.py      # LoRA layers
│   └── lora_model.py      # Model wrapper
│
├── continual/              # Anti-forgetting
│   ├── experience_replay.py
│   ├── ewc.py
│   └── continual_trainer.py  # Main interface (with brain mechanisms)
│
├── tokenization/           # Byte-level BPE tokenizer
│   ├── bpe_tokenizer.py
│   └── vocab_trainer.py
│
├── training/               # Training utilities
├── evaluation/             # Metrics
├── data/                   # Data loading
└── utils/                  # Helpers
```
## Data Flow

### Training Flow

```
Raw Text → Tokenizer → Token IDs → Model → Logits → Loss
                                      ↑
                                      │
                              ┌───────┴────────┐
                              │                │
                         LoRA Weights    Base Weights
                         (trainable)      (frozen)
```

### Continual Learning Flow

```
New Domain Data
    │
    ▼
┌─────────────────┐
│ Experience      │
│ Replay Buffer   │
└────────┬────────┘
         │
         ▼
    Mix 50/50
         │
         ▼
┌─────────────────┐       ┌──────────────┐
│ Training Batch  │──────►│ Compute Loss │
└─────────────────┘       └──────┬───────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │ Add EWC      │
                          │ Penalty      │
                          └──────┬───────┘
                                 │
                                 ▼
                          Update LoRA Only
```

## Memory Efficiency

**Model Size Breakdown (Medium, 127M params):**

| Component | Parameters | Memory (FP32) |
|-----------|------------|---------------|
| Embeddings | 6M | 24MB |
| Transformer Layers | 118M | 472MB |
| Output Layer | 3M | 12MB |
| **Total Base** | **127M** | **508MB** |
| LoRA Adapter | 3M | 12MB |
| **Total with LoRA** | **130M** | **520MB** |

**Training Memory (batch_size=4, seq_len=1024):**
- Model: ~500MB
- Gradients: ~500MB
- Optimizer states: ~1GB
- Activations: ~15-20GB
- **Total: ~20-25GB**

## Performance Characteristics

**Training (M3 Max, Medium Model):**
- Speed: 3-4 iterations/second
- Throughput: ~50K tokens/second
- Memory: 20-25GB

**Inference:**
- Latency: ~50ms/token
- Memory: ~1GB (with KV cache)
- Batch size: 1-8 typical

**Continual Learning:**
- Update time: <10 seconds for 1000 examples
- Adapter size: ~12MB on disk
- Zero forgetting: <5% degradation on old tasks

## Design Decisions

### Why GQA over MHA?
- 4x reduction in KV cache size
- Minimal quality loss
- Modern best practice for efficient attention

### Why LoRA for Continual Learning?
- Only update 2-3% of parameters
- Fast adaptation
- Multiple adapters can coexist
- Easy to save/load

### Why Experience Replay + EWC?
- Replay prevents practical forgetting
- EWC provides mathematical guarantees
- Together: robust zero-forgetting system

### Why Custom BPE Tokenizer?
- Domain-specific vocabulary
- No dependencies on external models
- Full control over special tokens
- Educational value

## Extensibility

The architecture is designed for easy extension:

**Add new model components:**
- Implement in `src/model/`
- Register in `ModelConfig`

**Add new LoRA targets:**
- Modify `LoRAModel._inject_lora()`

**Add new replay strategies:**
- Implement in `ExperienceReplayBuffer`

**Add new evaluation metrics:**
- Implement in `src/evaluation/metrics.py`

## References

### Core Architecture
- Vaswani et al., 2017: "Attention Is All You Need"
- Su et al., 2021: "RoFormer: Rotary Position Embeddings"
- Ainslie et al., 2023: "GQA: Grouped-Query Attention"
- Shazeer, 2020: "GLU Variants Improve Transformer"

### Continual Learning
- Hu et al., 2021: "LoRA: Low-Rank Adaptation"
- Kirkpatrick et al., 2017: "Overcoming Catastrophic Forgetting"
- Zenke et al., 2017: "Continual Learning Through Synaptic Intelligence"

---

For implementation details, see:
- [User Guide](user-guide.md) - Usage examples
- [API Reference](api-reference.md) - Detailed API docs
- [Research Notes](research/) - Deep dives and planning docs
