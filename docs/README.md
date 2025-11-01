# MiniMacLLM Documentation

Welcome to the MiniMacLLM documentation! This system implements a brain-inspired continual learning framework for language models with neurobiologically-realistic mechanisms.

## Table of Contents

### Getting Started
- **[Getting Started](getting-started.md)** - Installation, quick start, and first steps
- **[Examples](../examples/)** - Code examples and tutorials

### Core Documentation
- **[Architecture](architecture.md)** - System architecture and design decisions
- **[User Guide](user-guide.md)** - Training, continual learning, and generation
- **[API Reference](api-reference.md)** - Detailed API documentation

### Advanced Topics
- **[Evaluation](evaluation.md)** - Testing and benchmarking your models
- **[Project Status](project-status.md)** - Current status and roadmap

### Research & Development
- **[Research Notes](research/)** - Background research and design decisions

## Quick Links

**New to continual learning?** Start with [Getting Started](getting-started.md)

**Want to train a model?** See [User Guide - Training](user-guide.md#training)

**Looking for API docs?** Check [API Reference](api-reference.md)

**Found a bug?** See [Contributing](../CONTRIBUTING.md)

## What Makes MiniMacLLM Different?

Unlike traditional AI where humans set learning parameters, MiniMacLLM implements **true brain-inspired learning** where the system decides its own learning dynamics from mathematical principles:

- **Autonomous Learning Rate** - System determines learning speed from prediction error, uncertainty, success history, and metabolic constraints
- **Neuromodulation** - Dopamine, serotonin, and acetylcholine control learning dynamics
- **Hippocampal Memory** - Pattern separation and completion for intelligent replay
- **Sleep Consolidation** - Offline Hebbian strengthening without new data
- **Homeostatic Plasticity** - Prevents dead neurons and maintains capacity
- **Dendritic Computation** - Compartmentalized processing like real neurons

### Continual Learning Capabilities

- **Zero Catastrophic Forgetting** - Mathematical guarantees via EWC + Experience Replay
- **Real-Time Adaptation** - Learn new domains in 5-10 seconds
- **Multi-Domain Mastery** - One base model, unlimited specialized adapters
- **Local Deployment** - Runs on Apple Silicon (M1/M2/M3)

## Key Features

### Brain-Inspired Mechanisms
- **Autonomous Learning** - No human presets, emergent learning rate
- **Neuromodulation** - Chemical control of plasticity
- **Hippocampal Memory** - 2% sparse pattern separation
- **Sleep Consolidation** - Offline strengthening
- **Homeostatic Plasticity** - Stable neural activity
- **Dendritic Computation** - Network-in-a-neuron

### Modern Transformer Architecture
- Grouped Query Attention (GQA) - 4x cache reduction
- Rotary Position Embeddings (RoPE)
- SwiGLU activation functions
- RMSNorm normalization
- Byte-level BPE tokenization

### Continual Learning System
- **LoRA** - Fast, parameter-efficient updates (2.2% overhead)
- **Experience Replay** - Intelligent memory rehearsal
- **EWC** - Elastic Weight Consolidation
- **Multi-adapter** - Domain-specific expertise

### Production Ready
- 165+ tests passing
- Complete training pipeline
- Comprehensive evaluation metrics
- Optimized for Apple Silicon
- Multi-modal support (vision + text)

## Research Background

This system combines cutting-edge research:

**Continual Learning:**
- LoRA: Hu et al., 2021
- EWC: Kirkpatrick et al., 2017
- Experience Replay: Zenke et al., 2017

**Neuroscience:**
- Neuromodulation: Dayan & Yu, 2006
- Homeostatic Plasticity: Turrigiano, 2008; Nature, 2024
- Hippocampal Memory: Rolls, 2013
- Sleep Consolidation: NeuroDream Framework, Dec 2024

See [Research Notes](research/) for details.

## Use Cases

1. **Personal AI Assistant** - Learns from your documents with brain-like plasticity
2. **Domain Expert** - Medical, legal, engineering specialists
3. **Research Tool** - Study continual learning and neuroscience
4. **Production Apps** - Customer support, code assistants with adaptive learning
5. **Educational** - Understand how brains learn through code

## Support

- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)
- **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md)

---

**Ready to get started?** → [Getting Started Guide](getting-started.md)
