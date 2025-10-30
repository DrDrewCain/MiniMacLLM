# Continual LLM Documentation

Welcome to the Continual LLM documentation! This system implements a research-level continual learning framework for language models.

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

## What is Continual Learning?

Continual learning allows models to:
- Learn new information in real-time (seconds, not hours)
- Never forget previous knowledge (zero catastrophic forgetting)
- Adapt to multiple domains efficiently
- Run on consumer hardware (Apple Silicon, small GPUs)

## Key Features

### Modern Architecture
- Grouped Query Attention (GQA) - 4x cache reduction
- Rotary Position Embeddings (RoPE)
- SwiGLU activation functions
- RMSNorm normalization

### Continual Learning System
- **LoRA** - Fast, parameter-efficient updates
- **Experience Replay** - Prevents forgetting old knowledge
- **EWC** - Mathematical weight protection
- **Multi-adapter** - Domain-specific expertise

### Production Ready
- 100% test coverage (122/122 tests passing)
- Complete training pipeline
- Comprehensive evaluation metrics
- Optimized for Apple Silicon

## Research Background

This system is based on cutting-edge research:
- LoRA: Hu et al., 2021
- EWC: Kirkpatrick et al., 2017
- Experience Replay: Zenke et al., 2017

See [Research Notes](research/) for details.

## Use Cases

1. **Personal AI Assistant** - Learns from your documents
2. **Domain Expert** - Medical, legal, engineering
3. **Research Tool** - Study continual learning
4. **Production Apps** - Customer support, code assistants

## Support

- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)
- **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md)

---

**Ready to get started?** → [Getting Started Guide](getting-started.md)
