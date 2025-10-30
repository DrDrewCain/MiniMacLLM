# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete continual learning LLM implementation
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Experience Replay Buffer for anti-forgetting
- Elastic Weight Consolidation (EWC) for weight protection
- BPE tokenizer from scratch
- Pre-training, continual learning, and generation scripts
- Comprehensive test suite (122 tests, 100% passing)
- Model configurations (tiny, small, medium, large)
- Evaluation metrics framework
- Data loading and preprocessing utilities

### Architecture
- Modern transformer with GQA (Grouped Query Attention)
- RoPE (Rotary Position Embeddings)
- SwiGLU activation functions
- RMSNorm for normalization
- Apple Silicon (MPS) optimization

### Documentation
- Architecture documentation
- Quick start guide
- API reference
- Research notes and requirements

## [0.1.0] - 2025-01-XX

### Initial Release
- First working version of continual learning LLM
- Core features implemented and tested
- Ready for research and experimentation

---

## Version History

- **v0.1.0**: Initial release with core continual learning features
- **Unreleased**: Active development

## Future Roadmap

See [Project Status](docs/project-status.md) for planned features.
