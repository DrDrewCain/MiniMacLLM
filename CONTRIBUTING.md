# Contributing to Continual LLM

Thank you for your interest in contributing! This project implements a research-level continual learning system for language models.

## How to Contribute

### Reporting Bugs

- Check if the bug has already been reported in [Issues](../../issues)
- Use the bug report template
- Include:
  - Python version and OS
  - Steps to reproduce
  - Expected vs actual behavior
  - Error messages and logs

### Suggesting Enhancements

- Open an issue with the enhancement label
- Clearly describe the feature and its benefits
- Consider implementation complexity

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation

4. **Run tests**
   ```bash
   pytest tests/
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: description"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/continual-llm.git
cd continual-llm

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run specific test
pytest tests/unit/test_lora.py -v
```

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions focused and testable

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

## Documentation

- Update relevant docs in `docs/`
- Add docstrings to new functions/classes
- Update README.md if adding major features

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers get started

## Questions?

Open an issue with the "question" label or start a discussion.

---

**Thank you for contributing to continual learning research!** 🚀
