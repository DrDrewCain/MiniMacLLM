# Evaluation and Testing Guide

Complete guide for evaluating and testing the continual learning language model system.

---

## Table of Contents

1. [Overview](#overview)
2. [Evaluation Metrics](#evaluation-metrics)
3. [Running Evaluations](#running-evaluations)
4. [Testing the System](#testing-the-system)
5. [Measuring Forgetting](#measuring-forgetting)
6. [Quality Assessment](#quality-assessment)
7. [Performance Benchmarks](#performance-benchmarks)

---

## Overview

The evaluation system provides comprehensive tools to measure:

- **Language Modeling Performance**: Perplexity, cross-entropy, accuracy
- **Catastrophic Forgetting**: How much old knowledge is retained
- **Generation Quality**: Coherence, diversity, relevance
- **Domain-Specific Performance**: Math, code, reasoning tasks

---

## Evaluation Metrics

### 1. Perplexity

Measures how well the model predicts text. Lower is better.

```python
from src.evaluation import compute_perplexity
from src.model.llm import ContinualLLM, ModelConfig
import torch

# Load model
model = ContinualLLM(ModelConfig())
input_ids = torch.tensor([[1, 2, 3, 4, 5]])

# Compute perplexity
result = compute_perplexity(model, input_ids)

print(f"Perplexity: {result.perplexity:.2f}")
print(f"Cross-Entropy: {result.cross_entropy:.2f}")
print(f"Bits per Byte: {result.bits_per_byte:.2f}")
```

**Interpretation:**
- Perplexity of 50: Model is "confused" between ~50 choices per token
- Perplexity of 10: Much better, ~10 choices
- Perplexity of 1: Perfect (unrealistic)

### 2. Forgetting Rate

Measures how much performance drops on old tasks after learning new ones.

```python
from src.evaluation import compute_forgetting_rate

# Performance after learning Task A
perf_after_A = {'math': 0.90, 'code': 0.85}

# Learn Task B (psychology)...

# Performance after learning Task B
perf_after_B = {'math': 0.87, 'code': 0.82, 'psychology': 0.88}

# Compute forgetting
metrics = compute_forgetting_rate(perf_after_B, perf_after_A)

print(f"Forgetting Rate: {metrics.forgetting_rate:.2%}")
print(f"Backward Transfer: {metrics.backward_transfer:+.2%}")
print(f"Forward Transfer: {metrics.forward_transfer:.2%}")
```

**Interpretation:**
- Forgetting Rate: Average drop in performance (lower is better)
- Backward Transfer: Negative = forgetting, Positive = improvement
- Forward Transfer: Performance on new tasks

### 3. Accuracy

Next-token prediction accuracy.

```python
from src.evaluation import compute_accuracy

accuracy = compute_accuracy(model, input_ids, labels, device="cpu")
print(f"Accuracy: {accuracy:.2%}")
```

### 4. Generation Quality

Measures quality of generated text.

```python
from src.evaluation import evaluate_generation_quality

generated = "The quick brown fox jumps over the lazy dog"

quality = evaluate_generation_quality(
    generated,
    metrics=['length', 'unique_tokens', 'repetition']
)

print(f"Length: {quality['length']} tokens")
print(f"Unique tokens: {quality['unique_tokens']:.2%}")
print(f"Repetition: {quality['repetition']:.2%}")
```

---

## Running Evaluations

### Basic Evaluation

```python
from src.evaluation import Evaluator, EvaluationConfig
from src.continual.continual_trainer import ContinualLearner
from torch.utils.data import DataLoader

# Setup
config = EvaluationConfig(
    device="mps",
    compute_forgetting=True,
    generation_prompts=[
        "What is machine learning?",
        "Explain neural networks",
        "Define continual learning"
    ]
)

evaluator = Evaluator(model, tokenizer, config)

# Evaluate on a dataset
result = evaluator.evaluate(
    dataloader=test_dataloader,
    domain="math",
    step=0
)

print(result)
```

### Multi-Domain Evaluation

```python
# Evaluate across multiple domains
test_datasets = {
    'math': math_dataloader,
    'code': code_dataloader,
    'general': general_dataloader
}

results = evaluator.evaluate_continual_learning(
    test_datasets,
    step=5,
    save_dir=Path("evaluation_results")
)

for domain, result in results.items():
    print(f"\n{domain.upper()}:")
    print(f"  Perplexity: {result.perplexity:.2f}")
    print(f"  Accuracy: {result.accuracy:.2%}")
```

### Tracking Learning Curves

```python
# Train and evaluate periodically
for epoch in range(10):
    # Training...
    learner.learn_from_batch(batch)

    # Evaluate every epoch
    if epoch % 1 == 0:
        result = evaluator.evaluate(
            test_dataloader,
            domain="general",
            step=epoch
        )

# Get learning curve
steps, perplexities = evaluator.get_learning_curve(metric='perplexity')

# Plot (requires matplotlib)
import matplotlib.pyplot as plt
plt.plot(steps, perplexities)
plt.xlabel('Training Steps')
plt.ylabel('Perplexity')
plt.title('Learning Curve')
plt.savefig('learning_curve.png')
```

### Generating Reports

```python
# Generate comprehensive report
report = evaluator.generate_report(
    save_path=Path("evaluation_report.txt")
)

print(report)
```

---

## Testing the System

### Running Unit Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_attention.py -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

### Running Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific test
pytest tests/integration/test_continual_learning.py::TestMultiDomainLearning -v
```

### Test Coverage Areas

**Unit Tests:**
- ✓ `test_normalization.py` - RMSNorm tests
- ✓ `test_embeddings.py` - RoPE tests
- ✓ `test_attention.py` - GQA tests
- ✓ `test_lora.py` - LoRA layer tests
- ✓ `test_evaluation.py` - Metrics tests

**Integration Tests:**
- ✓ `test_continual_learning.py` - End-to-end continual learning
- Multi-domain learning
- Anti-forgetting mechanisms
- Checkpoint management
- Numerical stability

---

## Measuring Forgetting

### Step-by-Step Forgetting Analysis

```python
from src.evaluation import Evaluator, EvaluationConfig
from src.continual.continual_trainer import ContinualLearner

# Initialize
evaluator = Evaluator(model, tokenizer, EvaluationConfig())

# Step 1: Train on Math
for text in math_texts:
    learner.learn_from_text(text, domain="math")

# Evaluate math performance
result_math_initial = evaluator.evaluate(math_loader, domain="math", step=0)
print(f"Math (initial): {result_math_initial.perplexity:.2f}")

# Step 2: Train on Code
for text in code_texts:
    learner.learn_from_text(text, domain="code")

# Re-evaluate math (check for forgetting)
result_math_after_code = evaluator.evaluate(math_loader, domain="math", step=1)
print(f"Math (after code): {result_math_after_code.perplexity:.2f}")

# Evaluate code
result_code = evaluator.evaluate(code_loader, domain="code", step=1)
print(f"Code: {result_code.perplexity:.2f}")

# Compute forgetting
if result_math_after_code.forgetting_metrics:
    fm = result_math_after_code.forgetting_metrics
    print(f"\nForgetting Analysis:")
    print(f"  Forgetting Rate: {fm.forgetting_rate:.2%}")
    print(f"  Math degradation: {fm.backward_transfer:+.2%}")
```

### Forgetting Curve

```python
# Track forgetting over multiple tasks
domains = ['math', 'code', 'history', 'science', 'psychology']

for i, domain in enumerate(domains):
    # Learn new domain
    learner.learn_from_text(domain_data[domain], domain=domain)

    # Evaluate all previous domains
    for prev_domain in domains[:i+1]:
        result = evaluator.evaluate(
            test_loaders[prev_domain],
            domain=prev_domain,
            step=i
        )
        print(f"Step {i}, {prev_domain}: {result.perplexity:.2f}")

# Get forgetting curve for specific domain
steps, forgetting = evaluator.get_forgetting_curve("math")
```

---

## Quality Assessment

### Math Understanding

Test mathematical reasoning:

```python
from src.evaluation import compute_domain_specific_accuracy

math_test_cases = [
    ("2 + 2 =", "4"),
    ("3 * 5 =", "15"),
    ("sqrt(16) =", "4"),
    ("10 / 2 =", "5")
]

results = compute_domain_specific_accuracy(
    model,
    math_test_cases,
    tokenizer,
    device="mps"
)

print(f"Math Accuracy: {results['accuracy']:.2%}")
print(f"Correct: {results['correct']}/{results['total']}")

# See individual results
for result in results['results']:
    print(f"Prompt: {result['prompt']}")
    print(f"Expected: {result['expected']}")
    print(f"Got: {result['generated']}")
    print(f"Correct: {result['correct']}\n")
```

### Code Understanding

```python
code_test_cases = [
    ("def factorial(n):\n    if n == 0:", "return 1"),
    ("for i in range(10):", "# loop body"),
    ("import numpy as", "np"),
]

results = compute_domain_specific_accuracy(
    model,
    code_test_cases,
    tokenizer
)

print(f"Code Accuracy: {results['accuracy']:.2%}")
```

### Generation Diversity

```python
# Generate multiple samples
prompts = ["Tell me about", "Explain the concept of", "What is"]
generations = []

for prompt in prompts:
    for _ in range(5):  # 5 samples per prompt
        gen = learner.generate(prompt, temperature=0.8)
        generations.append(gen)

# Analyze diversity
from src.evaluation import evaluate_generation_quality

for gen in generations[:3]:  # Show first 3
    quality = evaluate_generation_quality(gen)
    print(f"Unique tokens: {quality['unique_tokens']:.2%}")
    print(f"Repetition: {quality['repetition']:.2%}")
    print(f"Text: {gen[:100]}...\n")
```

---

## Performance Benchmarks

### Expected Performance

**Small Model (50M params):**
- Perplexity: 30-50 (after pre-training)
- Accuracy: 30-40%
- Generation speed: ~50ms/token (M1 Mac)
- Memory: ~200MB

**Medium Model (200M params):**
- Perplexity: 15-25 (after pre-training)
- Accuracy: 40-50%
- Generation speed: ~100ms/token (M1 Mac)
- Memory: ~800MB

**Continual Learning:**
- Update time: 5-10 seconds (1000 examples)
- Forgetting rate: < 5% (with EWC + Replay)
- Visible improvement: After 1000 examples

### Benchmarking Your System

```python
import time
import torch

# 1. Inference speed
model.eval()
input_ids = torch.randint(0, 1000, (1, 100))

start = time.time()
with torch.no_grad():
    for _ in range(100):  # 100 forward passes
        logits, _, _ = model(input_ids)
end = time.time()

print(f"Avg inference time: {(end - start) / 100 * 1000:.2f}ms")

# 2. Memory usage
if torch.backends.mps.is_available():
    # MPS memory (approximate)
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Model size: {params / 1024**2:.2f} MB")

# 3. Training speed
start = time.time()
metrics = learner.learn_from_text("Test " * 100, domain="test")
end = time.time()

print(f"Update time: {end - start:.2f}s")
```

---

## Example: Complete Evaluation Workflow

```python
from pathlib import Path
from src.model.llm import ContinualLLM, ModelConfig
from src.tokenization.bpe_tokenizer import BPETokenizer
from src.continual.continual_trainer import ContinualLearner, ContinualLearningConfig
from src.evaluation import Evaluator, EvaluationConfig
from torch.utils.data import DataLoader

# 1. Load model and tokenizer
model = ContinualLLM.load("checkpoints/my_model/model.pt")
tokenizer = BPETokenizer.load("tokenizers/my_tokenizer")

# 2. Create continual learner
learner = ContinualLearner(
    model,
    ContinualLearningConfig(use_ewc=True, device="mps"),
    tokenizer
)

# 3. Setup evaluator
evaluator = Evaluator(
    model,
    tokenizer,
    EvaluationConfig(
        device="mps",
        compute_forgetting=True,
        generation_prompts=[
            "Explain machine learning",
            "Write a Python function",
            "What is psychology"
        ]
    )
)

# 4. Prepare test datasets
math_loader = DataLoader(math_dataset, batch_size=4)
code_loader = DataLoader(code_dataset, batch_size=4)

# 5. Initial evaluation
print("=== Initial Evaluation ===")
result_math_0 = evaluator.evaluate(math_loader, domain="math", step=0)
result_code_0 = evaluator.evaluate(code_loader, domain="code", step=0)

# 6. Learn new domain (psychology)
print("\n=== Learning Psychology ===")
for text in psychology_texts:
    learner.learn_from_text(text, domain="psychology")

# 7. Re-evaluate (check forgetting)
print("\n=== Post-Learning Evaluation ===")
result_math_1 = evaluator.evaluate(math_loader, domain="math", step=1)
result_code_1 = evaluator.evaluate(code_loader, domain="code", step=1)

# 8. Generate report
print("\n=== Report ===")
evaluator.print_summary()
report = evaluator.generate_report(save_path=Path("eval_report.txt"))

# 9. Save history
evaluator.save_history(Path("eval_history.json"))

print("\n✓ Evaluation complete!")
```

---

## Troubleshooting

### Issue: High Perplexity

**Possible Causes:**
- Model not trained enough
- Test data different from training data
- Model too small for task complexity

**Solutions:**
- Train longer
- Increase model size
- Use domain-specific pre-training

### Issue: High Forgetting Rate

**Possible Causes:**
- EWC lambda too low
- Replay buffer too small
- Not enough consolidation

**Solutions:**
```python
config = ContinualLearningConfig(
    ewc_lambda=5000.0,  # Increase (default: 1000)
    replay_buffer_size=50000,  # Increase (default: 10000)
    replay_ratio=0.7,  # More replay (default: 0.5)
    consolidation_frequency=500  # More frequent (default: 1000)
)
```

### Issue: Low Generation Quality

**Possible Causes:**
- Too greedy (temperature=0)
- Too random (temperature > 1.5)
- Repetition issues

**Solutions:**
```python
# Adjust generation parameters
generated = learner.generate(
    prompt,
    temperature=0.8,  # Try 0.7-0.9
    top_k=50,  # Limit to top 50 tokens
    top_p=0.9,  # Nucleus sampling
    repetition_penalty=1.2  # Penalize repetition
)
```

---

## Best Practices

1. **Evaluate Regularly**: After every major training milestone
2. **Track Multiple Metrics**: Don't rely on just one metric
3. **Use Multiple Test Sets**: Different domains, difficulties
4. **Monitor Forgetting**: Evaluate old tasks after learning new ones
5. **Save Evaluation History**: Track progress over time
6. **Generate Samples**: Qualitative assessment is important
7. **Test Edge Cases**: Empty inputs, very long sequences, etc.

---

## Additional Resources

- **Unit Tests**: See `tests/unit/` for component-level tests
- **Integration Tests**: See `tests/integration/` for end-to-end tests
- **Metrics Documentation**: See `src/evaluation/metrics.py`
- **Evaluator Documentation**: See `src/evaluation/evaluator.py`

---

**Happy Evaluating!** 📊
