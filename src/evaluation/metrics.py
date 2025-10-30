"""
Evaluation metrics for continual learning LLMs.

This module provides various metrics to evaluate:
1. Language modeling performance (perplexity, cross-entropy)
2. Catastrophic forgetting (forgetting rate, backward transfer)
3. Generation quality
4. Domain-specific performance
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math


@dataclass
class PerplexityResult:
    """Results from perplexity computation."""
    perplexity: float
    cross_entropy: float
    num_tokens: int
    bits_per_byte: float

    def __repr__(self) -> str:
        return (f"PerplexityResult(perplexity={self.perplexity:.4f}, "
                f"cross_entropy={self.cross_entropy:.4f}, "
                f"num_tokens={self.num_tokens}, "
                f"bits_per_byte={self.bits_per_byte:.4f})")


@dataclass
class ForgettingMetrics:
    """Metrics for measuring catastrophic forgetting."""
    forgetting_rate: float  # Average performance drop on old tasks
    backward_transfer: float  # How much old tasks improve/degrade
    forward_transfer: float  # How much new learning helps old tasks
    task_performances: Dict[str, float]  # Performance per task

    def __repr__(self) -> str:
        return (f"ForgettingMetrics(forgetting_rate={self.forgetting_rate:.4f}, "
                f"backward_transfer={self.backward_transfer:.4f}, "
                f"forward_transfer={self.forward_transfer:.4f})")


def compute_perplexity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    device: str = "cpu"
) -> PerplexityResult:
    """
    Compute perplexity on a sequence.

    Perplexity = exp(cross_entropy_loss)
    Lower is better (perfect model has perplexity = 1)

    Args:
        model: The language model
        input_ids: Token IDs [batch_size, seq_len]
        attention_mask: Optional attention mask [batch_size, seq_len]
        device: Device to run computation on

    Returns:
        PerplexityResult with perplexity, cross-entropy, and other metrics

    Example:
        >>> model = ContinualLLM(config)
        >>> input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        >>> result = compute_perplexity(model, input_ids)
        >>> print(f"Perplexity: {result.perplexity:.2f}")
    """
    model.eval()
    model = model.to(device)
    input_ids = input_ids.to(device)

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        # Get model outputs
        if hasattr(model, 'forward'):
            # For our ContinualLLM
            logits, _, _ = model(input_ids)
        else:
            # For generic models
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Apply attention mask if provided
        if attention_mask is not None:
            shift_mask = attention_mask[..., 1:].contiguous()
            # Only compute loss on non-masked tokens
            shift_logits = shift_logits[shift_mask.bool()]
            shift_labels = shift_labels[shift_mask.bool()]
            num_tokens = shift_mask.sum().item()
        else:
            num_tokens = shift_labels.numel()

        # Compute cross-entropy loss
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        cross_entropy = F.cross_entropy(
            shift_logits,
            shift_labels,
            reduction='mean'
        ).item()

        # Compute perplexity
        perplexity = math.exp(cross_entropy)

        # Compute bits per byte (useful for comparing tokenizers)
        bits_per_byte = cross_entropy / math.log(2)

    return PerplexityResult(
        perplexity=perplexity,
        cross_entropy=cross_entropy,
        num_tokens=num_tokens,
        bits_per_byte=bits_per_byte
    )


def compute_batch_perplexity(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    max_batches: Optional[int] = None
) -> PerplexityResult:
    """
    Compute perplexity over an entire dataset.

    Args:
        model: The language model
        dataloader: DataLoader providing batches of (input_ids, attention_mask)
        device: Device to run computation on
        max_batches: Optional limit on number of batches to evaluate

    Returns:
        PerplexityResult averaged over all batches
    """
    model.eval()
    model = model.to(device)

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(device) if isinstance(batch, dict) else batch[0].to(device)
            attention_mask = batch.get('attention_mask') if isinstance(batch, dict) else None

            # Compute loss for this batch
            if hasattr(model, 'forward'):
                logits, _, _ = model(input_ids)
            else:
                outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                shift_mask = attention_mask[..., 1:].contiguous()
                num_tokens = shift_mask.sum().item()

                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                shift_mask = shift_mask.view(-1)

                loss = F.cross_entropy(
                    shift_logits,
                    shift_labels,
                    reduction='none'
                )
                loss = (loss * shift_mask).sum() / shift_mask.sum()
            else:
                num_tokens = shift_labels.numel()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='mean'
                )

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1

    # Compute average metrics
    avg_cross_entropy = total_loss / total_tokens
    perplexity = math.exp(avg_cross_entropy)
    bits_per_byte = avg_cross_entropy / math.log(2)

    return PerplexityResult(
        perplexity=perplexity,
        cross_entropy=avg_cross_entropy,
        num_tokens=total_tokens,
        bits_per_byte=bits_per_byte
    )


def compute_forgetting_rate(
    current_performances: Dict[str, float],
    previous_performances: Dict[str, float]
) -> ForgettingMetrics:
    """
    Compute catastrophic forgetting metrics.

    Measures how much performance on old tasks degrades after learning new tasks.

    Args:
        current_performances: Dictionary mapping task names to current performance
        previous_performances: Dictionary mapping task names to previous performance

    Returns:
        ForgettingMetrics containing forgetting rate and transfer metrics

    Example:
        >>> # After learning Task A
        >>> perf_after_A = {'task_A': 0.95}
        >>>
        >>> # After learning Task B
        >>> perf_after_B = {'task_A': 0.90, 'task_B': 0.92}
        >>>
        >>> metrics = compute_forgetting_rate(perf_after_B, perf_after_A)
        >>> print(f"Forgetting rate: {metrics.forgetting_rate:.2%}")
    """
    # Find common tasks (old tasks)
    common_tasks = set(current_performances.keys()) & set(previous_performances.keys())

    if not common_tasks:
        # No old tasks - compute forward transfer from all current tasks (which are all new)
        forward_transfer = np.mean(list(current_performances.values())) if current_performances else 0.0
        return ForgettingMetrics(
            forgetting_rate=0.0,
            backward_transfer=0.0,
            forward_transfer=forward_transfer,
            task_performances=current_performances
        )

    # Compute forgetting for each old task
    forgetting_per_task = {}
    for task in common_tasks:
        prev_perf = previous_performances[task]
        curr_perf = current_performances[task]
        forgetting = max(0, prev_perf - curr_perf)  # Only count degradation
        forgetting_per_task[task] = forgetting

    # Average forgetting rate
    forgetting_rate = np.mean(list(forgetting_per_task.values()))

    # Backward transfer: average change in performance on old tasks
    # Positive means improvement, negative means forgetting
    backward_transfer = np.mean([
        current_performances[task] - previous_performances[task]
        for task in common_tasks
    ])

    # Forward transfer: performance on new tasks relative to baseline
    # (Would need baseline performance for proper computation)
    new_tasks = set(current_performances.keys()) - set(previous_performances.keys())
    if new_tasks:
        forward_transfer = np.mean([current_performances[task] for task in new_tasks])
    else:
        forward_transfer = 0.0

    return ForgettingMetrics(
        forgetting_rate=forgetting_rate,
        backward_transfer=backward_transfer,
        forward_transfer=forward_transfer,
        task_performances=current_performances
    )


def compute_accuracy(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cpu"
) -> float:
    """
    Compute next-token prediction accuracy.

    Args:
        model: The language model
        input_ids: Input token IDs [batch_size, seq_len]
        labels: Target token IDs [batch_size, seq_len]
        device: Device to run computation on

    Returns:
        Accuracy as a float between 0 and 1
    """
    model.eval()
    model = model.to(device)
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        if hasattr(model, 'forward'):
            logits, _, _ = model(input_ids)
        else:
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # Get predictions
        predictions = logits.argmax(dim=-1)

        # Shift for next-token prediction
        shift_predictions = predictions[..., :-1].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute accuracy
        correct = (shift_predictions == shift_labels).sum().item()
        total = shift_labels.numel()
        accuracy = correct / total

    return accuracy


def evaluate_generation_quality(
    generated_text: str,
    reference_text: Optional[str] = None,
    metrics: List[str] = ['length', 'unique_tokens', 'repetition']
) -> Dict[str, float]:
    """
    Evaluate quality of generated text.

    Args:
        generated_text: The generated text to evaluate
        reference_text: Optional reference text for comparison
        metrics: List of metrics to compute

    Returns:
        Dictionary mapping metric names to values

    Available metrics:
        - 'length': Number of tokens
        - 'unique_tokens': Ratio of unique tokens to total tokens
        - 'repetition': Measure of repetitiveness (lower is better)
        - 'bleu': BLEU score if reference_text provided
    """
    results = {}

    # Tokenize (simple whitespace tokenization for now)
    tokens = generated_text.split()

    if 'length' in metrics:
        results['length'] = len(tokens)

    if 'unique_tokens' in metrics:
        if len(tokens) > 0:
            results['unique_tokens'] = len(set(tokens)) / len(tokens)
        else:
            results['unique_tokens'] = 0.0

    if 'repetition' in metrics:
        # Measure repetition using bigram diversity
        if len(tokens) > 1:
            bigrams = list(zip(tokens[:-1], tokens[1:]))
            if len(bigrams) > 0:
                unique_bigrams = len(set(bigrams))
                results['repetition'] = 1.0 - (unique_bigrams / len(bigrams))
            else:
                results['repetition'] = 0.0
        else:
            results['repetition'] = 0.0

    if 'bleu' in metrics and reference_text is not None:
        # Simple BLEU-1 score (would need nltk for proper BLEU)
        ref_tokens = set(reference_text.split())
        gen_tokens = set(tokens)
        if len(gen_tokens) > 0:
            precision = len(ref_tokens & gen_tokens) / len(gen_tokens)
            results['bleu'] = precision
        else:
            results['bleu'] = 0.0

    return results


def compute_domain_specific_accuracy(
    model: torch.nn.Module,
    test_cases: List[Tuple[str, str]],
    tokenizer: Any,
    device: str = "cpu",
    max_length: int = 100
) -> Dict[str, Any]:
    """
    Evaluate model on domain-specific test cases.

    Useful for evaluating math, code, or other structured tasks.

    Args:
        model: The language model
        test_cases: List of (prompt, expected_answer) tuples
        tokenizer: Tokenizer for encoding text
        device: Device to run computation on
        max_length: Maximum generation length

    Returns:
        Dictionary with accuracy and per-case results

    Example:
        >>> test_cases = [
        ...     ("2 + 2 =", "4"),
        ...     ("3 * 5 =", "15"),
        ... ]
        >>> results = compute_domain_specific_accuracy(model, test_cases, tokenizer)
        >>> print(f"Math accuracy: {results['accuracy']:.2%}")
    """
    model.eval()
    model = model.to(device)

    correct = 0
    total = len(test_cases)
    results_per_case = []

    for prompt, expected in test_cases:
        # Encode prompt
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)

        # Generate
        with torch.no_grad():
            # Simple greedy generation
            for _ in range(max_length):
                logits, _, _ = model(input_ids)
                next_token_logits = logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

                # Stop at EOS or newline
                if next_token.item() in [tokenizer.eos_token_id, tokenizer.encode('\n')[0]]:
                    break

                input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Decode generated text
        generated = tokenizer.decode(input_ids[0, len(tokenizer.encode(prompt)):].tolist())

        # Check if expected answer is in generated text
        is_correct = expected.lower() in generated.lower()
        if is_correct:
            correct += 1

        results_per_case.append({
            'prompt': prompt,
            'expected': expected,
            'generated': generated,
            'correct': is_correct
        })

    accuracy = correct / total if total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results_per_case
    }


def compute_calibration_error(
    predicted_probs: torch.Tensor,
    true_labels: torch.Tensor,
    num_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well the model's confidence matches its actual accuracy.
    Lower is better (0 = perfectly calibrated).

    Args:
        predicted_probs: Predicted probabilities [N, num_classes]
        true_labels: True labels [N]
        num_bins: Number of bins for calibration

    Returns:
        Expected Calibration Error (ECE)
    """
    confidences, predictions = predicted_probs.max(dim=-1)
    accuracies = predictions.eq(true_labels)

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)

    for i in range(num_bins):
        # Find samples in this confidence bin
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)

        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


# Convenience functions for common evaluation scenarios

def quick_eval(
    model: torch.nn.Module,
    text: str,
    tokenizer: Any,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Quick evaluation of a model on a text sample.

    Returns perplexity, accuracy, and generation quality.
    """
    # Encode text
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens]).to(device)
    labels = input_ids.clone()

    # Compute metrics
    ppl_result = compute_perplexity(model, input_ids, device=device)
    acc = compute_accuracy(model, input_ids, labels, device=device)

    # Generate continuation
    gen_input = input_ids[:, :min(10, len(tokens))]  # Use first 10 tokens as prompt
    with torch.no_grad():
        for _ in range(50):
            logits, _, _ = model(gen_input)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_input = torch.cat([gen_input, next_token], dim=-1)

    generated = tokenizer.decode(gen_input[0].tolist())
    gen_quality = evaluate_generation_quality(generated)

    return {
        'perplexity': ppl_result.perplexity,
        'cross_entropy': ppl_result.cross_entropy,
        'accuracy': acc,
        'generation_quality': gen_quality,
        'generated_sample': generated[:200]  # First 200 chars
    }


if __name__ == "__main__":
    # Example usage
    print("Testing evaluation metrics...")

    # Test perplexity
    dummy_logits = torch.randn(1, 10, 1000)  # [batch, seq_len, vocab]
    dummy_labels = torch.randint(0, 1000, (1, 10))

    print("\n✓ Metrics module ready!")
    print("  - Perplexity computation")
    print("  - Forgetting rate measurement")
    print("  - Accuracy computation")
    print("  - Generation quality evaluation")
    print("  - Domain-specific testing")
    print("  - Calibration error")
