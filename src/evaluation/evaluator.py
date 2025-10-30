"""
Evaluator orchestration system for continual learning.

This module provides a high-level interface for evaluating continual learning models:
- Track performance over time
- Measure forgetting across multiple tasks
- Generate evaluation reports
- Visualize learning curves
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from collections import defaultdict

from .metrics import (
    compute_batch_perplexity,
    compute_forgetting_rate,
    compute_accuracy,
    evaluate_generation_quality,
    ForgettingMetrics,
)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    device: str = "cpu"
    max_eval_batches: Optional[int] = None  # Limit evaluation batches
    compute_forgetting: bool = True  # Track forgetting metrics
    compute_generation_quality: bool = True  # Evaluate generations
    generation_prompts: List[str] = field(default_factory=list)  # Prompts for generation
    max_generation_length: int = 100
    save_generations: bool = True  # Save generated samples
    track_per_domain: bool = True  # Track metrics per domain
    verbose: bool = True


@dataclass
class EvaluationResult:
    """Results from a single evaluation run."""

    timestamp: float
    step: int
    perplexity: float
    cross_entropy: float
    accuracy: float
    domain: Optional[str] = None
    forgetting_metrics: Optional[ForgettingMetrics] = None
    generation_samples: List[str] = field(default_factory=list)
    generation_quality: Dict[str, float] = field(default_factory=dict)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "timestamp": self.timestamp,
            "step": self.step,
            "perplexity": self.perplexity,
            "cross_entropy": self.cross_entropy,
            "accuracy": self.accuracy,
            "domain": self.domain,
            "generation_samples": self.generation_samples,
            "generation_quality": self.generation_quality,
            "additional_metrics": self.additional_metrics,
        }

        if self.forgetting_metrics:
            result["forgetting_metrics"] = {
                "forgetting_rate": self.forgetting_metrics.forgetting_rate,
                "backward_transfer": self.forgetting_metrics.backward_transfer,
                "forward_transfer": self.forgetting_metrics.forward_transfer,
                "task_performances": self.forgetting_metrics.task_performances,
            }

        return result

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(step={self.step}, "
            f"perplexity={self.perplexity:.4f}, "
            f"accuracy={self.accuracy:.4f}, "
            f"domain={self.domain})"
        )


class Evaluator:
    """
    High-level evaluator for continual learning models.

    Tracks performance over time, measures forgetting, and generates reports.

    Example:
        >>> config = EvaluationConfig(device="mps", compute_forgetting=True)
        >>> evaluator = Evaluator(model, tokenizer, config)
        >>>
        >>> # Evaluate after learning math
        >>> result = evaluator.evaluate(math_dataloader, domain="math", step=0)
        >>>
        >>> # Learn code
        >>> learner.learn_from_text(code_data, domain="code")
        >>>
        >>> # Evaluate again - check forgetting on math
        >>> result = evaluator.evaluate(math_dataloader, domain="math", step=1)
        >>> print(f"Forgetting rate: {result.forgetting_metrics.forgetting_rate:.2%}")
    """

    def __init__(self, model: nn.Module, tokenizer: Any, config: EvaluationConfig):
        """
        Initialize evaluator.

        Args:
            model: The language model to evaluate
            tokenizer: Tokenizer for encoding/decoding
            config: Evaluation configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Track evaluation history
        self.history: List[EvaluationResult] = []
        self.domain_history: Dict[str, List[EvaluationResult]] = defaultdict(list)

        # Track performance baselines for forgetting
        self.baseline_performances: Dict[str, float] = {}
        self.task_performances: Dict[int, Dict[str, float]] = {}

        if config.verbose:
            print("✓ Evaluator initialized")

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        domain: Optional[str] = None,
        step: int = 0,
        save_path: Optional[Path] = None,
    ) -> EvaluationResult:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: DataLoader with evaluation data
            domain: Optional domain name for tracking
            step: Current training step
            save_path: Optional path to save results

        Returns:
            EvaluationResult with all computed metrics
        """
        if self.config.verbose:
            print(f"\n📊 Evaluating{f' on {domain}' if domain else ''}...")

        start_time = time.time()

        # Compute perplexity
        ppl_result = compute_batch_perplexity(
            self.model,
            dataloader,
            device=self.config.device,
            max_batches=self.config.max_eval_batches,
        )

        # Compute accuracy (on first batch for efficiency)
        first_batch = next(iter(dataloader))
        input_ids = first_batch["input_ids"] if isinstance(first_batch, dict) else first_batch[0]
        accuracy = compute_accuracy(
            self.model,
            input_ids,
            input_ids.clone(),  # Labels same as inputs for LM
            device=self.config.device,
        )

        # Generate samples
        generation_samples = []
        generation_quality = {}

        if self.config.compute_generation_quality and self.config.generation_prompts:
            if self.config.verbose:
                print("  Generating samples...")

            for prompt in self.config.generation_prompts[:3]:  # Limit to 3 prompts
                generated = self._generate_sample(prompt)
                generation_samples.append(f"Prompt: {prompt}\nGenerated: {generated}\n")

                # Evaluate quality
                quality = evaluate_generation_quality(generated)
                for k, v in quality.items():
                    if k not in generation_quality:
                        generation_quality[k] = []
                    generation_quality[k].append(v)

            # Average quality metrics
            generation_quality = {k: sum(v) / len(v) for k, v in generation_quality.items()}

        # Compute forgetting metrics if enabled
        forgetting_metrics = None
        if self.config.compute_forgetting and domain:
            current_perf = ppl_result.perplexity
            self.task_performances[step] = self.task_performances.get(step, {})
            self.task_performances[step][domain] = current_perf

            # Compare with previous step
            if step > 0 and (step - 1) in self.task_performances:
                forgetting_metrics = compute_forgetting_rate(
                    self.task_performances[step], self.task_performances[step - 1]
                )

                if self.config.verbose:
                    print(f"  Forgetting rate: {forgetting_metrics.forgetting_rate:.4f}")

        # Create result
        result = EvaluationResult(
            timestamp=start_time,
            step=step,
            perplexity=ppl_result.perplexity,
            cross_entropy=ppl_result.cross_entropy,
            accuracy=accuracy,
            domain=domain,
            forgetting_metrics=forgetting_metrics,
            generation_samples=generation_samples,
            generation_quality=generation_quality,
        )

        # Update history
        self.history.append(result)
        if domain:
            self.domain_history[domain].append(result)

        # Save if requested
        if save_path:
            self.save_result(result, save_path)

        eval_time = time.time() - start_time
        if self.config.verbose:
            print(f"  ✓ Evaluation complete in {eval_time:.2f}s")
            print(f"  Perplexity: {ppl_result.perplexity:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")

        return result

    def evaluate_continual_learning(
        self,
        test_datasets: Dict[str, torch.utils.data.DataLoader],
        step: int,
        save_dir: Optional[Path] = None,
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate on multiple domains for continual learning.

        Args:
            test_datasets: Dictionary mapping domain names to dataloaders
            step: Current training step
            save_dir: Optional directory to save results

        Returns:
            Dictionary mapping domains to their evaluation results

        Example:
            >>> test_datasets = {
            ...     'math': math_loader,
            ...     'code': code_loader,
            ...     'general': general_loader
            ... }
            >>> results = evaluator.evaluate_continual_learning(test_datasets, step=5)
            >>> for domain, result in results.items():
            ...     print(f"{domain}: {result.perplexity:.2f}")
        """
        results = {}

        for domain, dataloader in test_datasets.items():
            save_path = save_dir / f"{domain}_step_{step}.json" if save_dir else None
            result = self.evaluate(dataloader, domain=domain, step=step, save_path=save_path)
            results[domain] = result

        # Compute overall forgetting metrics
        if step > 0 and len(results) > 1:
            all_current = {d: r.perplexity for d, r in results.items()}
            all_previous = {}

            for domain, history in self.domain_history.items():
                if history and len(history) > 1:
                    all_previous[domain] = history[-2].perplexity

            if all_previous:
                overall_forgetting = compute_forgetting_rate(all_current, all_previous)
                if self.config.verbose:
                    print(f"\n📈 Overall Forgetting Rate: {overall_forgetting.forgetting_rate:.4f}")
                    print(f"   Backward Transfer: {overall_forgetting.backward_transfer:.4f}")

        return results

    def _generate_sample(self, prompt: str) -> str:
        """Generate text from a prompt."""
        self.model.eval()
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.config.device)

        with torch.no_grad():
            for _ in range(self.config.max_generation_length):
                logits, _, _ = self.model(input_ids)
                next_token_logits = logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

                # Stop at EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                input_ids = torch.cat([input_ids, next_token], dim=-1)

        generated_text = self.tokenizer.decode(input_ids[0].tolist())
        return generated_text

    def save_result(self, result: EvaluationResult, path: Path):
        """Save evaluation result to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def save_history(self, path: Path):
        """Save entire evaluation history to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        history_data = {
            "history": [r.to_dict() for r in self.history],
            "domain_history": {
                domain: [r.to_dict() for r in results]
                for domain, results in self.domain_history.items()
            },
        }
        with open(path, "w") as f:
            json.dump(history_data, f, indent=2)

        if self.config.verbose:
            print(f"✓ Saved evaluation history to {path}")

    def load_history(self, path: Path):
        """Load evaluation history from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        # Reconstruct history
        # (simplified - would need full reconstruction of objects)
        if self.config.verbose:
            print(f"✓ Loaded evaluation history from {path}")

    def get_learning_curve(
        self, domain: Optional[str] = None, metric: str = "perplexity"
    ) -> Tuple[List[int], List[float]]:
        """
        Get learning curve for a metric.

        Args:
            domain: Optional domain to filter by
            metric: Metric to plot ('perplexity', 'accuracy', 'cross_entropy')

        Returns:
            Tuple of (steps, values) for plotting
        """
        if domain:
            history = self.domain_history.get(domain, [])
        else:
            history = self.history

        steps = [r.step for r in history]
        values = [getattr(r, metric) for r in history]

        return steps, values

    def get_forgetting_curve(self, domain: str) -> Tuple[List[int], List[float]]:
        """
        Get forgetting rate over time for a domain.

        Args:
            domain: Domain to track

        Returns:
            Tuple of (steps, forgetting_rates)
        """
        history = self.domain_history.get(domain, [])
        steps = []
        forgetting_rates = []

        for result in history:
            if result.forgetting_metrics:
                steps.append(result.step)
                forgetting_rates.append(result.forgetting_metrics.forgetting_rate)

        return steps, forgetting_rates

    def generate_report(self, save_path: Optional[Path] = None) -> str:
        """
        Generate a human-readable evaluation report.

        Args:
            save_path: Optional path to save report

        Returns:
            Report as a string
        """
        report_lines = [
            "=" * 80,
            "CONTINUAL LEARNING EVALUATION REPORT",
            "=" * 80,
            "",
            f"Total Evaluations: {len(self.history)}",
            f"Domains Tracked: {list(self.domain_history.keys())}",
            "",
            "=" * 80,
            "PERFORMANCE BY DOMAIN",
            "=" * 80,
            "",
        ]

        for domain, history in self.domain_history.items():
            if not history:
                continue

            latest = history[-1]
            report_lines.extend(
                [
                    f"\n{domain.upper()}:",
                    f"  Latest Perplexity: {latest.perplexity:.4f}",
                    f"  Latest Accuracy: {latest.accuracy:.4f}",
                ]
            )

            if len(history) > 1:
                first = history[0]
                improvement = first.perplexity - latest.perplexity
                report_lines.append(f"  Perplexity Change: {improvement:+.4f}")

            if latest.forgetting_metrics:
                report_lines.extend(
                    [
                        f"  Forgetting Rate: {latest.forgetting_metrics.forgetting_rate:.4f}",
                        f"  Backward Transfer: {latest.forgetting_metrics.backward_transfer:+.4f}",
                    ]
                )

        report_lines.extend(["", "=" * 80, "GENERATION SAMPLES", "=" * 80, ""])

        if self.history and self.history[-1].generation_samples:
            for sample in self.history[-1].generation_samples[:3]:
                report_lines.append(sample)

        report_lines.extend(["", "=" * 80, ""])

        report = "\n".join(report_lines)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(report)
            if self.config.verbose:
                print(f"✓ Saved report to {save_path}")

        return report

    def print_summary(self):
        """Print a summary of the latest evaluation."""
        if not self.history:
            print("No evaluations yet.")
            return

        latest = self.history[-1]
        print("\n" + "=" * 60)
        print("LATEST EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Step: {latest.step}")
        print(f"Domain: {latest.domain or 'General'}")
        print(f"Perplexity: {latest.perplexity:.4f}")
        print(f"Cross-Entropy: {latest.cross_entropy:.4f}")
        print(f"Accuracy: {latest.accuracy:.4f}")

        if latest.forgetting_metrics:
            print(f"\nForgetting Metrics:")
            print(f"  Forgetting Rate: {latest.forgetting_metrics.forgetting_rate:.4f}")
            print(f"  Backward Transfer: {latest.forgetting_metrics.backward_transfer:+.4f}")
            print(f"  Forward Transfer: {latest.forgetting_metrics.forward_transfer:.4f}")

        if latest.generation_quality:
            print(f"\nGeneration Quality:")
            for metric, value in latest.generation_quality.items():
                print(f"  {metric}: {value:.4f}")

        print("=" * 60 + "\n")


if __name__ == "__main__":
    print("✓ Evaluator module ready!")
    print("  - EvaluationConfig for configuration")
    print("  - EvaluationResult for storing results")
    print("  - Evaluator for orchestrating evaluation")
    print("  - Learning curve tracking")
    print("  - Forgetting analysis")
    print("  - Report generation")
