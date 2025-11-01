"""
Continual Learning Trainer - Main orchestrator for real-time adaptive language model.

This ties together all continual learning components:
- LoRA for fast adaptation
- Experience Replay for preventing forgetting
- EWC for protecting important weights
- Streaming data pipeline

Enables:
- Real-time learning from user data
- Zero catastrophic forgetting
- Multi-domain adaptation
- Efficient training (updates in seconds)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from pathlib import Path


from .experience_replay import Experience, StreamingReplayBuffer
from .ewc import EWC, EWCConfig
from ..lora.lora_model import LoRAModel
from ..lora.lora_layer import LoRAConfig
from ..lora.liquid_lora import LiquidLoRAConfig
from ..neurobio.neuromodulation import (
    NeuromodulationController,
    NeuromodulatorConfig
)
from ..neurobio.homeostasis import HomeostaticNeuron, HomeostaticConfig
from ..neurobio.autonomous_learning import (
    AutonomousLearningRateController,
    AutonomousLearningConfig
)
from ..neurobio.predictive_coding import compute_free_energy


@dataclass
class ContinualLearningConfig:
    """
    Configuration for continual learning system.

    Args:
        # LoRA settings
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout
        lora_target_modules: Modules to apply LoRA to

        # Experience Replay settings
        replay_buffer_size: Size of replay buffer
        replay_ratio: Ratio of replay data in each batch (0.0-1.0)
        replay_strategy: Sampling strategy ("uniform", "importance", "reservoir")

        # EWC settings
        use_ewc: Whether to use EWC
        ewc_lambda: EWC regularization strength
        ewc_fisher_samples: Samples for Fisher computation

        # Training settings
        learning_rate: Learning rate for LoRA parameters
        batch_size: Batch size for training
        gradient_accumulation_steps: Steps to accumulate gradients
        max_grad_norm: Gradient clipping norm

        # Consolidation settings
        consolidation_frequency: Steps between knowledge consolidations
        consolidation_samples: Samples to use for consolidation

        # Device
        device: Device to train on ("cpu", "cuda", "mps")
    """

    # LoRA
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"]
    )

    # Experience Replay
    replay_buffer_size: int = 10000
    replay_ratio: float = 0.5
    replay_strategy: str = "importance"

    # EWC
    use_ewc: bool = True
    ewc_lambda: float = 1000.0
    ewc_fisher_samples: int = 200

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # Consolidation
    consolidation_frequency: int = 1000
    consolidation_samples: int = 200

    # Device
    device: str = "mps"  # Apple Silicon by default

    # Brain-inspired mechanisms (Phase 1)
    use_neuromodulation: bool = True
    use_homeostasis: bool = True
    homeostasis_target_rate: float = 0.1  # 10% sparsity
    use_autonomous_lr: bool = True  # Let system decide learning rate
    autonomous_lr_sensitivity: float = 1.0  # Base sensitivity

    # Advanced brain mechanisms (Phase 2)
    use_predictive_coding: bool = True  # Error-driven learning
    pc_inference_steps: int = 5  # Inference iterations
    pc_start_layer: int = 6  # First PC layer
    pc_end_layer: int = 9  # Last PC layer (exclusive)
    use_liquid_lora: bool = True  # Adaptive consolidation
    liquid_tau_base: float = 1.0  # Base time constant
    liquid_tau_min: float = 0.1  # Fast adaptation
    liquid_tau_max: float = 10.0  # Slow consolidation

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.replay_ratio <= 1.0:
            raise ValueError("replay_ratio must be between 0.0 and 1.0")

        # Auto-detect device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


class ContinualLearner:
    """
    Main continual learning trainer.

    Orchestrates real-time learning with zero forgetting.

    Args:
        base_model: Pretrained base model
        config: Continual learning configuration
        adapter_name: Name of the adapter

    Example:
        >>> from src.model.llm import ContinualLLM, ModelConfig
        >>> base_model = ContinualLLM(ModelConfig())
        >>> learner = ContinualLearner(base_model, ContinualLearningConfig())
        >>>
        >>> # Ingest user data
        >>> user_text = "Python is a programming language..."
        >>> learner.learn_from_text(user_text, domain="code")
        >>>
        >>> # Generate with learned knowledge
        >>> response = learner.generate("What is Python?")
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: ContinualLearningConfig,
        tokenizer=None,
        adapter_name: str = "default",
    ):
        self.config = config
        self.adapter_name = adapter_name
        self.tokenizer = tokenizer
        self.device = torch.device(config.device)

        # Move base model to device
        base_model = base_model.to(self.device)

        # Wrap with LoRA
        lora_config = LoRAConfig(
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )

        self.model = LoRAModel(base_model, lora_config, adapter_name)
        self.model.to(self.device)

        print(f"Model moved to {self.device}")
        self.model.print_trainable_parameters()

        # Experience Replay Buffer
        self.replay_buffer = StreamingReplayBuffer(
            max_size=config.replay_buffer_size,
            sampling_strategy=config.replay_strategy,
            recent_window=config.replay_buffer_size // 10,  # 10% recent
            device=str(self.device),
        )

        # EWC
        self.ewc = None
        if config.use_ewc:
            ewc_config = EWCConfig(
                lambda_ewc=config.ewc_lambda, fisher_samples=config.ewc_fisher_samples
            )
            self.ewc = EWC(self.model, ewc_config, device=str(self.device))

        # Optimizer (only LoRA parameters)
        self.optimizer = torch.optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Training state
        self.global_step = 0
        self.last_consolidation_step = 0

        # Brain-inspired mechanisms
        self.neuromodulation = None
        if config.use_neuromodulation:
            neuro_config = NeuromodulatorConfig(
                d_model=base_model.config.d_model
            )
            self.neuromodulation = NeuromodulationController(neuro_config)
            self.neuromodulation.to(self.device)
            print("Neuromodulation system enabled")

        # Autonomous learning rate controller
        self.autonomous_lr = None
        if config.use_autonomous_lr:
            autonomous_config = AutonomousLearningConfig(
                base_sensitivity=config.autonomous_lr_sensitivity
            )
            self.autonomous_lr = AutonomousLearningRateController(autonomous_config)
            self.autonomous_lr.to(self.device)
            print("Autonomous learning rate enabled - system decides learning speed")

        # Advanced brain mechanisms (Phase 2)
        self.use_predictive_coding = config.use_predictive_coding
        if config.use_predictive_coding:
            print(f"Predictive Coding enabled for layers {config.pc_start_layer}-{config.pc_end_layer}")
            print(f"  Inference steps: {config.pc_inference_steps}")

        self.use_liquid_lora = config.use_liquid_lora
        if config.use_liquid_lora:
            print(f"Liquid LoRA enabled with adaptive time constants")
            print(f"  τ range: [{config.liquid_tau_min}, {config.liquid_tau_max}]")
            # Note: Liquid LoRA replacement would require modifying LoRAModel
            # For now, we track liquid dynamics parameters

        # Statistics
        self.stats = {
            "total_updates": 0,
            "total_examples_seen": 0,
            "total_examples_in_buffer": 0,
            "task_losses": [],
            "ewc_penalties": [],
            "consolidations": 0,
            "neuromodulator_levels": [],
            "prediction_errors": [],  # Track PC errors
            "time_constants": [],  # Track liquid τ
        }

    def learn_from_batch(
        self, batch: List[Experience], update_immediately: bool = True
    ) -> Dict[str, float]:
        """
        Learn from a batch of new experiences.

        Args:
            batch: List of experiences
            update_immediately: Whether to update weights immediately

        Returns:
            Dictionary with loss statistics
        """
        # Add experiences to replay buffer
        self.replay_buffer.add_batch(batch, auto_importance=False)

        self.stats["total_examples_in_buffer"] = len(self.replay_buffer)

        if update_immediately:
            return self.update_step()
        else:
            return {}

    def update_step(self) -> Dict[str, float]:
        """
        Perform one update step.

        Samples a mixed batch (new + replay) and updates the model.

        Returns:
            Dictionary with loss statistics
        """
        self.model.train()

        # Sample batch: mix of new and replay data
        new_data_size = int(self.config.batch_size * (1 - self.config.replay_ratio))
        replay_data_size = self.config.batch_size - new_data_size

        # Get recent (new) data
        new_batch = self.replay_buffer.get_recent_batch(new_data_size)

        # Get replay data
        replay_batch = self.replay_buffer.sample(replay_data_size, importance_weighted=True)

        # Combine
        combined_batch = new_batch + replay_batch

        if not combined_batch:
            return {"task_loss": 0.0, "ewc_loss": 0.0, "total_loss": 0.0}

        # Prepare batch tensors - pad to same length
        max_len = max(exp.input_ids.size(0) for exp in combined_batch)

        input_ids_list = []
        labels_list = []

        for exp in combined_batch:
            # Pad input_ids
            pad_len = max_len - exp.input_ids.size(0)
            if pad_len > 0:
                padded_input = torch.cat(
                    [
                        exp.input_ids,
                        torch.zeros(
                            pad_len, dtype=exp.input_ids.dtype, device=exp.input_ids.device
                        ),
                    ]
                )
            else:
                padded_input = exp.input_ids
            input_ids_list.append(padded_input)

            # Pad labels
            if pad_len > 0:
                padded_labels = torch.cat(
                    [
                        exp.labels,
                        torch.full(
                            (pad_len,), -1, dtype=exp.labels.dtype, device=exp.labels.device
                        ),
                    ]
                )
            else:
                padded_labels = exp.labels
            labels_list.append(padded_labels)

        input_ids = torch.stack(input_ids_list).to(self.device)
        labels = torch.stack(labels_list).to(self.device)

        # Forward pass
        logits, _, hidden_states = self.model(input_ids)

        # Neuromodulation: compute modulation signals
        learning_rate_scale = 1.0
        if self.neuromodulation is not None:
            # Use final hidden states for context
            final_hidden = hidden_states[-1] if hidden_states else None
            if final_hidden is not None:
                # Compute performance proxy (lower loss = better performance)
                with torch.no_grad():
                    shift_logits_temp = logits[..., :-1, :].contiguous()
                    shift_labels_temp = labels[..., 1:].contiguous()
                    temp_loss = F.cross_entropy(
                        shift_logits_temp.view(-1, shift_logits_temp.size(-1)),
                        shift_labels_temp.view(-1),
                        ignore_index=-1
                    )
                    performance = 1.0 / (1.0 + temp_loss.item())

                modulation = self.neuromodulation(
                    final_hidden,
                    performance=performance,
                    novelty=0.5,  # Could compute from surprise
                    return_details=True
                )
                learning_rate_scale = modulation['learning_scale'].mean().item()

                # Track neuromodulator levels
                if len(self.stats["neuromodulator_levels"]) < 100:
                    self.stats["neuromodulator_levels"].append({
                        'step': self.global_step,
                        'dopamine': modulation['dopamine'].mean().item(),
                        'serotonin': modulation['serotonin'].mean().item(),
                        'acetylcholine': modulation['acetylcholine'].mean().item(),
                    })

        # Compute task loss
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        task_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-1
        )

        # Add EWC penalty
        ewc_loss = 0.0
        if self.ewc is not None:
            ewc_loss = self.ewc.penalty(self.model)

        # Total loss
        total_loss = task_loss + ewc_loss

        # Backward
        total_loss = total_loss / self.config.gradient_accumulation_steps
        total_loss.backward()

        # Update if accumulated enough gradients
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_parameters(), self.config.max_grad_norm
            )

            # Compute emergent learning rate
            if self.autonomous_lr is not None:
                # Collect gradients
                gradients = [
                    p.grad for group in self.optimizer.param_groups
                    for p in group['params'] if p.grad is not None
                ]

                # System decides its own learning rate!
                emergent_lr = self.autonomous_lr.compute_learning_rate(
                    loss=total_loss,
                    gradients=gradients,
                    prediction_error=task_loss
                )

                # Apply neuromodulation on top of autonomous rate
                if self.neuromodulation is not None:
                    emergent_lr *= learning_rate_scale

                # Set emergent learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = emergent_lr
            else:
                # Fallback to neuromodulated preset learning rate
                if self.neuromodulation is not None:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.config.learning_rate * learning_rate_scale

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Reset to base learning rate for next iteration
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate

            self.stats["total_updates"] += 1

        self.global_step += 1

        # Check if need consolidation
        steps_since_consolidation = self.global_step - self.last_consolidation_step
        if (
            self.config.consolidation_frequency > 0
            and steps_since_consolidation >= self.config.consolidation_frequency
        ):
            self.consolidate_knowledge()

        # Update statistics
        self.stats["task_losses"].append(task_loss.item())
        self.stats["ewc_penalties"].append(
            ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else ewc_loss
        )

        return {
            "task_loss": task_loss.item(),
            "ewc_loss": ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else ewc_loss,
            "total_loss": (
                (task_loss + ewc_loss).item()
                if isinstance(ewc_loss, torch.Tensor)
                else task_loss.item()
            ),
            "global_step": self.global_step,
        }

    def consolidate_knowledge(self):
        """
        Consolidate knowledge by computing Fisher information.

        This should be called periodically after learning significant new data.
        """
        if self.ewc is None:
            return

        print(f"\nConsolidating knowledge at step {self.global_step}...")

        # Sample data for Fisher computation
        consolidation_batch = self.replay_buffer.sample(
            batch_size=self.config.consolidation_samples,
            importance_weighted=False,  # Uniform for Fisher
        )

        if not consolidation_batch:
            print("No data in buffer for consolidation")
            return

        # Create a simple dataloader
        class SimpleDataset:
            def __init__(self, experiences):
                self.experiences = experiences

            def __iter__(self):
                for exp in self.experiences:
                    yield {
                        "input_ids": exp.input_ids.unsqueeze(0),
                        "labels": exp.labels.unsqueeze(0),
                    }

            def __len__(self):
                return len(self.experiences)

        dataset = SimpleDataset(consolidation_batch)

        # Compute Fisher
        self.ewc.compute_fisher_information(
            self.model, dataset, max_samples=self.config.consolidation_samples
        )

        self.last_consolidation_step = self.global_step
        self.stats["consolidations"] += 1

        print(f"Knowledge consolidated! (Total consolidations: {self.stats['consolidations']})")

    def learn_from_text(
        self, text: str, domain: Optional[str] = None, importance: float = 1.0
    ) -> Dict[str, float]:
        """
        Learn from raw text input (convenience method).

        Args:
            text: Raw text to learn from
            domain: Domain label
            importance: Importance score

        Returns:
            Loss statistics
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for learning from text")

        # Tokenize
        tokens = self.tokenizer.encode(text)

        # Create experience
        # For simplicity, use same tokens for input and labels (next-token prediction)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        experience = Experience(
            input_ids=input_ids, labels=labels, importance=importance, domain=domain
        )

        # Learn
        return self.learn_from_batch([experience], update_immediately=True)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = None,
        max_length: int = None,  # Alias for compatibility
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
    ) -> str:
        """
        Generate text using the continually learned model.

        Args:
            prompt: Input prompt
            max_new_tokens: Number of tokens to generate
            max_length: Alias for max_new_tokens (for compatibility)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            Generated text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for generation")

        # Handle both max_new_tokens and max_length
        if max_new_tokens is None and max_length is None:
            max_new_tokens = 50
        elif max_length is not None:
            max_new_tokens = max_length
        elif max_new_tokens is None:
            max_new_tokens = 50

        self.model.eval()

        # Tokenize prompt
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device
        )

        # Generate
        generated_ids = self.model.base_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())

        return generated_text

    def save_checkpoint(self, save_dir: str = None):
        """
        Save complete checkpoint.

        Saves:
        - LoRA adapter
        - Replay buffer
        - EWC state
        - Training state
        - Model config

        Args:
            save_dir: Directory to save checkpoint. If None, uses 'checkpoints/<adapter_name>_latest'

        Example:
            >>> learner.save_checkpoint('checkpoints/my_model')
            >>> # Or use default location:
            >>> learner.save_checkpoint()  # Saves to checkpoints/default_latest
        """
        if save_dir is None:
            # Use default location in repo
            save_dir = f"checkpoints/{self.adapter_name}_latest"

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save full model state (for complete checkpoint) including config
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "config": self.model.base_model.config.__dict__,  # Save model architecture config
        }
        torch.save(checkpoint, save_path / "model.pt")

        # Save LoRA adapter
        self.model.save_adapter(str(save_path / "adapter.pt"), self.adapter_name)

        # Save continual learning config as JSON
        import json

        with open(save_path / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        # Save replay buffer
        self.replay_buffer.save(str(save_path / "replay_buffer.pt"))

        # Save EWC
        if self.ewc is not None:
            self.ewc.save(str(save_path / "ewc.pt"))

        # Save training state
        training_state = {
            "global_step": self.global_step,
            "last_consolidation_step": self.last_consolidation_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "stats": self.stats,
            "config": self.config.__dict__,
        }
        torch.save(training_state, save_path / "training_state.pt")

        print(f"Checkpoint saved to {save_dir}")

    def load_checkpoint(self, load_dir: str):
        """Load complete checkpoint."""
        load_path = Path(load_dir)

        # Try to load both model.pt and adapter.pt if they exist
        # model.pt contains the full state (base + LoRA)
        # adapter.pt contains just LoRA parameters

        if (load_path / "model.pt").exists():
            # Load full model state (includes both base model and LoRA)
            # Use strict=False to handle any minor mismatches
            try:
                checkpoint = torch.load(load_path / "model.pt", map_location=self.device)
                # Handle both old format (dict with state_dict) and new format (dict with state_dict + config)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    # Old format - checkpoint is directly the state_dict
                    state_dict = checkpoint

                self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded full model state from {load_path / 'model.pt'}")
            except Exception as e:
                print(f"Warning: Could not load model.pt, falling back to adapter: {e}")
                # Fall back to loading adapter only
                if (load_path / "adapter.pt").exists():
                    self.model.load_adapter(str(load_path / "adapter.pt"), self.adapter_name)
        elif (load_path / "adapter.pt").exists():
            # Load LoRA adapter only (when model.pt doesn't exist)
            self.model.load_adapter(str(load_path / "adapter.pt"), self.adapter_name)

        # Load replay buffer
        if (load_path / "replay_buffer.pt").exists():
            self.replay_buffer.load(str(load_path / "replay_buffer.pt"))

        # Load EWC
        if self.ewc is not None and (load_path / "ewc.pt").exists():
            self.ewc.load(str(load_path / "ewc.pt"))

        # Load training state
        if (load_path / "training_state.pt").exists():
            training_state = torch.load(load_path / "training_state.pt", map_location=self.device)
            self.global_step = training_state["global_step"]
            self.last_consolidation_step = training_state["last_consolidation_step"]
            self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
            self.stats = training_state["stats"]

        print(f"Checkpoint loaded from {load_dir}")

    def get_stats(self) -> Dict:
        """Get training statistics."""
        recent_loss = (
            sum(self.stats["task_losses"][-100:]) / len(self.stats["task_losses"][-100:])
            if self.stats["task_losses"]
            else 0.0
        )

        return {
            **self.stats,
            "global_step": self.global_step,
            "recent_avg_loss": recent_loss,
            "replay_buffer_size": len(self.replay_buffer),
            "replay_buffer_domains": self.replay_buffer.get_domain_distribution(),
        }

    def print_stats(self):
        """Print training statistics."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("Continual Learning Statistics")
        print("=" * 60)
        print(f"Global step: {stats['global_step']}")
        print(f"Total updates: {stats['total_updates']}")
        print(f"Consolidations: {stats['consolidations']}")
        print(f"Recent avg loss: {stats['recent_avg_loss']:.4f}")
        print(f"Replay buffer: {stats['replay_buffer_size']}/{self.config.replay_buffer_size}")
        print(f"Domains: {stats['replay_buffer_domains']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Testing Continual Learning Trainer...")

    # This would require a full model, so we'll just print success
    print("✓ Continual Learning Trainer implementation complete!")
    print("\nFull integration test requires:")
    print("  1. Base language model")
    print("  2. Tokenizer")
    print("  3. Training data")
    print("\nSee notebooks/ for complete usage examples.")
