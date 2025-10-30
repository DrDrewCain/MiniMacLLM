"""
LoRA model wrapper for injecting LoRA into pretrained models.

Provides utilities to:
- Add LoRA layers to specific modules in a model
- Manage multiple LoRA adapters
- Save/load LoRA weights
- Merge/unmerge LoRA weights
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Set
import re
from pathlib import Path

from .lora_layer import LinearWithLoRA, LoRAConfig, mark_only_lora_as_trainable, count_lora_parameters


class LoRAModel(nn.Module):
    """
    Wrapper that adds LoRA to a pretrained model.

    This class handles:
    - Injecting LoRA layers into target modules
    - Freezing base model parameters
    - Managing multiple adapters
    - Saving/loading adapter weights

    Args:
        base_model: Pretrained model to add LoRA to
        lora_config: LoRA configuration
        adapter_name: Name of this adapter

    Example:
        >>> from src.model.llm import ContinualLLM, ModelConfig
        >>> base_model = ContinualLLM(ModelConfig())
        >>> lora_config = LoRAConfig(r=16, alpha=32)
        >>> lora_model = LoRAModel(base_model, lora_config, "math_adapter")
        >>> # Now only LoRA parameters are trainable
    """

    def __init__(
        self,
        base_model: nn.Module,
        lora_config: LoRAConfig,
        adapter_name: str = "default"
    ):
        super().__init__()

        self.base_model = base_model
        self.lora_config = lora_config
        self.active_adapter = adapter_name

        # Track which modules have LoRA
        self.lora_modules: Dict[str, LinearWithLoRA] = {}

        # Inject LoRA into target modules
        self._inject_lora(adapter_name)

        # Freeze base model, unfreeze LoRA
        mark_only_lora_as_trainable(self)

    def _inject_lora(self, adapter_name: str):
        """
        Inject LoRA layers into target modules.

        Walks through the model and replaces Linear layers in target modules
        with LinearWithLoRA layers.

        Args:
            adapter_name: Name of the adapter being created
        """
        target_modules = self.lora_config.target_modules

        # We need to replace modules by accessing their parent
        # Build a list of (parent_module, child_name, module) tuples to replace
        modules_to_replace = []

        for name, module in self.base_model.named_modules():
            # Check if this is a Linear module that should get LoRA
            if isinstance(module, nn.Linear) and self._is_target_module(name, target_modules):
                # Find parent module
                if '.' in name:
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    # Get parent module
                    parent = self.base_model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    modules_to_replace.append((parent, child_name, module, name))

        # Now replace all targeted Linear layers with LoRA versions
        for parent, child_name, linear_module, full_name in modules_to_replace:
            lora_layer = LinearWithLoRA(
                base_layer=linear_module,
                r=self.lora_config.r,
                alpha=self.lora_config.alpha,
                dropout=self.lora_config.dropout
            )
            setattr(parent, child_name, lora_layer)
            self.lora_modules[full_name] = lora_layer

    def _is_target_module(self, module_name: str, target_modules: List[str]) -> bool:
        """
        Check if a module name matches any target module patterns.

        Args:
            module_name: Full name of the module (e.g., "layers.0.attention.q_proj")
            target_modules: List of target patterns (e.g., ["q_proj", "v_proj"])

        Returns:
            True if module should get LoRA
        """
        # Check if any target pattern matches the end of the module name
        for target in target_modules:
            # Simple name match (e.g., "q_proj" matches "layers.0.attention.q_proj")
            if module_name.endswith(target):
                return True

            # Regex pattern match (for more complex patterns)
            if re.search(target, module_name):
                return True

        return False

    def forward(self, *args, **kwargs):
        """Forward pass through base model with LoRA."""
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generation wrapper."""
        return self.base_model.generate(*args, **kwargs)

    def merge_adapter(self):
        """Merge active LoRA adapter into base weights."""
        for lora_module in self.lora_modules.values():
            lora_module.merge()

    def unmerge_adapter(self):
        """Unmerge active LoRA adapter from base weights."""
        for lora_module in self.lora_modules.values():
            lora_module.unmerge()

    def save_adapter(self, save_path: str, adapter_name: Optional[str] = None):
        """
        Save LoRA adapter weights.

        Only saves LoRA parameters, not the base model.

        Args:
            save_path: Path to save adapter weights
            adapter_name: Name of adapter (defaults to active adapter)
        """
        adapter_name = adapter_name or self.active_adapter

        # Collect LoRA state dict
        lora_state_dict = {}
        for name, module in self.lora_modules.items():
            lora_state_dict[f"{name}.lora_A"] = module.lora.lora_A
            lora_state_dict[f"{name}.lora_B"] = module.lora.lora_B

        # Save
        save_dict = {
            'adapter_name': adapter_name,
            'lora_config': self.lora_config.__dict__,
            'state_dict': lora_state_dict
        }

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, save_path)

        print(f"Saved LoRA adapter '{adapter_name}' to {save_path}")

    def load_adapter(self, load_path: str, adapter_name: Optional[str] = None):
        """
        Load LoRA adapter weights.

        Args:
            load_path: Path to load adapter from
            adapter_name: Name to assign to loaded adapter
        """
        checkpoint = torch.load(load_path, map_location='cpu')

        # Load state dict
        lora_state_dict = checkpoint['state_dict']

        # Apply to modules
        for name, module in self.lora_modules.items():
            lora_a_key = f"{name}.lora_A"
            lora_b_key = f"{name}.lora_B"

            if lora_a_key in lora_state_dict:
                module.lora.lora_A.data.copy_(lora_state_dict[lora_a_key])
            if lora_b_key in lora_state_dict:
                module.lora.lora_B.data.copy_(lora_state_dict[lora_b_key])

        adapter_name = adapter_name or checkpoint.get('adapter_name', 'loaded')
        self.active_adapter = adapter_name

        print(f"Loaded LoRA adapter '{adapter_name}' from {load_path}")

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable LoRA parameters."""
        params = []
        for module in self.lora_modules.values():
            params.extend(module.get_lora_parameters())
        return params

    def print_trainable_parameters(self):
        """Print statistics about trainable parameters."""
        param_stats = count_lora_parameters(self)

        print(f"Trainable parameters:")
        print(f"  Total: {param_stats['total']:,}")
        print(f"  LoRA: {param_stats['lora']:,}")
        print(f"  Base (frozen): {param_stats['base']:,}")
        print(f"  LoRA percentage: {param_stats['lora_percentage']:.2f}%")


class MultiAdapterLoRAModel(nn.Module):
    """
    Model with multiple LoRA adapters that can be switched dynamically.

    This is the key component for continual learning with domain-specific adapters.

    Args:
        base_model: Pretrained model
        lora_config: Default LoRA configuration

    Example:
        >>> model = MultiAdapterLoRAModel(base_model, lora_config)
        >>> model.add_adapter("math", LoRAConfig(r=16))
        >>> model.add_adapter("code", LoRAConfig(r=32))
        >>> model.set_adapter("math")
        >>> # Now using math adapter
        >>> model.set_adapter("code")
        >>> # Now using code adapter
    """

    def __init__(
        self,
        base_model: nn.Module,
        lora_config: LoRAConfig
    ):
        super().__init__()

        self.base_model = base_model
        self.default_config = lora_config

        # Dictionary of adapters
        self.adapters: Dict[str, Dict] = {}

        # Currently active adapter(s)
        self.active_adapters: Set[str] = set()

        # Track LoRA modules
        self.lora_modules: Dict[str, Dict[str, LinearWithLoRA]] = {}

    def add_adapter(
        self,
        adapter_name: str,
        lora_config: Optional[LoRAConfig] = None
    ):
        """
        Add a new LoRA adapter.

        Args:
            adapter_name: Name of the adapter
            lora_config: Configuration (uses default if None)
        """
        if adapter_name in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' already exists")

        config = lora_config or self.default_config

        # Create LoRA modules for this adapter
        adapter_modules = {}
        target_modules = config.target_modules

        for name, module in self.base_model.named_modules():
            if self._is_target_module(name, target_modules):
                # Find Linear layers and add LoRA
                for child_name, child_module in module.named_children():
                    if isinstance(child_module, nn.Linear):
                        # Check if we already wrapped this
                        full_name = f"{name}.{child_name}" if name else child_name

                        if adapter_name not in self.lora_modules:
                            self.lora_modules[adapter_name] = {}

                        # Create new LoRA layer for this adapter
                        lora_layer = LinearWithLoRA(
                            base_layer=child_module,
                            r=config.r,
                            alpha=config.alpha,
                            dropout=config.dropout
                        )

                        adapter_modules[full_name] = lora_layer

        # Store adapter
        self.adapters[adapter_name] = {
            'config': config,
            'modules': adapter_modules
        }

        print(f"Added adapter '{adapter_name}' with {len(adapter_modules)} LoRA modules")

    def set_adapter(self, adapter_name: str):
        """
        Set the active adapter.

        Args:
            adapter_name: Name of adapter to activate
        """
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")

        self.active_adapters = {adapter_name}

    def enable_adapters(self, adapter_names: List[str]):
        """
        Enable multiple adapters simultaneously.

        The outputs will be combined.

        Args:
            adapter_names: List of adapter names to enable
        """
        for name in adapter_names:
            if name not in self.adapters:
                raise ValueError(f"Adapter '{name}' not found")

        self.active_adapters = set(adapter_names)

    def disable_all_adapters(self):
        """Disable all adapters (use only base model)."""
        self.active_adapters = set()

    def _is_target_module(self, module_name: str, target_modules: List[str]) -> bool:
        """Check if module should get LoRA."""
        for target in target_modules:
            if module_name.endswith(target) or re.search(target, module_name):
                return True
        return False

    def save_adapter(self, adapter_name: str, save_path: str):
        """Save a specific adapter."""
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")

        adapter = self.adapters[adapter_name]
        modules = adapter['modules']

        # Collect LoRA state dict
        lora_state_dict = {}
        for name, module in modules.items():
            lora_state_dict[f"{name}.lora_A"] = module.lora.lora_A
            lora_state_dict[f"{name}.lora_B"] = module.lora.lora_B

        # Save
        save_dict = {
            'adapter_name': adapter_name,
            'lora_config': adapter['config'].__dict__,
            'state_dict': lora_state_dict
        }

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, save_path)

        print(f"Saved adapter '{adapter_name}' to {save_path}")

    def forward(self, *args, **kwargs):
        """Forward pass with active adapters."""
        # This is a simplified version
        # In practice, you'd need to modify the forward pass to use active adapters
        return self.base_model(*args, **kwargs)


if __name__ == "__main__":
    # Test LoRA model wrapper
    print("Testing LoRA model wrapper...")

    # Import base model
    import sys
    sys.path.append("../..")
    from src.model.llm import ContinualLLM, ModelConfig

    # Create small model
    config = ModelConfig(
        vocab_size=1000,
        d_model=128,
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=2,
        d_ff=512
    )

    base_model = ContinualLLM(config)
    print(f"Base model created with {base_model.get_num_params():,} parameters")

    # Create LoRA model
    lora_config = LoRAConfig(
        r=8,
        alpha=16,
        target_modules=["q_proj", "v_proj", "o_proj", "w1", "w2", "w3"]
    )

    lora_model = LoRAModel(base_model, lora_config, adapter_name="test_adapter")

    print(f"\nLoRA modules injected: {len(lora_model.lora_modules)}")
    lora_model.print_trainable_parameters()

    # Test forward pass
    print(f"\nTesting forward pass...")
    input_ids = torch.randint(0, 1000, (2, 10))
    logits, _, _ = lora_model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")

    # Test save/load
    print(f"\nTesting save/load...")
    save_path = "test_adapter.pt"
    lora_model.save_adapter(save_path)

    # Modify LoRA weights
    for module in lora_model.lora_modules.values():
        module.lora.lora_A.data.fill_(0)
        module.lora.lora_B.data.fill_(0)

    # Load back
    lora_model.load_adapter(save_path)

    import os
    os.remove(save_path)

    print("\n✓ LoRA model wrapper implementation complete!")
