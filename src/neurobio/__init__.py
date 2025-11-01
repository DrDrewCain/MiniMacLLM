"""
Neurobiologically-Inspired Mechanisms for Continual Learning.

This module implements brain-inspired components that enhance
continual learning without catastrophic forgetting:

Phase 1 (Original):
- Neuromodulation: Dopamine, serotonin, acetylcholine control
- Homeostatic plasticity: Maintains neural activity
- Dendritic computation: Compartmentalized processing
- Hippocampal memory: Pattern separation and completion
- Sleep consolidation: Offline memory strengthening
- Autonomous learning: Emergent learning rate control
- Working memory: Prefrontal cortex-inspired temporary storage

Phase 2 (Advanced):
- Predictive coding: Hierarchical prediction errors
- Liquid dynamics: Adaptive time constants
- Spiking neurons: Event-driven processing (coming soon)
- Grid cells: Spatial reasoning (coming soon)
- Active inference: Free energy minimization (coming soon)
"""

from .neuromodulation import (
    NeuromodulationController,
    NeuromodulatorConfig
)
from .homeostasis import (
    HomeostaticNeuron,
    HomeostaticConfig,
    HomeostaticWrapper
)
from .dendritic import (
    DendriticLayer,
    DendriticConfig,
    DendriticMLP
)
from .hippocampal_memory import (
    HippocampalMemory,
    HippocampalConfig
)
from .sleep_consolidation import (
    SleepConsolidation,
    SleepConfig
)
from .autonomous_learning import (
    AutonomousLearningRateController,
    AutonomousLearningConfig,
    AdaptiveOptimizer
)
from .working_memory import (
    WorkingMemory,
    WorkingMemoryConfig
)
from .predictive_coding import (
    PredictiveCodingLayer,
    PredictiveCodingNetwork,
    PredictiveCodingConfig,
    compute_free_energy
)
from .liquid_dynamics import (
    LiquidTimeConstantCell,
    LiquidLayer,
    LiquidMLP,
    LiquidConfig
)

__all__ = [
    # Phase 1
    'NeuromodulationController',
    'NeuromodulatorConfig',
    'HomeostaticNeuron',
    'HomeostaticConfig',
    'HomeostaticWrapper',
    'DendriticLayer',
    'DendriticConfig',
    'DendriticMLP',
    'HippocampalMemory',
    'HippocampalConfig',
    'SleepConsolidation',
    'SleepConfig',
    'AutonomousLearningRateController',
    'AutonomousLearningConfig',
    'AdaptiveOptimizer',
    'WorkingMemory',
    'WorkingMemoryConfig',
    # Phase 2
    'PredictiveCodingLayer',
    'PredictiveCodingNetwork',
    'PredictiveCodingConfig',
    'compute_free_energy',
    'LiquidTimeConstantCell',
    'LiquidLayer',
    'LiquidMLP',
    'LiquidConfig',
]
