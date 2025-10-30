# Continual Learning Mini-LLM Architecture
## Real-Time Adaptive Language Model with Zero Forgetting

---

## Executive Summary

**Goal:** Build a small, efficient LLM (~100M-500M parameters) that can:
- Learn continuously from user-provided data in real-time
- Update weights on-the-fly without forgetting previous knowledge
- Achieve quality comparable to large LLMs on specialized tasks
- Run efficiently on Apple Silicon (M1/M2/M3)

**This is NOT a traditional LLM.** This is a **continual learning system** that adapts in real-time.

---

## The Core Problem: Catastrophic Forgetting

### Traditional LLM Paradigm ❌
```
Pre-train (months) → Fine-tune (hours) → Deploy (static) → Forget old task when learning new
```

### Our Paradigm ✅
```
Base Model → Continuous Learning → Real-time Updates → NEVER FORGET
           ↓
    User Data Stream → Update Weights → Retain All Knowledge
```

### The Challenge
When neural networks learn new tasks, they **catastrophically forget** old tasks. This is the #1 problem we must solve.

**Solutions We'll Implement:**
1. **Elastic Weight Consolidation (EWC)** - Protect important weights
2. **Experience Replay** - Rehearse old examples while learning new ones
3. **LoRA Adapters** - Add new knowledge without modifying core weights
4. **Progressive Neural Networks** - Grow the network for new tasks
5. **Knowledge Distillation** - Compress large model knowledge into small model

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT STREAM                         │
│              (Documents, Code, Conversations)                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA PREPROCESSING                          │
│  • Tokenization  • Quality Filtering  • Chunking            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 EXPERIENCE REPLAY BUFFER                     │
│  • Store important examples  • Sample for rehearsal         │
│  • Prevent forgetting  • Balance old vs new                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    CONTINUAL LEARNER                         │
│  ┌──────────────┐    ┌─────────────┐    ┌───────────────┐ │
│  │  Base Model  │───▶│ LoRA Module │───▶│  EWC Module   │ │
│  │  (Frozen)    │    │  (Adapts)   │    │  (Protects)   │ │
│  └──────────────┘    └─────────────┘    └───────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 ADAPTER MANAGEMENT                           │
│  • Multiple specialized LoRA adapters                       │
│  • Domain-specific knowledge modules                        │
│  • Dynamic adapter selection & merging                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE ENGINE                           │
│  • Fast generation  • MPS optimized  • Multi-adapter        │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Small Base Model (100M-500M Parameters)

**Why Small?**
- Fast updates (seconds, not hours)
- Runs on consumer hardware
- Can be frequently updated without huge compute

**Architecture:**
```yaml
# Optimized for Apple Silicon MPS
d_model: 512          # Smaller than typical
num_layers: 12        # Fewer layers
num_query_heads: 8
num_kv_heads: 4       # GQA for efficiency
d_ff: 2048
vocab_size: 32000
max_seq_len: 2048     # Start with shorter context

# Estimated: ~200M parameters
```

**Training Strategy:**
```
1. Pre-train on general corpus (one-time)
2. [Optional] Distill from large LLM (GPT-4, Claude)
3. Deploy as frozen base
4. All adaptation via LoRA
```

---

### 2. LoRA (Low-Rank Adaptation) - CRITICAL COMPONENT

**Why LoRA is Perfect for Real-Time Learning:**
- Only updates ~0.1% of parameters
- Updates in seconds, not hours
- Can have multiple adapters for different domains
- Base model stays frozen (prevents forgetting core knowledge)

**Architecture:**
```python
class ContinualLoRAModel(nn.Module):
    """
    Base model + multiple LoRA adapters
    """
    def __init__(self, base_model, max_adapters=10):
        self.base_model = base_model  # Frozen
        self.adapters = nn.ModuleDict()  # Active adapters
        self.adapter_weights = {}  # Importance weights

    def add_adapter(self, name, task_type=None):
        """Add new LoRA adapter for new knowledge domain"""
        adapter = LoRAAdapter(
            r=16,  # Low rank
            alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj",
                          "o_proj", "gate_proj", "up_proj"]
        )
        self.adapters[name] = adapter

    def forward(self, x, active_adapters=None):
        """
        Forward pass through base + selected adapters
        """
        # 1. Base model forward (frozen)
        base_output = self.base_model(x)

        # 2. Apply adapter modifications
        if active_adapters:
            for adapter_name in active_adapters:
                adapter = self.adapters[adapter_name]
                base_output = base_output + adapter(x)

        return base_output
```

**Multi-Adapter Strategy:**
```
User Data: "Python code"
→ Activate: [base, coding_adapter, python_adapter]

User Data: "Medical document"
→ Activate: [base, medical_adapter]

User Data: "General conversation"
→ Activate: [base, chat_adapter]
```

---

### 3. Experience Replay Buffer - PREVENTS FORGETTING

**The Problem:**
When you train on new data, the model forgets old data.

**The Solution:**
Keep a buffer of important past examples. When training on new data, also train on replayed old data.

```python
class ExperienceReplayBuffer:
    """
    Stores important past examples to prevent forgetting
    """
    def __init__(self, max_size=10000, strategy="reservoir"):
        self.buffer = []
        self.max_size = max_size
        self.strategy = strategy

    def add(self, example, importance=1.0):
        """
        Add example to buffer

        Importance scoring:
        - User explicitly marked as important: 1.0
        - High loss (difficult example): 0.8
        - Diverse/unique example: 0.7
        - Recent example: 0.5
        - Random example: 0.3
        """
        if len(self.buffer) < self.max_size:
            self.buffer.append((example, importance))
        else:
            # Replace least important example
            self._replace_strategy(example, importance)

    def sample(self, batch_size, new_data_ratio=0.5):
        """
        Sample batch: mix of new data + replay data

        Args:
            new_data_ratio: 0.5 = 50% new data, 50% replayed data
        """
        num_replay = int(batch_size * (1 - new_data_ratio))

        # Sample from buffer (importance-weighted)
        replay_samples = self._importance_sampling(num_replay)

        return replay_samples

    def _importance_sampling(self, n):
        """Sample based on importance scores"""
        examples, importances = zip(*self.buffer)
        probabilities = np.array(importances) / sum(importances)
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            p=probabilities,
            replace=False
        )
        return [self.buffer[i][0] for i in indices]
```

**Buffer Management Strategies:**
1. **Reservoir Sampling** - Random replacement with probability
2. **Importance-Based** - Keep high-loss/difficult examples
3. **Diversity-Based** - Keep examples covering different topics
4. **Hybrid** - Combine importance + diversity + recency

---

### 4. Elastic Weight Consolidation (EWC) - CRITICAL

**The Idea:**
Some weights are more important than others. Protect important weights from changing too much.

```python
class EWCLoss:
    """
    Elastic Weight Consolidation Loss
    Penalizes changes to important weights
    """
    def __init__(self, model, old_model, fisher_information):
        self.model = model
        self.old_params = {n: p.clone() for n, p in old_model.named_parameters()}
        self.fisher = fisher_information

    def penalty(self, lambda_ewc=1000):
        """
        EWC penalty term

        L_EWC = λ/2 * Σ F_i * (θ_i - θ_i*)²

        Where:
        - F_i = Fisher information (importance of weight i)
        - θ_i = current weight
        - θ_i* = old weight
        - λ = strength of regularization
        """
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.old_params and name in self.fisher:
                # Penalty for changing important weights
                loss += (self.fisher[name] *
                        (param - self.old_params[name]).pow(2)).sum()

        return lambda_ewc / 2 * loss

    @staticmethod
    def compute_fisher_information(model, dataloader):
        """
        Compute Fisher Information Matrix

        F_i = E[(∂log p(y|x,θ) / ∂θ_i)²]

        Measures how much weight i affects the output
        """
        fisher = {}
        model.eval()

        for batch in dataloader:
            model.zero_grad()
            output = model(batch)
            loss = F.cross_entropy(output.logits, batch.labels)
            loss.backward()

            # Accumulate squared gradients (Fisher approximation)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in fisher:
                        fisher[name] = param.grad.data.clone().pow(2)
                    else:
                        fisher[name] += param.grad.data.clone().pow(2)

        # Average over dataset
        for name in fisher:
            fisher[name] /= len(dataloader)

        return fisher
```

**How It Works:**
1. After learning task A, compute Fisher information (importance of each weight)
2. When learning task B, add EWC penalty to loss
3. Important weights for task A are protected
4. Model can still learn task B by adjusting less important weights

---

### 5. Streaming Data Pipeline

```python
class ContinualDataStream:
    """
    Handles continuous stream of user data
    """
    def __init__(self, replay_buffer, preprocessor):
        self.replay_buffer = replay_buffer
        self.preprocessor = preprocessor
        self.new_data_queue = queue.Queue()

    def ingest(self, raw_data, metadata=None):
        """
        User provides new data

        Args:
            raw_data: text, code, documents, etc.
            metadata: {"importance": 0.9, "domain": "python"}
        """
        # 1. Preprocess
        processed = self.preprocessor.process(raw_data)

        # 2. Quality check
        if not self._quality_check(processed):
            return

        # 3. Add to queue
        self.new_data_queue.put({
            'data': processed,
            'metadata': metadata or {},
            'timestamp': time.time()
        })

        # 4. Add to replay buffer
        importance = metadata.get('importance', 0.5)
        self.replay_buffer.add(processed, importance)

    def get_training_batch(self, batch_size=32):
        """
        Create training batch: new data + replayed data
        """
        # Get new data from queue
        new_data = []
        while len(new_data) < batch_size // 2 and not self.new_data_queue.empty():
            new_data.append(self.new_data_queue.get())

        # Get replay data from buffer
        replay_data = self.replay_buffer.sample(
            batch_size - len(new_data)
        )

        # Combine
        return new_data + replay_data
```

---

### 6. Continual Learning Trainer

```python
class ContinualLearner:
    """
    Main training loop for continual learning
    """
    def __init__(
        self,
        model,
        tokenizer,
        replay_buffer,
        device="mps"  # Apple Silicon
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.replay_buffer = replay_buffer
        self.device = device

        # Optimizers for different components
        self.adapter_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4
        )

        # EWC for preventing forgetting
        self.ewc = None
        self.fisher_information = None

    def update_online(self, new_data_batch):
        """
        Real-time update as new data arrives

        Steps:
        1. Sample replay data
        2. Combine with new data
        3. Compute loss + EWC penalty
        4. Update adapter weights
        5. Update replay buffer
        """
        # 1. Get replay samples
        replay_batch = self.replay_buffer.sample(
            batch_size=len(new_data_batch),
            new_data_ratio=0.5
        )

        # 2. Combine batches
        combined_batch = new_data_batch + replay_batch

        # 3. Forward pass
        self.model.train()
        outputs = self.model(combined_batch)

        # 4. Compute loss
        task_loss = F.cross_entropy(outputs.logits, combined_batch.labels)

        # 5. Add EWC penalty (if available)
        if self.ewc is not None:
            ewc_loss = self.ewc.penalty()
            total_loss = task_loss + ewc_loss
        else:
            total_loss = task_loss

        # 6. Backward + update
        self.adapter_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.adapter_optimizer.step()

        return {
            'task_loss': task_loss.item(),
            'ewc_loss': ewc_loss.item() if self.ewc else 0,
            'total_loss': total_loss.item()
        }

    def consolidate_knowledge(self, dataloader):
        """
        Periodically consolidate knowledge

        Call this after learning a significant amount of new data
        to update the Fisher information matrix
        """
        print("Consolidating knowledge...")

        # Compute Fisher information on current knowledge
        self.fisher_information = EWCLoss.compute_fisher_information(
            self.model,
            dataloader
        )

        # Create new EWC with updated Fisher
        self.ewc = EWCLoss(
            self.model,
            old_model=copy.deepcopy(self.model),
            fisher_information=self.fisher_information
        )

        print("Knowledge consolidated!")

    def save_checkpoint(self, path):
        """Save model + all adapters + replay buffer"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'adapters': {name: adapter.state_dict()
                        for name, adapter in self.model.adapters.items()},
            'replay_buffer': self.replay_buffer.buffer,
            'fisher_information': self.fisher_information
        }, path)
```

---

## Apple Silicon (MPS) Optimization

### Why MPS Matters
- Apple Silicon uses unified memory architecture
- MPS (Metal Performance Shaders) = GPU backend for PyTorch on Mac
- Different optimization strategies than CUDA

### Optimizations:
```python
class MPSOptimizedTrainer(ContinualLearner):
    """
    Optimizations specific to Apple Silicon
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # MPS setup
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Move model to MPS
        self.model.to(self.device)

        # Enable MPS-specific optimizations
        torch.mps.set_per_process_memory_fraction(0.8)  # Use 80% of memory

    def optimize_for_mps(self):
        """MPS-specific optimizations"""
        # 1. Use float16 instead of bfloat16 (MPS doesn't support bfloat16)
        self.use_amp = True
        self.scaler = torch.cuda.amp.GradScaler()  # Also works for MPS

        # 2. Smaller batch sizes (MPS is memory-constrained)
        self.batch_size = 16

        # 3. Gradient accumulation to simulate larger batches
        self.gradient_accumulation_steps = 4

        # 4. Disable some CUDA-only features
        self.use_flash_attention = False
```

---

## Knowledge Distillation from Large LLMs

**Goal:** Transfer knowledge from GPT-4/Claude/Llama-70B into our small model

```python
class KnowledgeDistillation:
    """
    Distill large LLM knowledge into small model
    """
    def __init__(self, student_model, teacher_api):
        self.student = student_model
        self.teacher = teacher_api  # API to GPT-4, Claude, etc.

    def generate_training_data(self, prompts, num_samples=1000):
        """
        Generate training data from teacher model

        1. User provides prompts/topics
        2. Teacher generates responses
        3. Student learns from (prompt, response) pairs
        """
        training_data = []

        for prompt in tqdm(prompts, desc="Generating from teacher"):
            # Get teacher response
            teacher_response = self.teacher.generate(prompt)

            # Store as training example
            training_data.append({
                'prompt': prompt,
                'response': teacher_response,
                'source': 'teacher_distillation'
            })

        return training_data

    def distillation_loss(self, student_logits, teacher_logits, temperature=2.0):
        """
        Knowledge distillation loss

        L = α * KL(softmax(teacher/T) || softmax(student/T)) +
            (1-α) * CrossEntropy(student, labels)
        """
        # Soften probabilities
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)

        # KL divergence loss
        distill_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (temperature ** 2)

        return distill_loss
```

---

## Complete System Workflow

### Initial Setup (One-Time)
```bash
# 1. Create small base model
python scripts/pretrain_base.py --size small --params 200M

# 2. [Optional] Distill from large LLM
python scripts/distill_from_gpt4.py --teacher gpt-4 --student base_model

# 3. Initialize system
python scripts/init_continual_system.py
```

### Runtime (Continuous)
```python
# User workflow
continual_llm = ContinualLLM()

# User provides data
with open("my_documents.txt") as f:
    user_data = f.read()

# System ingests and learns in real-time
continual_llm.ingest_and_learn(
    data=user_data,
    importance=0.8,  # Important data
    domain="technical_docs"
)

# Use the model
response = continual_llm.generate("Based on the documents I provided, ...")

# Model has learned from user data while retaining all previous knowledge!
```

### Background Processes
```python
# Process runs continuously
while True:
    # 1. Check for new data
    if continual_llm.has_new_data():
        # 2. Create training batch (new + replay)
        batch = continual_llm.get_training_batch()

        # 3. Quick update (seconds)
        continual_llm.update_weights(batch)

    # 4. Periodically consolidate knowledge
    if continual_llm.should_consolidate():
        continual_llm.consolidate_knowledge()

    # 5. Save checkpoint
    if continual_llm.should_checkpoint():
        continual_llm.save_checkpoint()

    time.sleep(1)  # Check every second
```

---

## File Structure (Updated)

```
Custom_ML_Agent/
├── src/
│   ├── continual/                    # NEW: Continual learning
│   │   ├── __init__.py
│   │   ├── experience_replay.py     # Replay buffer
│   │   ├── ewc.py                   # Elastic Weight Consolidation
│   │   ├── continual_learner.py     # Main training loop
│   │   ├── adapter_manager.py       # Multi-LoRA management
│   │   └── knowledge_distillation.py
│   │
│   ├── model/                        # Core architecture
│   │   ├── attention.py             # GQA
│   │   ├── embeddings.py            # RoPE
│   │   ├── feedforward.py           # SwiGLU
│   │   ├── normalization.py         # RMSNorm
│   │   ├── transformer_block.py
│   │   └── llm.py                   # Base model
│   │
│   ├── lora/                         # LoRA implementation
│   │   ├── __init__.py
│   │   ├── lora_layer.py
│   │   ├── lora_model.py
│   │   └── lora_config.py
│   │
│   ├── data/
│   │   ├── streaming_pipeline.py    # Real-time data ingestion
│   │   ├── preprocessing.py
│   │   └── quality_filter.py
│   │
│   └── utils/
│       ├── mps_optimization.py      # Apple Silicon optimizations
│       └── memory_management.py
│
├── scripts/
│   ├── pretrain_base.py             # One-time base model training
│   ├── distill_from_llm.py          # Knowledge distillation
│   ├── run_continual_learning.py    # Main runtime script
│   └── test_forgetting.py           # Test for catastrophic forgetting
│
└── configs/
    ├── base_model_small.yaml        # 100M-200M params
    ├── continual_learning.yaml      # CL hyperparameters
    └── mps_optimized.yaml           # Apple Silicon config
```

---

## Implementation Priority (REVISED)

### Phase 1: Core Model (Week 1)
**Status:** Partially done ✅
1. ✅ Basic transformer
2. 🔄 Add RoPE, GQA, SwiGLU, RMSNorm
3. 🔄 Apple Silicon (MPS) compatibility
4. 🔄 Small model configuration (~200M params)

### Phase 2: LoRA Foundation (Week 1-2)
**Status:** CRITICAL PATH 🔥
1. 🔄 Implement LoRA layers
2. 🔄 Multi-adapter management
3. 🔄 Adapter merging/switching
4. 🔄 Fast adapter training

### Phase 3: Continual Learning (Week 2-3)
**Status:** CORE INNOVATION 🔥
1. 🔄 Experience replay buffer
2. 🔄 Elastic Weight Consolidation (EWC)
3. 🔄 Continual learning trainer
4. 🔄 Knowledge consolidation

### Phase 4: Streaming Pipeline (Week 3)
**Status:** USER-FACING 🔥
1. 🔄 Real-time data ingestion
2. 🔄 Background training loop
3. 🔄 Checkpoint management
4. 🔄 User API

### Phase 5: Testing & Optimization (Week 4)
**Status:** VALIDATION
1. 🔄 Test for catastrophic forgetting
2. 🔄 Benchmark update speed
3. 🔄 Memory profiling
4. 🔄 MPS optimization

### Phase 6: Knowledge Distillation (Week 5)
**Status:** QUALITY BOOST
1. 🔄 Distillation from GPT-4/Claude API
2. 🔄 Synthetic data generation
3. 🔄 Quality filtering

---

## Success Metrics (REVISED)

### 1. Forgetting Metric ⭐ MOST CRITICAL
```python
# Test: Learn task A, then task B
# Measure: Performance on A after learning B

forgetting_rate = (accuracy_A_before - accuracy_A_after) / accuracy_A_before

Target: < 5% forgetting (95% retention)
```

### 2. Update Speed
```
Target: < 10 seconds to update on 1000 new examples
```

### 3. Memory Efficiency
```
Target: Full system < 8GB RAM on M1 Mac
```

### 4. Specialization Quality
```
Target: Match GPT-3.5 quality on specialized domain after seeing 10K examples
```

### 5. Adapter Efficiency
```
Target: < 50MB per domain adapter
```

---

## Key Insights & Research Papers

### Catastrophic Forgetting Solutions:
1. **EWC** (2017): "Overcoming catastrophic forgetting in neural networks"
2. **Experience Replay** (2013): From reinforcement learning
3. **Progressive Networks** (2016): Grow network for new tasks
4. **PackNet** (2018): Prune network to make room for new tasks

### Parameter-Efficient Fine-Tuning:
1. **LoRA** (2021): "Low-Rank Adaptation of Large Language Models"
2. **QLoRA** (2023): 4-bit quantization + LoRA
3. **AdaLoRA** (2023): Adaptive rank allocation

### Continual Learning for LLMs:
1. **LFPT5** (2022): Lifelong pretraining
2. **O-LoRA** (2023): Orthogonal LoRA for continual learning
3. **COPAL** (2023): Continual pretraining with adapters

---

## Next Steps

### Immediate Actions:
1. ✅ Understand the continual learning problem
2. 🔄 Start with Phase 1: Core model implementation
3. 🔄 Implement LoRA (Phase 2) - This is the foundation
4. 🔄 Build experience replay buffer
5. 🔄 Test basic continual learning on toy task

### Questions to Address:
1. Should we pre-train a base model or use an existing small model (TinyLlama, etc.)?
2. What should be the initial knowledge of the base model?
3. How to handle multi-domain data (code + docs + chat)?
4. UI/UX: How should users provide data? (File upload? API? Live monitoring?)

---

**This is a research-level system!** We're building something that doesn't fully exist yet in open-source. Let's start implementing! 🚀

**Document Version:** 1.0
**Last Updated:** October 2025
**Status:** Architecture Complete - Ready to Build
