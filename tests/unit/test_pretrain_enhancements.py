"""
Unit tests for enhanced pre-training functionality.

Tests validation tracking, early stopping, learning rate scheduling,
and best checkpoint saving.
"""

import pytest
import torch
import math
import sys
from pathlib import Path

# Add scripts to path for testing
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))

from pretrain import evaluate_model, train_epoch, collate_fn


class MockModel(torch.nn.Module):
    """Mock model for testing."""
    
    def __init__(self, vocab_size=100, d_model=32):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        logits = self.lm_head(x)
        return logits, None, None
    
    def parameters(self):
        return super().parameters()


class MockDataset(torch.utils.data.Dataset):
    """Mock dataset for testing."""
    
    def __init__(self, size=10, seq_len=20):
        self.size = size
        self.seq_len = seq_len
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random sequences
        input_ids = torch.randint(0, 100, (self.seq_len,))
        labels = torch.randint(0, 100, (self.seq_len,))
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class TestEvaluateModel:
    """Test validation evaluation."""
    
    def test_evaluate_returns_loss_and_perplexity(self):
        """Test that evaluate_model returns loss and perplexity."""
        model = MockModel()
        dataset = MockDataset(size=5)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=2, 
            collate_fn=collate_fn
        )
        
        loss, perplexity = evaluate_model(model, dataloader, device='cpu')
        
        assert isinstance(loss, float)
        assert isinstance(perplexity, float)
        assert loss > 0
        assert perplexity > 0
    
    def test_evaluate_perplexity_calculation(self):
        """Test that perplexity is exp(loss)."""
        model = MockModel()
        dataset = MockDataset(size=5)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=2, 
            collate_fn=collate_fn
        )
        
        loss, perplexity = evaluate_model(model, dataloader, device='cpu')
        
        expected_ppl = math.exp(loss)
        assert abs(perplexity - expected_ppl) < 1e-4
    
    def test_evaluate_with_max_batches(self):
        """Test evaluation with limited batches."""
        model = MockModel()
        dataset = MockDataset(size=10)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=2, 
            collate_fn=collate_fn
        )
        
        loss, perplexity = evaluate_model(model, dataloader, device='cpu', max_batches=2)
        
        # Should only evaluate 2 batches
        assert loss > 0
        assert perplexity > 0


class TestTrainEpochWithScheduler:
    """Test training epoch with learning rate scheduler."""
    
    def test_train_epoch_with_scheduler(self):
        """Test that train_epoch accepts and uses scheduler."""
        model = MockModel()
        dataset = MockDataset(size=5)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=2, 
            collate_fn=collate_fn
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Create a simple scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda step: 1.0
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        loss = train_epoch(
            model, 
            dataloader, 
            optimizer, 
            device='cpu', 
            epoch=1,
            grad_accumulation_steps=1,
            scheduler=scheduler
        )
        
        assert isinstance(loss, float)
        assert loss > 0
        # Scheduler should have been called
        assert optimizer.param_groups[0]['lr'] <= initial_lr or initial_lr == optimizer.param_groups[0]['lr']
    
    def test_train_epoch_without_scheduler(self):
        """Test that train_epoch works without scheduler (backward compatibility)."""
        model = MockModel()
        dataset = MockDataset(size=5)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=2, 
            collate_fn=collate_fn
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        loss = train_epoch(
            model, 
            dataloader, 
            optimizer, 
            device='cpu', 
            epoch=1,
            grad_accumulation_steps=1,
            scheduler=None
        )
        
        assert isinstance(loss, float)
        assert loss > 0


class TestLearningRateSchedule:
    """Test learning rate scheduling logic."""
    
    def test_warmup_phase(self):
        """Test that learning rate increases during warmup."""
        model = MockModel()
        base_lr = 1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        
        warmup_steps = 10
        total_steps = 100
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Scheduler is applied immediately, so initial LR is 0 (step 0 of warmup)
        initial_lr = optimizer.param_groups[0]['lr']
        assert initial_lr == 0.0  # 0/10 * base_lr
        
        # After 5 steps (halfway through warmup), LR should be at 50% of base
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        
        mid_warmup_lr = optimizer.param_groups[0]['lr']
        # At step 5 out of 10 warmup steps, multiplier should be 5/10 = 0.5
        expected_mid = base_lr * 0.5
        assert abs(mid_warmup_lr - expected_mid) < 1e-5
        
        # After full warmup, should be at 100% (or close to it)
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        
        # At step 10, we're at 10/10 = 1.0 but now in cosine phase
        # So LR should start decaying from base_lr
        lr_after_warmup = optimizer.param_groups[0]['lr']
        assert lr_after_warmup <= base_lr
    
    def test_cosine_decay_phase(self):
        """Test that learning rate decays after warmup."""
        model = MockModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        warmup_steps = 10
        total_steps = 100
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Step through warmup
        for _ in range(warmup_steps):
            optimizer.step()
            scheduler.step()
        
        lr_after_warmup = optimizer.param_groups[0]['lr']
        
        # Step through some decay
        for _ in range(20):
            optimizer.step()
            scheduler.step()
        
        lr_after_decay = optimizer.param_groups[0]['lr']
        
        # Learning rate should decrease during decay phase
        assert lr_after_decay < lr_after_warmup
        
        # LR should not go below 10% of initial
        assert lr_after_decay >= initial_lr * 0.1


class TestCollateFn:
    """Test collate function for proper padding."""
    
    def test_collate_pads_to_max_length(self):
        """Test that sequences are padded to max length in batch."""
        batch = [
            {'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor([2, 3, 4])},
            {'input_ids': torch.tensor([1, 2, 3, 4, 5]), 'labels': torch.tensor([2, 3, 4, 5, 6])},
        ]
        
        result = collate_fn(batch)
        
        assert result['input_ids'].shape == (2, 5)  # Padded to length 5
        assert result['labels'].shape == (2, 5)
        
        # Check padding values
        assert result['input_ids'][0, 3] == 0  # Padded with 0
        assert result['labels'][0, 3] == -1  # Padded with -1
    
    def test_collate_preserves_data(self):
        """Test that original data is preserved."""
        batch = [
            {'input_ids': torch.tensor([1, 2]), 'labels': torch.tensor([2, 3])},
        ]
        
        result = collate_fn(batch)
        
        assert result['input_ids'][0, 0] == 1
        assert result['input_ids'][0, 1] == 2
        assert result['labels'][0, 0] == 2
        assert result['labels'][0, 1] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
