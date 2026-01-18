"""
Training Module

Implements the training loop with:
- Distributed Data Parallel (DDP) for multi-GPU training
- Mixed precision (FP16) for efficiency
- Gradient accumulation
- Logging to WandB and TensorBoard
- Checkpointing and early stopping
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_scheduler
from typing import Dict, Optional, Any, Tuple
import numpy as np

from .losses import WeightedBCELoss, FocalLoss
from .metrics import compute_metrics, effort_aware_metrics


class Trainer:
    """
    Training orchestrator for HiAttention-XAI model.
    
    Features:
    - Multi-GPU training with DDP
    - Mixed precision training
    - Gradient clipping and accumulation
    - WandB logging
    - Early stopping
    - Checkpoint saving
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1
    ):
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        
        # Setup model with DDP if distributed
        if world_size > 1:
            self.model = DDP(model.to(device), device_ids=[rank])
        else:
            self.model = model.to(device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        num_training_steps = len(train_loader) * config.get('num_epochs', 50)
        self.scheduler = get_scheduler(
            'cosine',
            optimizer=self.optimizer,
            num_warmup_steps=config.get('warmup_steps', 1000),
            num_training_steps=num_training_steps
        )
        
        # Setup loss function
        class_weights = config.get('class_weights', {'clean': 1.0, 'defective': 10.0})
        if config.get('loss_function') == 'focal':
            self.loss_fn = FocalLoss(gamma=config.get('focal_gamma', 2.0))
        else:
            self.loss_fn = WeightedBCELoss(
                pos_weight=torch.tensor(class_weights['defective'] / class_weights['clean'])
            ).to(device)
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.get('mixed_precision', True) else None
        
        # Logging
        self.use_wandb = config.get('use_wandb', False) and self.is_main
        if self.use_wandb:
            import wandb
            wandb.init(
                project=config.get('wandb_project', 'hiattention-xai'),
                config=config,
                name=config.get('experiment_name', 'training')
            )
        
        # Tracking
        self.global_step = 0
        self.best_val_metric = 0.0
        self.patience_counter = 0
        
        # Directories
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        if self.is_main:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate params for weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.get('weight_decay', 1e-5)
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.get('learning_rate', 1e-3),
            betas=tuple(self.config.get('betas', [0.9, 0.999])),
            eps=self.config.get('eps', 1e-8)
        )
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training history
        """
        num_epochs = self.config.get('num_epochs', 50)
        log_interval = self.config.get('log_interval', 100)
        eval_interval = self.config.get('eval_interval', 500)
        patience = self.config.get('eval_patience', 10)
        
        history = {
            'train_loss': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        if self.is_main:
            print(f"\n{'='*60}")
            print(f"Starting training for {num_epochs} epochs")
            print(f"Device: {self.device}, World size: {self.world_size}")
            print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train one epoch
            train_loss = self._train_epoch(epoch, log_interval, eval_interval)
            history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self.evaluate()
            history['val_metrics'].append(val_metrics)
            history['learning_rates'].append(self.scheduler.get_last_lr()[0])
            
            epoch_time = time.time() - epoch_start
            
            if self.is_main:
                print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.1f}s")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Metrics: {val_metrics}")
                
                # Log to wandb
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        **{f'val/{k}': v for k, v in val_metrics.items()}
                    })
            
            # Check for improvement
            current_metric = val_metrics.get('recall_at_top20', val_metrics.get('f1', 0))
            
            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.patience_counter = 0
                
                if self.is_main:
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"  New best model! Metric: {current_metric:.4f}")
            else:
                self.patience_counter += 1
                if self.is_main:
                    print(f"  No improvement. Patience: {self.patience_counter}/{patience}")
            
            # Early stopping
            if self.patience_counter >= patience:
                if self.is_main:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Save periodic checkpoint
            if self.is_main and (epoch + 1) % self.config.get('save_every_n_epochs', 5) == 0:
                self.save_checkpoint(epoch)
        
        return history
    
    def _train_epoch(
        self,
        epoch: int,
        log_interval: int,
        eval_interval: int
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(
                        token_ids=batch['token_ids'],
                        line_positions=batch['line_positions'],
                        preceding_mask=batch['preceding_mask'],
                        attention_mask=batch['attention_mask']
                    )
                    
                    loss = self.loss_fn(
                        outputs['defect_logits'].squeeze(-1),
                        batch['label'].float()
                    )
                
                # Backward with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('gradient_clip', 1.0)
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    token_ids=batch['token_ids'],
                    line_positions=batch['line_positions'],
                    preceding_mask=batch['preceding_mask'],
                    attention_mask=batch['attention_mask']
                )
                
                loss = self.loss_fn(
                    outputs['defect_logits'].squeeze(-1),
                    batch['label'].float()
                )
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('gradient_clip', 1.0)
                )
                
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.is_main and self.global_step % log_interval == 0:
                avg_loss = epoch_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        'step': self.global_step,
                        'train_loss_step': loss.item(),
                        'learning_rate': lr
                    })
        
        return epoch_loss / max(1, num_batches)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                outputs = self.model(
                    token_ids=batch['token_ids'],
                    line_positions=batch['line_positions'],
                    preceding_mask=batch['preceding_mask'],
                    attention_mask=batch['attention_mask']
                )
                
                loss = self.loss_fn(
                    outputs['defect_logits'].squeeze(-1),
                    batch['label'].float()
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                probs = outputs['defect_probability'].squeeze(-1).cpu().numpy()
                labels = batch['label'].cpu().numpy()
                
                all_preds.extend(probs)
                all_labels.extend(labels)
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = compute_metrics(all_preds, all_labels)
        effort_metrics = effort_aware_metrics(all_preds, all_labels)
        
        metrics.update(effort_metrics)
        metrics['val_loss'] = total_loss / max(1, num_batches)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }
        
        # Save latest
        path = os.path.join(self.checkpoint_dir, 'latest.pt')
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pt')
            torch.save(checkpoint, best_path)
        
        # Save periodic
        epoch_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch+1}.pt')
        torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, path: str):
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint['best_val_metric']
        
        return checkpoint['epoch']


def setup_distributed(rank: int, world_size: int):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()
