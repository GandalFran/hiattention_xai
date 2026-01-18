#!/usr/bin/env python3
"""
Quick training run on synthetic data for demonstration.
Uses smaller model config for faster execution.
"""

import os
import sys
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hiattention_xai.models import HiAttentionXAI
from hiattention_xai.training.losses import WeightedBCELoss
from hiattention_xai.training.metrics import compute_metrics


class SyntheticDataset(Dataset):
    """Load synthetic HDF5 data."""
    
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.token_ids = torch.tensor(f['token_ids'][:], dtype=torch.long)
            self.line_positions = torch.tensor(f['line_positions'][:], dtype=torch.long)
            self.preceding_mask = torch.tensor(f['preceding_mask'][:], dtype=torch.bool)
            self.attention_mask = torch.tensor(f['attention_mask'][:], dtype=torch.float)
            self.labels = torch.tensor(f['labels'][:], dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'token_ids': self.token_ids[idx],
            'line_positions': self.line_positions[idx],
            'preceding_mask': self.preceding_mask[idx],
            'attention_mask': self.attention_mask[idx],
            'label': self.labels[idx]
        }


def main():
    print("=" * 60)
    print("HiAttention-XAI Quick Training Demo")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Config (smaller for demo)
    config = {
        'vocab_size': 100,
        'embedding_dim': 64,
        'hidden_dim': 32,
        'num_attention_heads': 4,
        'gnn_hidden_dims': [64, 64],
        'dropout': 0.3,
        'use_pretrained': False  # Use simple embeddings for demo
    }
    
    # Load data
    print("\nLoading data...")
    data_dir = "datasets/processed/synthetic"
    train_ds = SyntheticDataset(os.path.join(data_dir, 'train.h5'))
    val_ds = SyntheticDataset(os.path.join(data_dir, 'val.h5'))
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    
    # Model
    print("\nCreating model...")
    model = HiAttentionXAI(config).to(device)
    params = model.count_parameters()
    print(f"Parameters: {params['total']:,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = WeightedBCELoss(pos_weight=torch.tensor(3.0).to(device))
    
    # Training loop
    num_epochs = 5
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            token_ids = batch['token_ids'].to(device)
            line_positions = batch['line_positions'].to(device)
            preceding_mask = batch['preceding_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].float().to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                token_ids=token_ids,
                line_positions=line_positions,
                preceding_mask=preceding_mask,
                attention_mask=attention_mask
            )
            
            loss = loss_fn(outputs['defect_logits'].squeeze(-1), labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    token_ids=batch['token_ids'].to(device),
                    line_positions=batch['line_positions'].to(device),
                    preceding_mask=batch['preceding_mask'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )
                val_preds.extend(outputs['defect_probability'].squeeze(-1).cpu().numpy())
                val_labels.extend(batch['label'].numpy())
        
        import numpy as np
        metrics = compute_metrics(np.array(val_preds), np.array(val_labels))
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | "
              f"F1: {metrics['f1']:.3f} | AUC: {metrics['auc_roc']:.3f}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Validation Results")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_metrics': metrics
    }, 'checkpoints/demo_model.pt')
    print(f"\nModel saved to checkpoints/demo_model.pt")
    
    print("\n" + "=" * 60)
    print("Training Demo Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
