#!/usr/bin/env python3
"""
Generate synthetic training data for quick demo/testing.
Creates a small dataset with synthetic code and defect labels.
"""

import os
import sys
import json
import h5py
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def generate_synthetic_code_samples(n_samples=1000, max_length=128):
    """Generate synthetic code-like token sequences."""
    
    # Simplified token vocabulary
    vocab = {
        'def': 1, 'class': 2, 'if': 3, 'else': 4, 'for': 5, 'while': 6,
        'return': 7, 'import': 8, 'from': 9, 'try': 10, 'except': 11,
        'with': 12, 'as': 13, 'in': 14, 'not': 15, 'and': 16, 'or': 17,
        'True': 18, 'False': 19, 'None': 20, 'self': 21, 'cls': 22,
        '(': 23, ')': 24, '[': 25, ']': 26, '{': 27, '}': 28, ':': 29,
        '=': 30, '+': 31, '-': 32, '*': 33, '/': 34, '<': 35, '>': 36,
        'print': 37, 'len': 38, 'range': 39, 'str': 40, 'int': 41,
        'list': 42, 'dict': 43, 'open': 44, 'read': 45, 'write': 46,
        'execute': 47, 'query': 48, 'connect': 49, 'close': 50,
        # Defect-related tokens (more likely in buggy code)
        'eval': 51, 'exec': 52, 'system': 53, 'shell': 54, 'password': 55,
        'secret': 56, 'unsafe': 57, 'TODO': 58, 'FIXME': 59, 'HACK': 60,
    }
    vocab_size = 100
    
    samples = []
    
    for i in range(n_samples):
        # Determine if this sample is defective
        is_defective = np.random.random() < 0.3  # 30% defect rate
        
        # Generate sequence length
        seq_len = np.random.randint(32, max_length)
        
        # Generate tokens
        if is_defective:
            # Include some "suspicious" tokens
            base_tokens = np.random.randint(1, 50, size=seq_len - 3)
            suspicious = np.random.choice([51, 52, 53, 54, 55, 56, 57, 58, 59, 60], size=3)
            tokens = np.concatenate([base_tokens, suspicious])
            np.random.shuffle(tokens)
        else:
            tokens = np.random.randint(1, 50, size=seq_len)
        
        # Pad to max_length
        padded = np.zeros(max_length, dtype=np.int64)
        padded[:len(tokens)] = tokens
        
        # Generate line positions
        line_positions = np.zeros(max_length, dtype=np.int64)
        current_line = 0
        for j in range(len(tokens)):
            line_positions[j] = current_line
            if np.random.random() < 0.1:  # New line
                current_line += 1
        
        # Preceding mask (lines before current)
        preceding_mask = np.zeros(max_length, dtype=bool)
        if len(tokens) > 10:
            preceding_mask[:10] = True
        
        # Attention mask
        attention_mask = np.zeros(max_length, dtype=np.float32)
        attention_mask[:len(tokens)] = 1.0
        
        samples.append({
            'token_ids': padded,
            'line_positions': line_positions,
            'preceding_mask': preceding_mask,
            'attention_mask': attention_mask,
            'label': 1 if is_defective else 0,
            'seq_len': len(tokens)
        })
    
    return samples


def save_as_hdf5(samples, output_dir):
    """Save samples as HDF5 files for train/val/test splits."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Split: 70% train, 15% val, 15% test
    n = len(samples)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    splits = {
        'train': samples[:train_end],
        'val': samples[train_end:val_end],
        'test': samples[val_end:]
    }
    
    for split_name, split_samples in splits.items():
        path = os.path.join(output_dir, f'{split_name}.h5')
        
        with h5py.File(path, 'w') as f:
            n_samples = len(split_samples)
            
            token_ids = np.stack([s['token_ids'] for s in split_samples])
            line_positions = np.stack([s['line_positions'] for s in split_samples])
            preceding_mask = np.stack([s['preceding_mask'] for s in split_samples])
            attention_mask = np.stack([s['attention_mask'] for s in split_samples])
            labels = np.array([s['label'] for s in split_samples])
            
            f.create_dataset('token_ids', data=token_ids)
            f.create_dataset('line_positions', data=line_positions)
            f.create_dataset('preceding_mask', data=preceding_mask)
            f.create_dataset('attention_mask', data=attention_mask)
            f.create_dataset('labels', data=labels)
        
        defect_count = sum(s['label'] for s in split_samples)
        print(f"  {split_name}: {n_samples} samples ({defect_count} defective)")


def main():
    print("=" * 60)
    print("Generating Synthetic Training Data")
    print("=" * 60)
    
    output_dir = "datasets/processed/synthetic"
    n_samples = 5000
    max_length = 128
    
    print(f"\nGenerating {n_samples} synthetic samples...")
    samples = generate_synthetic_code_samples(n_samples, max_length)
    
    print(f"\nSaving to {output_dir}/...")
    save_as_hdf5(samples, output_dir)
    
    # Save stats
    stats = {
        'total_samples': n_samples,
        'max_length': max_length,
        'defect_ratio': sum(s['label'] for s in samples) / n_samples,
        'vocab_size': 100
    }
    
    with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStats: {stats}")
    print("\n" + "=" * 60)
    print("Synthetic dataset created successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
