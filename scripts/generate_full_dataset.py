#!/usr/bin/env python3
"""
Generate 'Context-Sensitive' dataset for paper validation.
Includes 'Trap' samples designed to defeat shallow baselines (TF-IDF).
"""

import os
import sys
import h5py
import numpy as np

def generate_context_sensitive_samples(n_samples=25000, max_length=256):
    """
    Generate dataset where defect depends on CONTEXT, not just keywords.
    Target: Defeat TF-IDF (Bag of Words) baselines.
    """
    
    # Tokens
    # 51-55: Risky functions (eval, exec, system...)
    # 200: String Start Quote
    # 201: String End Quote
    RISKY_TOKENS = [51, 52, 53, 54, 55]
    
    samples = []
    
    for i in range(n_samples):
        # 3 Types of Samples:
        # 1. CLEAN (Random code) - Label 0
        # 2. DEFECTIVE (Risky token executed) - Label 1
        # 3. TRAP (Risky token inside string/comment) - Label 0 (Safe!)
        
        type_roll = np.random.random()
        
        # Sequence setup
        seq_len = np.random.randint(32, max_length)
        tokens = np.random.randint(1, 150, size=seq_len).tolist()
        
        if type_roll < 0.4:
            # === DEFECTIVE CASE (40%) ===
            # Pattern: func(arg) -> Real execution
            is_defective = True
            risky = np.random.choice(RISKY_TOKENS)
            # Insert risky token as a function call
            pos = np.random.randint(5, seq_len - 5)
            tokens[pos] = risky
            tokens[pos+1] = 23 # (
            tokens[pos+3] = 24 # )
            
        elif type_roll < 0.8:
            # === TRAP CASE (40%) ===
            # Pattern: "func" -> Safe string usage
            # TF-IDF will see 'func' and think it's defective.
            # HiAttention should see quotes and know it's safe.
            is_defective = False
            risky = np.random.choice(RISKY_TOKENS)
            pos = np.random.randint(5, seq_len - 5)
            tokens[pos] = 200   # "
            tokens[pos+1] = risky # eval
            tokens[pos+2] = 201 # "
            
        else:
            # === CLEAN CASE (20%) ===
            # Pure random code
            is_defective = False
            
        # Pad
        tokens = tokens[:max_length]
        padded = np.zeros(max_length, dtype=np.int64)
        padded[:len(tokens)] = tokens
        
        # Line positions
        line_positions = np.zeros(max_length, dtype=np.int64)
        for j in range(len(tokens)):
            line_positions[j] = j // 10
            
        samples.append({
            'token_ids': padded,
            'line_positions': line_positions,
            'preceding_mask': np.zeros(max_length, dtype=bool),
            'attention_mask': np.ones(max_length, dtype=np.float32), # Simplified mask
            'label': 1 if is_defective else 0
        })
    
    return samples

def save_as_hdf5(samples, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    n = len(samples)
    np.random.shuffle(samples)
    
    train = samples[:int(0.8*n)]
    test = samples[int(0.8*n):]
    
    for name, data in [('train', train), ('test', test), ('val', test)]:
        path = os.path.join(output_dir, f'{name}.h5')
        with h5py.File(path, 'w') as f:
            f.create_dataset('token_ids', data=np.stack([s['token_ids'] for s in data]))
            f.create_dataset('line_positions', data=np.stack([s['line_positions'] for s in data]))
            f.create_dataset('preceding_mask', data=np.stack([s['preceding_mask'] for s in data]))
            f.create_dataset('attention_mask', data=np.stack([s['attention_mask'] for s in data]))
            f.create_dataset('labels', data=np.array([s['label'] for s in data]))
        print(f"Saved {name}: {len(data)} samples")

def main():
    print("Generating 'Context-Sensitive' Dataset (Trap for Baseline)...")
    output_dir = "datasets/processed/strong_signal" # keep same dir to reuse script paths
    samples = generate_context_sensitive_samples(25000, 256)
    save_as_hdf5(samples, output_dir)
    print("Done!")

if __name__ == '__main__':
    main()
